#!/usr/bin/env python3
"""W02-A offline full-Action-DiT invariance check on frozen DEV prefixes.

This is engineering evidence, not task success or learned rejection. It uses
the strict checkpoint/dataset loaders and the production FastWAM inference
kernel, with factual cached history and no future/teacher tensors. It does not
qualify BoundOnlineStep retrieval or simulator restoration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def select_prefixes(records, tasks, episodes_per_task, prefixes_per_episode, horizon=32, stride=4):
    """Freeze equal-episode prefixes without scores, gates or success labels."""
    grouped = defaultdict(list)
    for index, (query, task) in enumerate(records):
        if task in tasks:
            grouped[(task, query.dataset_index, query.episode_index)].append((index, query))
    chosen = []
    for task in tasks:
        episodes = sorted(key for key in grouped if key[0] == task)
        if len(episodes) < episodes_per_task:
            raise ValueError(f"insufficient DEV episodes for {task}")
        for key in episodes[:episodes_per_task]:
            rows = grouped[key]
            length = max(q.frame_index for _, q in rows) + 1
            eligible = [(idx, q) for idx, q in rows if q.frame_index + horizon < length and q.frame_index % stride == 0]
            if len(eligible) < prefixes_per_episode:
                raise ValueError("episode has insufficient complete, distinct prefixes")
            # Equal internal quantiles: 1/(P+1), ..., P/(P+1), fixed before inference.
            for p in range(1, prefixes_per_episode + 1):
                idx, q = eligible[(len(eligible) - 1) * p // (prefixes_per_episode + 1)]
                chosen.append(dict(task=task, dataset_index=q.dataset_index, episode_index=q.episode_index,
                                   frame_index=q.frame_index, dataset_sample_index=idx))
    if len({(r['dataset_index'], r['episode_index'], r['frame_index']) for r in chosen}) != len(chosen):
        raise ValueError("duplicate frozen prefix")
    return chosen


def factual_context(sample, *, device, dtype):
    import torch
    from fastwam.models.warm import retrospection_model as rm
    mapping = {
        "query_context": "WARM_CURRENT_CONTEXT", "candidate_context": "WARM_CANDIDATE_CONTEXT",
        "candidate_actions": "WARM_CANDIDATE_MU", "candidate_start_proprio": "WARM_CANDIDATE_START_PROPRIO",
        "candidate_effect_pre": "WARM_CANDIDATE_EFFECT_PRE", "candidate_effect_delta": "WARM_CANDIDATE_EFFECT_DELTA",
        "candidate_timing": "WARM_CANDIDATE_TIMING", "candidate_support": "WARM_CANDIDATE_SUPPORT",
        "candidate_valid_mask": "WARM_CANDIDATE_MASK", "candidate_normalized_phase": "WARM_CANDIDATE_NORMALIZED_PHASE",
        "candidate_event_ordinal": "WARM_CANDIDATE_EVENT_ORDINAL", "episode_tokens": "WARM_EPISODE_TOKENS",
        "episode_mask": "WARM_EPISODE_MASK", "episode_action_summaries": "WARM_EPISODE_ACTION_SUMMARIES",
        "episode_action_mask": "WARM_EPISODE_ACTION_MASK", "episode_role_ids": "WARM_EPISODE_ROLE_IDS",
        "episode_relative_age": "WARM_EPISODE_RELATIVE_AGE", "episode_action_relative_age": "WARM_EPISODE_ACTION_RELATIVE_AGE",
    }
    values = {}
    for field, constant in mapping.items():
        value = sample[getattr(rm, constant)]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"factual sample field {field} must be tensor")
        values[field] = value.unsqueeze(0).to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
    return rm.RetrospectiveSourceContext(**values)


def perturb_candidates(context):
    """Synthetic finite payload replacement; preserve masks/history/token count."""
    values = {}
    for name in ("candidate_context", "candidate_actions", "candidate_start_proprio", "candidate_effect_pre", "candidate_effect_delta"):
        value = getattr(context, name)
        mask = context.candidate_valid_mask.reshape(*value.shape[:2], *([1] * (value.ndim - 2)))
        values[name] = (-1.7 * value + .125) * mask.to(value.dtype)
    return replace(context, **values)


def source_pair_metrics(reference, alternative):
    """Reject a confounded source-only intervention before reporting its effect."""
    import torch
    fixed = ("base_gaussian", "conditioning", "g", "alpha", "selected_index", "adapted_actions", "valid")
    for name in fixed:
        if not torch.equal(reference["arrays"][name], alternative["arrays"][name]):
            raise ValueError(f"source-only comparison changed fixed field: {name}")
    arrays = alternative["arrays"]
    return dict(mean_g=float(arrays["g"].float().mean()),
                mean_noise_scale_squared=float(arrays["source_noise_scale"].float().square().mean()),
                source_rms_delta=float((arrays["source"].float()-reference["arrays"]["source"].float()).square().mean().sqrt()))


def source_query(context, infer, output, query_id, identity, prefix, mode):
    """One immutable source mode per process; compare modes only after collection."""
    import numpy as np
    from fastwam.research.evidence import append_record, write_probe
    results, probes, times = {}, {}, {}
    tolerance = None
    for call in (("full", "repeat") if mode == "full" else (mode,)):
        results[call], times[call], probes[call] = infer(context)
        if call == "repeat":
            tolerance = max(1e-6, float((results["full"]-results["repeat"]).abs().max()))
            append_record(output / "events.jsonl", dict(kind="tolerance_frozen", query_id=query_id, tolerance=tolerance, **prefix))
    probe_path = write_probe(output / "probes", query_id + "-" + mode, probes[mode], identity)
    arrays = probes[mode]['arrays']
    row = dict(**prefix, query_id=query_id, mode=mode, mean_g=float(arrays['g'].float().mean()),
               mean_noise_scale_squared=float(arrays['source_noise_scale'].float().square().mean()), tolerance=tolerance,
               nonzero_gate=bool(arrays['g'].any()), inference_seconds=times[mode], probe=probe_path)
    append_record(output / "source_results.jsonl", row)
    with (output / f"{query_id}-source-actions.npz").open("xb") as handle:
        np.savez_compressed(handle, **{key: value.numpy() for key,value in results.items()})
    print(json.dumps(dict(query_id=query_id, task=prefix["task"], source_mode=mode, nonzero_gate=row["nonzero_gate"])), flush=True)
    return [row]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["press_button", "put_back_block"])
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--prefixes-per-episode", type=int, default=5)
    parser.add_argument("--nfe", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--experiment", choices=("null", "source"), default="null")
    parser.add_argument("--source-mode", choices=("full", "scale_only", "gaussian"), default="full")
    args = parser.parse_args()
    if min(args.episodes_per_task, args.prefixes_per_episode, args.nfe) < 1 or len(set(args.tasks)) != len(args.tasks):
        raise ValueError("positive budgets and distinct tasks required")
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("output must be outside the source checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      DIFFSYNTH_SKIP_DOWNLOAD="true", DIFFSYNTH_MODEL_BASE_PATH=str(args.asset_root.resolve()), WANDB_MODE="disabled")
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.runtime import _wrap_warm_candidate_dataset
    from fastwam.utils.misc import register_work_dir
    from fastwam.memory.manifest import sha256_file
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.research.evidence import append_record, canonical, write_probe

    if not torch.cuda.is_available():
        raise RuntimeError("one CUDA GPU required")
    register_work_dir(args.output)
    started = time.perf_counter()
    cfg = OmegaConf.load(args.train_config)
    if cfg.data.val.episode_split != "dev" or cfg.data.val.is_training_set:
        raise ValueError("probe requires explicit DEV split")
    cfg.data.val.episode_task_allowlist = args.tasks
    (args.output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    print("stage=build_dev_dataset", flush=True)
    dataset = _wrap_warm_candidate_dataset(instantiate(cfg.data.val), cfg.data.warm_candidates.val, expected_query_split="dev")
    selected = select_prefixes(dataset.sampling_query_records(), args.tasks, args.episodes_per_task, args.prefixes_per_episode)
    (args.output / "frozen_prefixes.json").write_bytes(canonical(selected))
    print(f"stage=load_model frozen_prefixes={len(selected)}", flush=True)
    torch.manual_seed(args.seed)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
    model.load_checkpoint(str(args.checkpoint))
    model.eval().requires_grad_(False)
    model.validate_validation_dataset(dataset)
    model.configure_research_probe(source_mode=args.source_mode, force_null=args.experiment == "null", capture=True)
    identity = dict(evidence_type="engineering", checkpoint_sha256=model._warm_loaded_checkpoint_sha256,
                    checkpoint_path=str(args.checkpoint), train_config_sha256=sha256_file(args.train_config),
                    bank_sha256=dataset.resolver.bank_content_sha256, query_corpus_sha256=dataset.resolver.query_corpus_sha256,
                    normalizer_sha256=sha256_file(Path(cfg.data.warm_candidates.val.normalization_stats_path)),
                    prefix_manifest_sha256=sha256_file(args.output / "frozen_prefixes.json"),
                    torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0), dtype="bfloat16", nfe=args.nfe,
                    source_path="FastWAM.infer_action with factual DEV context; no online binding or simulator",
                    absolute_tolerance_floor=1e-6, tolerance_rule="max(1e-6, identical-call max error), frozen before replacement",
                    experiment=args.experiment, source_mode=args.source_mode, setup_seconds=time.perf_counter()-started)
    (args.output / "manifest.json").write_bytes(canonical(identity))
    rows = []
    source_rows = []
    for number, prefix in enumerate(selected):
        begin = time.perf_counter()
        sample = dataset[prefix["dataset_sample_index"]]
        ctx = factual_context(sample, device=model.device, dtype=model.torch_dtype)
        changed = perturb_candidates(ctx)
        if not ctx.candidate_valid_mask.any() or torch.equal(ctx.candidate_actions, changed.candidate_actions):
            raise RuntimeError("frozen prefix cannot test candidate dependence; do not replace it selectively")
        kwargs = dict(prompt=None, input_image=sample["video"][:, 0], action_horizon=32,
                      proprio=sample["proprio"][0], context=sample["context"], context_mask=sample["context_mask"],
                      seed=args.seed + number, num_inference_steps=args.nfe, rand_device="cpu",
                      memory_sigma=model._require_retrospection().source_sigma_min)

        def infer(context):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.inference_mode():
                output = FastWAM.infer_action(model, action_source_context=context, **kwargs)
            torch.cuda.synchronize()
            action = output["action"]
            if not torch.isfinite(action).all() or tuple(action.shape) != (32, 14):
                raise RuntimeError("invalid full Action DiT output")
            return action, time.perf_counter()-tick, model._last_research_probe

        torch.cuda.reset_peak_memory_stats()
        if args.experiment == "source":
            source_rows.extend(source_query(ctx, infer, args.output, f"prefix-{number:04d}", identity, prefix, args.source_mode))
            continue
        base, t0, probe0 = infer(ctx)
        repeat, t1, probe1 = infer(ctx)
        tolerance = max(1e-6, float((base-repeat).abs().max()))
        query_id = f"prefix-{number:04d}"
        append_record(args.output / "events.jsonl", dict(kind="tolerance_frozen", query_id=query_id, tolerance=tolerance, **prefix))
        modified, t2, probe2 = infer(changed)
        error = float((base-modified).abs().max())
        path0 = write_probe(args.output / "probes", query_id + "-base", probe0, identity)
        path2 = write_probe(args.output / "probes", query_id + "-changed", probe2, identity)
        null_paths = all(torch.equal(p["arrays"]["source"], p["arrays"]["base_gaussian"]) and not bool(p["arrays"]["g"].any()) for p in (probe0, probe1, probe2))
        conditioning_error = float((probe0["arrays"]["conditioning"] - probe2["arrays"]["conditioning"]).abs().max())
        with (args.output / f"{query_id}-actions.npz").open("xb") as f:
            np.savez_compressed(f, base=base.numpy(), repeat=repeat.numpy(), changed=modified.numpy())
        row = dict(**prefix, query_id=query_id, passed=bool(error <= tolerance and null_paths and conditioning_error == 0),
                   identical_max_error=float((base-repeat).abs().max()), changed_max_error=error, tolerance=tolerance,
                   conditioning_max_error=conditioning_error, null_paths=null_paths, valid_candidates=int(ctx.candidate_valid_mask.sum()),
                   inference_seconds=[t0,t1,t2], query_seconds=time.perf_counter()-begin,
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3, base_probe=path0, changed_probe=path2)
        append_record(args.output / "query_results.jsonl", row)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ('query_id','task','passed','changed_max_error','query_seconds','peak_allocated_gib')}), flush=True)
    if args.experiment == "source":
        summary = dict(status="complete", evidence_type="engineering", queries=len(selected),
                       episodes=len({(r['task'],r['episode_index']) for r in selected}),
                       nonzero_gate_queries=sum(r['nonzero_gate'] for r in source_rows),
                       elapsed_seconds=time.perf_counter()-started,
                       modes={mode: {key: float(np.mean([r[key] for r in source_rows if r['mode'] == mode]))
                                     for key in ('mean_g', 'mean_noise_scale_squared')}
                              for mode in (args.source_mode,)},
                       limitation="one source mode collected; cross-mode pairing still required; no task SR")
        (args.output / "summary.json").write_bytes(canonical(summary))
        print(json.dumps(summary), flush=True)
        return 0
    summary = dict(status="passed" if all(r['passed'] for r in rows) else "failed", evidence_type="engineering",
                   queries=len(rows), episodes=len({(r['task'],r['episode_index']) for r in rows}),
                   max_error=max(r['changed_max_error'] for r in rows), elapsed_seconds=time.perf_counter()-started,
                   mean_query_seconds=float(np.mean([r['query_seconds'] for r in rows])),
                   median_inference_seconds=float(np.median([t for r in rows for t in r['inference_seconds']])),
                   p95_inference_seconds=float(np.quantile([t for r in rows for t in r['inference_seconds']],.95)),
                   peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
                   limitation="smoke-checkpoint offline prefix invariance; not learned rejection, online parity or task success")
    (args.output / "summary.json").write_bytes(canonical(summary))
    print(json.dumps(summary), flush=True)
    return 0 if summary['status'] == 'passed' else 2


if __name__ == "__main__":
    raise SystemExit(main())
