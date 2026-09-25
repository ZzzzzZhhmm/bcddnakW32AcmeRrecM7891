#!/usr/bin/env python3
"""Collect real LIBERO candidate branches with the checkpoint's legacy model.

This is a research collector, not a benchmark success-rate evaluator. All
candidate predictions and commands are sealed before counterfactual execution.
No retrieval or policy updates occur inside a branch. Existing evaluation
contracts supply immutable model/data identities; this collector writes its own
protocol because its sampling and outcome definitions differ from that rollout.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-code", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True,
                        help="Directory containing task_XX/seed_3407/formal-s3407-v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--episodes", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--frames", default="80,160")
    parser.add_argument("--cohort", default="libero10-table4-s3407-v1")
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--expected-checkpoint-sha256", default="")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # Import only the research tooling from this release. All trained model,
    # retrieval, processor and memory implementations come from the old bundle.
    release_src = Path(__file__).resolve().parents[1]/"src"
    sys.path.insert(0, str(release_src))
    import fastwam
    from fastwam.research.branches import execute_branches
    from fastwam.research.evidence import append_record, canonical, write_probe
    from fastwam.research.libero_branches import LiberoBranchBackend, digest, plain_fields, qualify
    from fastwam.research.libero_table4 import make_plan, goal_progress, to_environment_actions
    fastwam.__path__.insert(0, str(args.legacy_code/"src"/"fastwam"))
    for path in (args.legacy_code, args.legacy_code/"src", args.legacy_code/"experiments"/"libero"):
        sys.path.insert(0, str(path))
    os.chdir(args.legacy_code)
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(args.project/"checkpoints"))
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("HF_HOME", str(args.project/"cache"/"huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import yaml
    from fastwam.research.libero_legacy_compat import install
    preview_path = args.eval_root/f"task_{int(args.tasks.split(',')[0]):02d}"/"seed_3407"/"formal-s3407-v2"/"resolved_config.yaml"
    preview = yaml.safe_load(preview_path.read_text())
    compatibility = install(preview["EVALUATION"]["warm_online"]["encoder_contract_path"])
    from fastwam.research.libero_egl import install_software_egl
    compatibility["renderer_device_selection"] = install_software_egl()
    (args.output/"compatibility.json").write_bytes(canonical(compatibility))
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from fastwam.memory.manifest import sha256_file
    from fastwam.memory.online_retrieval import FrozenDinoOnlineRetriever
    from fastwam.memory.online_episode_memory import OnlineRetrospectiveEpisodeMemory
    from fastwam.models.warm.online_contract import WarmOnlineRunContract
    from fastwam.models.warm.source_contract import WarmSourceRunContract
    from fastwam.models.warm.retrospection_model import _mean_effect
    evaluator = importlib.import_module("eval_libero_single")
    ints = lambda value: tuple(int(x) for x in value.split(","))
    plan = make_plan(tasks=ints(args.tasks), episodes=ints(args.episodes), frames=ints(args.frames))
    if args.qualification_only:
        if plan["episodes"] != [49] or plan["tasks"] != [0] or plan["frames"] != [80]:
            raise ValueError("qualification uses reserved task0/init49/frame80, disjoint from formal resets")
    elif 49 in plan["episodes"]:
        raise ValueError("init49 is reserved for qualification")
    plan.update(cohort=args.cohort, qualification_only=args.qualification_only,
                legacy_code=str(args.legacy_code), env_backend=os.environ.get("MUJOCO_GL"),
                software_rendering=os.environ.get("LIBGL_ALWAYS_SOFTWARE"), max_hours=args.max_hours)
    (args.output/"plan.json").write_bytes(canonical(plan))
    suite = benchmark.get_benchmark_dict()[plan["suite"]]()
    paths = {t: args.eval_root/f"task_{t:02d}"/"seed_3407"/"formal-s3407-v2" for t in plan["tasks"]}
    configs = {t: OmegaConf.load(p/"resolved_config.yaml") for t, p in paths.items()}
    first = configs[plan["tasks"][0]]
    checkpoint = Path(str(first.ckpt)).resolve()
    print(f"table4_checkpoint={checkpoint}", flush=True)
    model = instantiate(first.model, model_dtype=torch.bfloat16, device="cuda:0")
    evaluator._load_model_checkpoint(model, str(checkpoint))
    model = model.to("cuda:0").eval()
    checkpoint_sha = model._warm_loaded_checkpoint_sha256
    if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("loaded checkpoint differs from the reused completed shard")
    stats, stats_sha = evaluator._load_dataset_stats_stable(Path(first.EVALUATION.dataset_stats_path))
    processor = instantiate(first.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    print(f"table4_model_ready elapsed={time.perf_counter()-started:.1f}", flush=True)
    capture = {}
    def hook(module, inputs, kwargs, output):
        capture.clear()
        capture.update(adapted_actions=output.adapted_action_mean.detach().clone(),
                       predicted_effect=output.predicted_effect.detach().clone(),
                       historical_effect=_mean_effect(kwargs["event_delta_tokens"]).detach().clone(),
                       valid=kwargs["candidate_valid_mask"].detach().clone())
    handle = model.retrospective_event_adapter.register_forward_hook(hook, with_kwargs=True)
    branch_times = []
    complete_queries = 0
    completed = False
    def log(name, row):
        append_record(args.output/f"{name}.jsonl", row)
    try:
        for task_id in plan["tasks"]:
            cfg = configs[task_id]
            oc = cfg.EVALUATION.warm_online
            contract = WarmOnlineRunContract.from_dict(json.loads((paths[task_id]/"online_contract.json").read_text()))
            if (Path(str(cfg.ckpt)).resolve() != checkpoint or contract.warm_checkpoint_sha256 != checkpoint_sha
                    or contract.normalization_stats_sha256 != stats_sha or contract.root_seed != 3407
                    or contract.task_suite != "libero_10" or contract.task_id != task_id
                    or contract.top_k != 32 or contract.action_horizon != 32):
                raise ValueError("model, statistics or task contract mismatch")
            if OmegaConf.to_container(cfg.model, resolve=True) != OmegaConf.to_container(first.model, resolve=True):
                raise ValueError("tasks disagree on model architecture")
            source = WarmSourceRunContract.from_dict(json.loads(Path(oc.training_run_contract_path).read_text()))
            validation = WarmSourceRunContract.from_dict(json.loads(Path(oc.validation_run_contract_path).read_text()))
            if source.sha256 != contract.training_run_contract_sha256 or validation.sha256 != contract.validation_run_contract_sha256:
                raise ValueError("source contracts mismatch")
            if sha256_file(Path(oc.training_attestation_path)) != contract.training_attestation_sha256:
                raise ValueError("training attestation differs from checkpoint contract")
            enc = Path(oc.encoder_contract_path)
            retriever = FrozenDinoOnlineRetriever.from_artifacts(
                oc.bank_directory, source_run_contract=source, online_run_contract=contract,
                normalizer_contract_path=oc.normalizer_contract_path, encoder_contract_path=enc,
                camera_contract_path=oc.camera_contract_path,
                normalization_stats_path=cfg.EVALUATION.dataset_stats_path,
                catalog_path=oc.catalog_path, audit_report_path=oc.audit_report_path,
                dino_checkpoint_path=oc.dino_checkpoint_path, processor=processor,
                dino_device="cuda", dino_torch_dtype=evaluator._dino_torch_dtype_from_encoder_contract(
                    enc, expected_sha256=contract.encoder_contract_sha256), dino_batch_size=1)
            model.bind_online_retriever(retriever)
            task = suite.get_task(task_id)
            bddl = Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
            if task.language != contract.task_description or sha256_file(bddl) != contract.bddl_sha256:
                raise ValueError("task language/BDDL differs from contract")
            memory = OnlineRetrospectiveEpisodeMemory(action_dim=7, action_horizon=32, semantic_dim=768,
                                                      gripper_indices=(6,), recent_event_capacity=6)
            runtime = evaluator.WarmOnlineEvalRuntime(
                contract=contract, pair_contract=None, pair_side="full_retrospection", source_contract=source,
                validation_source_contract=validation, retriever=retriever, retrospective_episode_memory=memory,
                null_image_adapter=None, normalization_stats_loaded_sha256=stats_sha,
                pair_contract_file_sha256=None, pair_contract_path=None, parity_report_file_sha256=None,
                parity_report_path=None, m1_data_config_path=Path(oc.m1_data_config_path).resolve(),
                training_attestation_file_sha256=sha256_file(Path(oc.training_attestation_path)),
                training_attestation_path=Path(oc.training_attestation_path),
                training_runtime_sha256=contract.training_runtime_sha256,
                shared_training_recipe_sha256=contract.shared_training_recipe_sha256,
                model=model, checkpoint_path=checkpoint, bddl_path=bddl, git_commit=contract.git_commit)
            def policy_fp():
                keys = ("_active_episode_index", "_last_frame_index", "_seen_episode_indices", "_issued",
                        "_inflight", "_consumed", "_delivered_step_digests", "_model_consumed")
                return digest(dict(memory=memory._memory.snapshot().sha256,
                                   runtime=plain_fields(runtime),
                                   retrieval={k: copy.deepcopy(getattr(retriever, k)) for k in keys}))
            def encode(obs):
                raw, _ = retriever._prepare_raw_cameras(evaluator.get_libero_image(obs))
                prepared = retriever._image_adapter.prepare(raw)
                features = retriever._dino.encode(prepared.camera_frames[retriever._semantic_processor_camera], batch_size=1)
                return np.asarray(features.spatial[0], dtype=np.float32)
            env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            task_qualified = False
            try:
                initial_states = suite.get_task_init_states(task_id)
                for episode in plan["episodes"]:
                    runtime.begin_episode(episode)
                    seed = evaluator._derive_episode_simulator_seed(3407, "libero_10", task_id, episode)
                    env.seed(seed)
                    evaluator.set_global_seed(seed, get_worker_init_fn=False)
                    env.reset()
                    obs = env.set_init_state(initial_states[episode])
                    for _ in range(30):
                        obs, _, done, _ = env.step(evaluator.get_libero_dummy_action())
                        if done:
                            raise RuntimeError("initial reset already terminal")
                    for frame in range(0, max(plan["frames"])+1, 10):
                        if time.perf_counter()-started > args.max_hours*3600:
                            raise TimeoutError("bounded experiment budget reached; retain partial evidence")
                        actions, _, _, telemetry = evaluator._predict_action_chunk(
                            obs, task.language, model, processor, cfg, action_horizon=32,
                            input_w=448, input_h=224, model_device="cuda:0", online_runtime=runtime, frame_index=frame)
                        if frame in plan["frames"]:
                            if len(capture) != 4:
                                raise RuntimeError("model hook did not capture exactly one real proposal batch")
                            query = f"libero10-task{task_id:02d}-init{episode:02d}-frame{frame:04d}"
                            capture["required_effect"] = model._last_retrospection_diagnostics["required_transition"].detach().clone()
                            proposal = write_probe(args.output/"probes", query, dict(arrays=capture),
                                dict(checkpoint_sha256=checkpoint_sha, task_description=task.language,
                                     event_ids=[c["event_id"] for c in telemetry["candidates"]],
                                     plan_sha256=sha256_file(args.output/"plan.json")))
                            log("index", dict(task=f"libero_10/{task_id}", episode_id=str(episode), query_id=query,
                                              cohort=args.cohort, probe_path=proposal["path"],
                                              probe_metadata_sha256=proposal["metadata_sha256"]))
                            backend = LiberoBranchBackend(env, obs, encode=encode, policy_fingerprint=policy_fp)
                            goals = copy.deepcopy(env.env.parsed_problem["goal_state"])
                            before = [bool(env.env._eval_predicate(g)) for g in goals]
                            if not task_qualified:
                                log("qualification", dict(query_id=query, **qualify(backend)))
                                first_read, second_read = encode(obs), encode(obs)
                                repeat_error = float(np.max(np.abs(first_read-second_read)))
                                if repeat_error > 1e-6:
                                    raise RuntimeError("DINO repeated-read error exceeds frozen tolerance")
                                log("qualification", dict(query_id=query, dino_repeat_max_abs=repeat_error))
                                task_qualified = True
                            env_commands = to_environment_actions(capture["adapted_actions"][0], processor,
                                evaluator, binarize=bool(cfg.EVALUATION.binarize_gripper))
                            valid = capture["valid"][0].cpu().numpy()
                            candidates = {f"rank-{rank}": env_commands[rank] for rank in np.flatnonzero(valid)}
                            def outcome():
                                after = [bool(env.env._eval_predicate(g)) for g in goals]
                                return dict(goals=goals, before=before, after=after,
                                            applicable=goal_progress(before, after),
                                            label_provenance=plan["outcome"])
                            def emit(row):
                                log("branches", row)
                                if row["kind"] == "branch_result" and "independent_outcome" in row:
                                    log("labels", dict(query_id=query, candidate_id=row["candidate_id"],
                                                       **row["independent_outcome"]))
                            tick = time.perf_counter()
                            execute_branches(backend, query_id=query, candidates=candidates,
                                proposal_sha256=proposal["metadata_sha256"],
                                project_effect=lambda pre, delta: delta.mean(axis=0), emit=emit,
                                observe_outcome=outcome)
                            elapsed = time.perf_counter()-tick
                            branch_times.append(elapsed)
                            complete_queries += 1
                            print(f"table4_query_done task={task_id} init={episode} frame={frame} candidates={len(candidates)} seconds={elapsed:.2f}", flush=True)
                            log("timing", dict(query_id=query, seconds=elapsed, candidates=len(candidates)))
                            obs = copy.deepcopy(backend.obs)
                        if frame == max(plan["frames"]):
                            break
                        for command in actions[:10]:
                            obs, _, done, _ = env.step(command)
                            runtime.note_executed_action(command, model_space_action=evaluator._executed_action_to_model_space(command, processor))
                            if done:
                                break
                        if done:
                            log("coverage", dict(task=task_id, episode=episode, terminated_at_or_before=frame+10,
                                                 skipped_queries=[f for f in plan["frames"] if f > frame]))
                            break
            finally:
                env.close()
        completed = True
    finally:
        handle.remove()
        summary = dict(completed=completed, complete_queries=complete_queries, elapsed_seconds=time.perf_counter()-started,
                       branch_query_seconds=branch_times, checkpoint_sha256=checkpoint_sha)
        (args.output/"execution.json").write_bytes(canonical(summary))
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
