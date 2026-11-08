#!/usr/bin/env python3
"""Time the official FastWAM deployment on an already frozen WARM cost cohort.

This is a recorded-observation compute reference, not matched-training SR.
The official baseline preprocessing and queue implementation are retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def reference_schedule(reference):
    """Reject incomplete references and retain every prescribed query."""
    def read(name):
        return json.loads((reference / name).read_text())
    manifest, summary = read("manifest.json"), read("summary.json")
    episodes, runtime = read("frozen_episodes.json"), read("runtime_args.json")
    rows = [json.loads(line) for line in (reference / "timings.jsonl").read_text().splitlines() if line]
    if summary["status"] != "complete" or manifest["precision"] != "bf16" or manifest["batch"] != 1:
        raise ValueError("requires a completed BF16 batch-one reference")
    if len(rows) != summary["queries"] + summary["warmup_queries"]:
        raise ValueError("incomplete reference timings")
    stride = int(manifest["replan_steps"])
    if stride != runtime["replan_steps"] or manifest["nfe"] != runtime["num_inference_steps"]:
        raise ValueError("reference inference budget disagrees")
    counts = []
    for episode in episodes:
        frames = [r["recorded_frame"] for r in rows if not r["warmup"] and
                  (r["dataset_index"], r["recorded_episode"]) ==
                  (episode["dataset_index"], episode["recorded_episode"])]
        if frames != list(range(0, len(frames) * stride, stride)) or not frames:
            raise ValueError("reference query schedule is incomplete or duplicated")
        counts.append(len(frames))
    warmup = summary["warmup_queries"]
    if not episodes or type(warmup) is not int or warmup < 1:
        raise ValueError("positive warmup and a nonempty frozen cohort required")
    expected = [(True, episodes[0], warmup)] + [(False, e, n) for e, n in zip(episodes, counts, strict=True)]
    keys = [(w, e["dataset_index"], e["recorded_episode"], q * stride)
            for w, e, n in expected for q in range(n)]
    if [(r["warmup"], r["dataset_index"], r["recorded_episode"], r["recorded_frame"]) for r in rows] != keys:
        raise ValueError("reference ordering or total count disagrees")
    return manifest, runtime, episodes, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    reference = args.reference_profile.resolve(strict=True)
    manifest, runtime, episodes, schedule = reference_schedule(reference)
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("outputs must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    from fastwam.memory.manifest import sha256_file, sha256_array
    from fastwam.research.evidence import append_record
    from experiments.robotwin.fastwam_policy.deploy_policy import get_model
    from scripts.profile_nonreal_online_replay import summarize

    started = time.perf_counter()
    if sha256_file(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("baseline checkpoint identity mismatch")
    if sha256_file(args.train_config) != manifest["train_config_sha256"]:
        raise ValueError("reference training configuration mismatch")
    cfg = OmegaConf.load(args.train_config)
    if cfg.data.val.episode_split != "dev" or cfg.data.val.is_training_set:
        raise ValueError("explicit DEV split required")
    stride = manifest["replan_steps"]
    dataset = BaseLerobotDataset(dataset_dirs=list(cfg.data.val.dataset_dirs),
        shape_meta=OmegaConf.to_container(cfg.data.val.shape_meta, resolve=True),
        obs_size=stride+1, action_size=stride, val_set_proportion=0,
        is_training_set=False, episode_catalog_path=str(cfg.data.val.episode_catalog_path),
        episode_split="dev", episode_task_allowlist=[manifest["task"]], strict_sample_loading=True)
    dataset._set_return_images(True)
    cached = {}
    for _warmup, episode, count in schedule:
        for q in range(count):
            frame = q*stride
            key = (episode["dataset_index"], episode["recorded_episode"], frame)
            if key in cached:
                continue
            sample = dataset[episode["start"]+frame]
            observed = (int(torch.as_tensor(sample["episode_index"]).reshape(-1)[0]),
                        int(torch.as_tensor(sample["frame_index"]).reshape(-1)[0]))
            if observed != key[1:] or bool(sample["action_is_pad"].any()):
                raise ValueError("frozen prefix identity/padding mismatch")
            if bool(sample["state_is_pad"][0]) or bool(sample["image_is_pad"][0]):
                raise ValueError("padded current observation")
            cameras = {}
            for source, target in [("cam_high", "head_camera"), ("cam_left_wrist", "left_camera"),
                                   ("cam_right_wrist", "right_camera")]:
                images = sample["images"][source]
                if images.dtype != torch.uint8 or tuple(images.shape) != (stride+1, 3, 240, 320):
                    raise ValueError("camera dtype/shape mismatch")
                cameras[target] = {"rgb": np.ascontiguousarray(images[0].permute(1, 2, 0).numpy())}
            observation = {"observation": cameras, "joint_action": {"vector":
                np.ascontiguousarray(sample["state"]["default"][0].numpy(), dtype=np.float32)}}
            cached[key] = (observation, str(sample["task"]))
    catalog_sha256 = sha256_file(cfg.data.val.episode_catalog_path)
    reference_data = json.loads((reference/"data_qualification.json").read_text())
    if reference_data["catalog_sha256"] != catalog_sha256 or reference_data["recorded_prefixes"] != len(cached):
        raise ValueError("reference DEV catalog/count mismatch")
    telemetry = [json.loads(line) for line in (reference/"policy_telemetry.jsonl").read_text().splitlines() if line]
    reference_queries = iter(row for row in telemetry if row.get("kind") == "replan")
    checked = 0
    for _warmup, episode, count in schedule:
        for q in range(count):
            observation, _instruction = cached[(episode["dataset_index"], episode["recorded_episode"], q*stride)]
            query = next(reference_queries)
            camera_hashes = {source:sha256_array(observation["observation"][target]["rgb"])
                            for source,target in [("cam_high","head_camera"), ("cam_left_wrist","left_camera"),
                                                  ("cam_right_wrist","right_camera")]}
            if query["frame_index"] != q*stride or query["raw_camera_sha256"] != camera_hashes:
                raise ValueError("baseline raw cameras differ from the timed WARM query")
            checked += 1
    if next(reference_queries, None) is not None:
        raise ValueError("extra reference policy queries")
    (output/"data_qualification.json").write_text(json.dumps({"status":"qualified",
        "recorded_prefixes":len(cached), "catalog_sha256":catalog_sha256,
        "raw_camera_matches_including_warmup":checked}, indent=2))
    (output/"frozen_episodes.json").write_text(json.dumps(episodes, indent=2))
    if args.data_only:
        return
    baseline_args = {key:runtime[key] for key in ("device", "mixed_precision", "dataset_stats_path",
        "sim_cfg_path", "action_horizon", "replan_steps", "num_inference_steps", "seed",
        "text_cfg_scale", "negative_prompt", "rand_device", "tiled", "sigma_shift")}
    baseline_args.update(ckpt_setting=str(args.checkpoint), sim_task="rmbench_fastwam_online_3cam384",
                         task_name=manifest["task"], timing_enabled=True,
                         evaluation_namespace=runtime["warm_evaluation_namespace"])
    (output/"runtime_args.json").write_text(json.dumps(baseline_args, indent=2))
    print("stage=load_official_fastwam", flush=True)
    tick = time.perf_counter()
    policy = get_model(baseline_args)
    torch.cuda.synchronize()
    report_manifest = {"checkpoint_sha256":args.checkpoint_sha256, "reference_profile":str(reference),
        "reference_manifest_sha256":sha256_file(reference/"manifest.json"), "task":manifest["task"],
        "gpu":torch.cuda.get_device_name(), "precision":"bf16", "batch":1,
        "nfe":policy.num_inference_steps, "horizon":policy.action_horizon, "replan_steps":policy.replan_steps,
        "policy_construction_seconds":time.perf_counter()-tick,
        "timing":"CUDA-synchronized official _fill_action_queue; excludes decoding and external evidence writing",
        "preprocessing":"Official baseline canvas [1,3,384,320]; same raw cameras, native baseline numerical preprocessing",
        "scope":"Release-checkpoint deployment compute reference only; no matched-training or task SR claim"}
    for key in ("nfe", "horizon", "replan_steps", "gpu"):
        if report_manifest[key] != manifest[key]:
            raise ValueError(f"baseline/reference {key} mismatch")
    (output/"manifest.json").write_text(json.dumps(report_manifest, indent=2))
    rows = []
    for warmup, episode, count in schedule:
        policy.reset()
        for q in range(count):
            frame = q*stride
            observation, instruction = cached[(episode["dataset_index"], episode["recorded_episode"], frame)]
            if policy.step_count != frame or policy.pending_actions:
                raise ValueError("baseline cursor/queue mismatch")
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            tick = time.perf_counter()
            policy._fill_action_queue(observation, instruction)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-tick
            prefix = np.stack(list(policy.pending_actions))
            if prefix.shape != (stride, 14) or not np.isfinite(prefix).all():
                raise ValueError("invalid generated baseline prefix")
            row = dict(dataset_index=episode["dataset_index"], recorded_episode=episode["recorded_episode"],
                recorded_frame=frame, warmup=warmup, end_to_end_s=elapsed,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                raw_camera_sha256={name:sha256_array(value["rgb"]) for name,value in observation["observation"].items()},
                environment_prefix_sha256=sha256_array(prefix))
            append_record(output/"timings.jsonl", row); rows.append(row)
            policy.pending_actions.clear(); policy.step_count += stride
        print(json.dumps(dict(episode=episode["recorded_episode"], warmup=warmup, queries=count)), flush=True)
    report = summarize(rows, expected_count=sum(n for w,e,n in schedule if not w))
    report.update(elapsed_s=time.perf_counter()-started, scope=report_manifest["scope"])
    (output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
