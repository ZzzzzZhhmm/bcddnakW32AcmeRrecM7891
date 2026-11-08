#!/usr/bin/env python3
"""Qualify one RMBench reset, camera observation and bounded qpos hold.

Run against a frozen copy of the official envs/task_config trees, with its
assets link pointing to the pinned asset store. This is an engineering check,
not a learned policy rollout, branch-restoration certificate or task-SR trial.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def source_identity(root):
    files = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for name in ("envs", "task_config")
        for p in sorted((root / name).rglob("*"))
        if p.is_file() and p.suffix in (".py", ".yml", ".yaml")
    }
    return {"files": files, "sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest()}


def resolved_config(root, task, seed):
    import yaml

    def load(path):
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    config = load(root / "task_config/demo_clean.yml")
    cameras = load(root / "task_config/_camera_config.yml")
    embodiments = load(root / "task_config/_embodiment_config.yml")
    names = config["embodiment"]
    if len(names) != 1:
        raise ValueError("this qualification only admits the official dual-arm configuration")
    robot_file = root / embodiments[names[0]]["file_path"]
    robot_config = load(robot_file / "config.yml")
    config.update(
        task_name=task, task_config="demo_clean", seed=seed, now_ep_num=0,
        eval_mode=True, is_test=True, dual_arm=True, dual_arm_embodied=True,
        left_robot_file=str(robot_file), right_robot_file=str(robot_file),
        left_embodiment_config=robot_config, right_embodiment_config=robot_config,
        save_data=False, collect_data=False, eval_video_log=False,
        eval_video_save_dir=None,
    )
    camera = cameras[config["camera"]["head_camera_type"]]
    config.update(head_camera_h=camera["h"], head_camera_w=camera["w"])
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmbench-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", required=True,
                        choices=("press_button", "put_back_block", "rearrange_blocks", "battery_try"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--commands", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.commands <= 32:
        raise ValueError("qualification requires 1..32 bounded hold commands")
    root = args.rmbench_root.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    initial = source_identity(root)
    (out / "source_manifest.json").write_text(json.dumps(initial, indent=2) + "\n")
    os.chdir(root)
    sys.path.insert(0, str(root))
    started = time.monotonic()
    env = None
    report = {
        "schema": "warm.rmbench-task-qualification.v1", "task": args.task,
        "seed": args.seed, "source_sha256": initial["sha256"],
        "status": "running", "full_state_restoration_qualified": False,
        "scope": "Simulator reset/camera/qpos-hold engineering only; no learned policy, independent labels or task success-rate estimate.",
    }
    try:
        import numpy as np
        from PIL import Image

        config = resolved_config(root, args.task, args.seed)
        (out / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
        task_class = getattr(importlib.import_module(f"envs.{args.task}"), args.task)
        env = task_class()
        print(f"stage=reset task={args.task} seed={args.seed}", flush=True)
        env.setup_demo(**config)
        report["reset_elapsed_seconds"] = time.monotonic() - started
        print("stage=initial_observation", flush=True)
        obs = env.get_obs()
        action = np.asarray(obs["joint_action"]["vector"], dtype=np.float64)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("expected a finite 14-dimensional official qpos command")
        np.save(out / "hold_action.npy", action, allow_pickle=False)

        def save_images(observation, suffix):
            images = {}
            for name, data in observation["observation"].items():
                if "rgb" not in data:
                    continue
                rgb = np.asarray(data["rgb"])
                if rgb.ndim != 3 or rgb.shape[2] != 3 or not np.isfinite(rgb).all():
                    raise ValueError(f"invalid RGB from {name}")
                np.save(out / f"{name}-{suffix}.npy", rgb, allow_pickle=False)
                Image.fromarray(rgb).save(out / f"{name}-{suffix}.png")
                images[name] = {"shape": list(rgb.shape), "dtype": str(rgb.dtype),
                                "std": float(rgb.std())}
            if len(images) < 3:
                raise ValueError("expected head and both wrist RGB cameras")
            return images

        report["initial_cameras"] = save_images(obs, "initial")
        report["snapshot_api_inventory"] = {
            "scene": [name for name in dir(env.scene) if "pack" in name or "state" in name],
            "physx_system": [name for name in dir(env.scene.physx_system)
                             if "pack" in name or "state" in name],
        }
        report["task_scalar_attributes"] = {
            k: v for k, v in vars(env).items() if type(v) in (int, float, bool, str)
        }
        print(f"stage=hold commands={args.commands}", flush=True)
        trace = []
        for i in range(args.commands):
            before = env.take_action_cnt
            env.take_action(action.copy(), action_type="qpos")
            trace.append({"index": i, "action_count_before": before,
                          "action_count_after": env.take_action_cnt,
                          "eval_success": bool(env.eval_success)})
        (out / "hold_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        report["final_cameras"] = save_images(env.get_obs(), "final")
        report.update(status="qualified_reset_camera_hold",
                      commands_executed=int(env.take_action_cnt),
                      final_eval_success=bool(env.eval_success))
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["source_sha256_after"] = source_identity(root)["sha256"]
        if report["source_sha256_after"] != initial["sha256"]:
            report.update(status="failed", error="official source changed during qualification")
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if env is not None and hasattr(env, "scene"):
            env.close_env(clear_cache=True)
    if report["status"] != "qualified_reset_camera_hold":
        raise RuntimeError(report.get("error", "qualification failed"))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
