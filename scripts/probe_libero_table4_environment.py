#!/usr/bin/env python3
"""Bounded read-only LIBERO renderer/interface probe; not Table 4 evidence."""
import argparse
import json
import os
import time
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualify-branches", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "scope": "environment probe only", "backend": os.getenv("MUJOCO_GL")}
    env = None
    code = 2
    tick = time.perf_counter()
    try:
        import numpy as np
        import mujoco
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
        from fastwam.research.libero_egl import install_software_egl
        report["renderer_device_selection"] = install_software_egl()
        if args.require_cuda and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError("worker must have exactly one usable CUDA device")
        if args.require_cuda:
            torch.zeros(1, device="cuda:0")
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        suite = benchmark.get_benchmark_dict()[args.suite]()
        task = suite.get_task(args.task)
        env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file),
                                 camera_heights=256, camera_widths=256)
        env.seed(3407)
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(args.task)[0])
        for _ in range(5):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        cameras = {}
        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            arr = np.asarray(obs[key])
            if arr.shape != (256, 256, 3) or arr.dtype != np.uint8 or arr.std() < 1:
                raise ValueError(f"invalid rendered camera: {key}")
            from PIL import Image
            Image.fromarray(arr[::-1, ::-1]).save(args.output/f"{key}.png")
            cameras[key] = dict(shape=list(arr.shape), dtype=str(arr.dtype), std=float(arr.std()))
        # Record the deployed Python state interfaces for backend implementation.
        def inventory(obj):
            return {k: type(v).__module__+"."+type(v).__name__ for k, v in vars(obj).items()}
        report.update(status="passed", cameras=cameras, mujoco=mujoco.__version__,
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      task_description=task.language, env_state=inventory(env.env),
                      robot_state=[inventory(r) for r in env.robots],
                      controller_state=[inventory(r.controller) for r in env.robots],
                      sim_state=inventory(env.sim))
        if args.qualify_branches:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
            from fastwam.research.libero_branches import LiberoBranchBackend, qualify
            backend = LiberoBranchBackend(env, obs, encode=lambda observation: np.zeros((4, 768)),
                                          policy_fingerprint=lambda: "no_policy_in_environment_probe")
            report["branch_qualification"] = qualify(backend)
        code = 0
    except Exception as error:
        import traceback
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    finally:
        report["elapsed_seconds"] = time.perf_counter()-tick
        if env is not None:
            try:
                env.close()
            except Exception as error:
                report["close_error"] = repr(error)
        (args.output/"environment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
