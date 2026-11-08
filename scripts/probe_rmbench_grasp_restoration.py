#!/usr/bin/env python3
"""Exact restoration probe after the first official expert grasp macro.

This uses privileged simulator contact information only for engineering
qualification. No learned policy, applicability labels or success rates.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

from scripts.qualify_rebuttal_rmbench_task import resolved_config, source_identity


class ProbeFinished(Exception):
    pass


def entity_ids(actor):
    if hasattr(actor, "get_links"):
        return {link.entity.get_global_id() for link in actor.get_links()}
    entity = actor.entity if hasattr(actor, "entity") else actor
    return {entity.get_global_id()}


def measured_contacts(env):
    import numpy as np

    robot = entity_ids(env.robot.left_entity) | entity_ids(env.robot.right_entity)
    names = ("block1",) if env.task_name == "rearrange_blocks" else ("battery1", "battery2")
    targets = set().union(*(entity_ids(getattr(env, name).actor) for name in names))
    rows = []
    for contact in env.scene.physx_system.get_contacts():
        a, b = contact.bodies
        ids = (a.entity.get_global_id(), b.entity.get_global_id())
        if (ids[0] in robot and ids[1] in targets) or (ids[1] in robot and ids[0] in targets):
            rows.append({"body_names": [a.entity.name, b.entity.name], "entity_ids": ids,
                         "points": [{"separation": float(p.separation),
                                     "impulse_norm": float(np.linalg.norm(p.impulse))}
                                    for p in contact.points]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rmbench-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("rearrange_blocks", "battery_try"))
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    root = args.rmbench_root.resolve(strict=True)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    initial = source_identity(root)
    (out/"source_manifest.json").write_text(json.dumps(initial, indent=2)+"\n")
    os.chdir(root); sys.path.insert(0, str(root))
    started = time.monotonic(); env = None
    report = {"schema":"warm.rmbench-grasp-restoration-probe.v1", "task":args.task,
              "seed":args.seed, "status":"running", "full_scientific_qualification":False,
              "source_sha256":initial["sha256"], "learned_policy_trials":0}
    try:
        import numpy as np
        from PIL import Image
        from fastwam.research.rmbench_branches import RMBenchQposBackend, qualify_reset_probe, prepare_native_query

        config = resolved_config(root,args.task,args.seed)
        (out/"resolved_config.json").write_text(json.dumps(config,indent=2)+"\n")
        env = getattr(importlib.import_module(f"envs.{args.task}"),args.task)()
        env.setup_demo(**config)
        original_move = env.move
        macros = 0
        def move_and_probe(*actions, **kwargs):
            nonlocal macros
            result = original_move(*actions, **kwargs)
            macros += 1
            if macros != 1:
                return result
            if not env.plan_success:
                raise RuntimeError("official expert first grasp macro did not plan successfully")
            contacts = measured_contacts(env)
            report["robot_target_contacts"] = contacts
            if not any(p["impulse_norm"]>0 for c in contacts for p in c["points"]):
                raise RuntimeError("no measured robot-target contact after expert first grasp")
            preparation_directory=out/"query_preparation";preparation_directory.mkdir()
            preparation_count=0
            def save_preparation(record):
                nonlocal preparation_count
                (preparation_directory/f"{preparation_count:03d}.pickle").write_bytes(pickle.dumps(record,protocol=4))
                preparation_count+=1
            observation, canonicalization = prepare_native_query(env,emit=save_preparation)
            report["query_preparation"] = canonicalization
            for name, camera in observation["observation"].items():
                if "rgb" in camera:
                    np.save(out/f"{name}-parent.npy",camera["rgb"],allow_pickle=False)
                    Image.fromarray(camera["rgb"]).save(out/f"{name}-parent.png")
            backend = RMBenchQposBackend(env,observation,encode=lambda observation:np.zeros(1),
                policy_fingerprint=lambda:"no learned policy in expert-grasp engineering probe",canonicalization=canonicalization)
            traces = out/"restore_trace";traces.mkdir()
            count = 0
            def emit(record):
                nonlocal count
                (traces/f"{count:03d}-{record['kind']}.pickle").write_bytes(pickle.dumps(record,protocol=4))
                count += 1
            try:
                report["restoration"] = qualify_reset_probe(backend,horizon=32,context="expert_grasp",emit=emit)
            finally:
                backend.close()
            report.update(status="passed_one_expert_grasp_state_only", expert_macros=macros)
            raise ProbeFinished()
        env.move = move_and_probe
        try:
            env.play_once()
        except ProbeFinished:
            pass
        finally:
            del env.move
        if report["status"] != "passed_one_expert_grasp_state_only":
            raise RuntimeError("official expert did not reach the prespecified grasp checkpoint")
    except BaseException as exc:
        report.update(status="failed",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        report["source_sha256_after"] = source_identity(root)["sha256"]
        (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        if env is not None and hasattr(env,"scene"):
            env.close_env(clear_cache=True)
    if report["source_sha256"] != report["source_sha256_after"]:
        raise RuntimeError("official source changed during probe")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
