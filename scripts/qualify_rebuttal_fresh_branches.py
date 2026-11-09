#!/usr/bin/env python3
"""Qualify sealed raw/refined branches by independent fresh-scene replay.

This engineering witness uses the first complete fixed Full source and q8.
It does not infer a new policy action, unpack physics, use a separately evolved
null rollout as a counterfactual, or establish an aggregate repair benefit.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("core-code", "rmbench-runtime", "source", "plan", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    core = args.core_code.resolve(strict=True)
    runtime = args.rmbench_runtime.resolve(strict=True)
    if out.is_relative_to(ROOT) or out.is_relative_to(core):
        raise ValueError("evidence must stay outside immutable source")
    out.mkdir(parents=True, exist_ok=False)

    import numpy as np
    from scripts.nonreal_job import source_identity
    from scripts.qualify_rebuttal_rmbench_task import resolved_config, source_identity as runtime_identity
    from fastwam.memory.robotwin_artifacts import RobotwinQposZScore
    from fastwam.research.branches import InvalidBranchExecution
    from fastwam.research.candidate_pool import decode_branch_actions
    from fastwam.research.evidence import append_record
    from fastwam.research.fresh_branches import OfficialQposGuard, require_repeated_endpoints, require_sealed_actions
    from fastwam.research.native_replay import native_parent, observation_arrays, require_observation, require_trace_step, state_hashes
    from fastwam.research.resumable_pairs import file_sha, read_json, read_roles
    from fastwam.research.rmbench_outcomes import observe_task_progress, label_next_interaction

    identity = read_json(args.source / "identity.json")
    plan = read_json(args.plan)
    query, episode = 8, 0
    if query not in plan["local_recovery"]["query_replan_ordinals"]:
        raise ValueError("q8 must belong to the existing fixed plan")
    prefix = plan["inference"]["execution_prefix"]
    if prefix != 4 or plan["inference"]["horizon"] != 32:
        raise ValueError("original H32/prefix4 protocol required")
    task = identity["task"]
    if task not in ("rearrange_blocks", "battery_try"):
        raise ValueError("only inspected native task implementations admitted")
    source = source_identity(core)
    sim = runtime_identity(runtime)
    trials = args.source / "role_trials.jsonl"
    if not trials.exists():
        trials = args.source / "pairs.jsonl"
    rows = [json.loads(s) for s in trials.read_text().splitlines() if s.strip()]
    binding = dict(task=task, checkpoint_sha256=plan["checkpoint_sha256"], evidence_plan_sha256=file_sha(args.plan),
        core_source_sha256=source["sha256"], runtime_source_sha256=sim["sha256"],
        official_step_limit={"rearrange_blocks":700,"battery_try":1000}[task])
    records = read_roles(args.source, binding, {r["episode_id"]:r["seed"] for r in rows})
    full, = [r for r in records if r["role"] == "full" and r["row"]["episode_id"] == episode]
    seed = full["row"]["seed"]
    case = Path(full["worker_evidence"])
    expert = read_json(case.parent / "expert_info.json")
    with np.load(case / "qpos_trace.npz", allow_pickle=False) as data:
        trace = {k:data[k].copy() for k in data.files}
    frame = query * prefix
    if frame + 32 > len(trace["commands"]):
        raise ValueError("fixed source does not reach the entire planned branch horizon")
    contract = read_json(Path(identity["contract_path"]))
    stats = Path(read_json(args.source / "full-runtime_args.json")["dataset_stats_path"])
    if file_sha(stats) != contract["normalization_stats_sha256"]:
        raise ValueError("original normalization statistics changed")
    normalizer = RobotwinQposZScore.from_dataset_stats(read_json(stats))
    probe_path = case / f"probes/e{episode:04d}-f{frame:04d}/proposal.npz"
    probe_metadata = read_json(probe_path.with_suffix(".json"))
    if file_sha(probe_path) != probe_metadata["array_sha256"] or probe_metadata["capture_phase"] != "before_any_branch_truth":
        raise ValueError("original proposal file no longer matches its pre-execution seal")
    with np.load(probe_path, allow_pickle=False) as probe:
        selected = int(probe["selected_index"].reshape(-1)[0])
        if selected < 0 or not bool(probe["valid"][0, selected]):
            raise ValueError("fixed source has no selected raw proposal; do not replace query")
        raw = decode_branch_actions(normalizer, probe["adapted_actions"][0, selected])
    generation_dir = case / f"generations/replan-{query:04d}"
    generation = read_json(generation_dir / "generation.json")
    if not generation["before_any_execution_of_this_chunk"] or generation["frame_index"] != frame:
        raise ValueError("generation was not sealed before original execution")
    if file_sha(generation_dir / "model_actions.npy") != generation["array_sha256"]:
        raise ValueError("original generation file differs from its pre-execution seal")
    if generation["checkpoint_sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("generation belongs to a different model")
    model_actions = np.load(generation_dir / "model_actions.npy", allow_pickle=False)
    refined = decode_branch_actions(normalizer, model_actions)
    commands = require_sealed_actions(raw, refined, trace, frame, prefix=prefix)
    np.savez_compressed(out / "sealed_native_actions.npz", **commands)
    files = [case / "qpos_trace.npz", case / "initial_identity.json", case / f"replan-{query:04d}.npz",
        probe_path, probe_path.with_suffix(".json"), generation_dir / "generation.json", generation_dir / "model_actions.npy",
        stats, Path(identity["contract_path"]), args.plan]
    report = dict(status="running", scope="Engineering fresh-scene A-A-B-A only; no aggregate repair, candidate-null, or policy success-rate claim.",
        task=task,seed=seed,episode_id=episode,query_ordinal=query,frame_index=frame,horizon=32,
        checkpoint_sha256=plan["checkpoint_sha256"], core_source_sha256=source["sha256"], runtime_source_sha256=sim["sha256"],
        collector_source_sha256=source_identity(ROOT)["sha256"], inputs=[dict(path=str(p),sha256=file_sha(p)) for p in files],
        sealed_actions_sha256=file_sha(out / "sealed_native_actions.npz"),
        candidate_null_branches=0,physical_unpack_calls=0,factual_policy_updates=0,full_scientific_qualification=False)
    (out / "identity.json").write_text(json.dumps(report, indent=2))
    os.chdir(runtime)
    sys.path[:0] = [str(runtime), str(runtime / "description/utils")]
    from generate_episode_instructions import generate_episode_descriptions
    cls = getattr(importlib.import_module("envs." + task), task)
    config = resolved_config(runtime, task, seed)
    config["now_ep_num"] = episode
    env = None
    completed = []
    started = time.monotonic()
    try:
        for index, role in enumerate(("refined", "refined", "raw", "refined")):
            env = cls()
            random.seed(seed)
            env.setup_demo(**config)
            if type(env.scene.physx_system).__name__ != "PhysxCpuSystem" or env.eval_video_path is not None or env.save_data or env.render_freq:
                raise ValueError("pinned CPU physics, no video, no data writes or interactive renderer required")
            instruction = str(np.random.choice(generate_episode_descriptions(task, [expert["info"]], 100)[0]["unseen"]))
            env.set_instruction(instruction=instruction)
            if instruction != full["initial_identity"]["instruction"]:
                raise ValueError("source instruction differs")
            with np.load(case / "initial_observation.npz", allow_pickle=False) as data:
                require_observation({k:data[k] for k in data.files}, observation_arrays(env.get_obs()))
            with OfficialQposGuard(env) as guard:
                for step, action in enumerate(trace["commands"][:frame]):
                    before = np.array(env.get_obs()["joint_action"]["vector"], copy=True)
                    if not guard.step(action):
                        raise RuntimeError("original replay prefix terminated before its sealed query")
                    after = np.array(env.get_obs()["joint_action"]["vector"], copy=True)
                    require_trace_step(trace, step, before, after)
            observation = env.get_obs()
            with np.load(case / f"replan-{query:04d}.npz", allow_pickle=False) as data:
                require_observation({k:data[k] for k in data.files}, observation_arrays(observation))
            parent = native_parent(env, observation)
            parent_hashes = state_hashes(parent)
            if completed and parent_hashes != completed[0]["parent_hashes"]:
                raise RuntimeError("fresh-scene branch parent differs before intervention")
            (out / f"branch-{index}-parent.pickle").write_bytes(pickle.dumps(parent, protocol=4))
            parent_outcome = observe_task_progress(env)
            count, rejection = 0, None
            with OfficialQposGuard(env) as guard:
                for step, action in enumerate(commands[role]):
                    before = np.array(env.get_obs()["joint_action"]["vector"], copy=True)
                    try:
                        executed = guard.step(action)
                    except InvalidBranchExecution as exc:
                        rejection = str(exc)
                        break
                    if not executed:
                        rejection = "official executor stopped before complete H32"
                        break
                    after = np.array(env.get_obs()["joint_action"]["vector"], copy=True)
                    append_record(out / "branch_commands.jsonl", dict(branch=index,role=role,step=step,
                        command=action.tolist(),official_target_before=before.tolist(),official_target_after=after.tolist()))
                    count += 1
                    if env.eval_success and count < 32:
                        rejection = "early success; endpoint retained, full H32 effect unavailable"
                        break
            endpoint = native_parent(env, env.get_obs())
            endpoint_outcome = observe_task_progress(env)
            row = dict(index=index,role=role,commands=count,valid=count==32 and rejection is None,rejection=rejection,
                parent_hashes=parent_hashes,endpoint_hashes=state_hashes(endpoint),
                parent_outcome=parent_outcome,endpoint_outcome=endpoint_outcome,
                independent_label=label_next_interaction(parent_outcome,endpoint_outcome,valid_branch=count==32 and rejection is None))
            (out / f"branch-{index}-endpoint.pickle").write_bytes(pickle.dumps(endpoint, protocol=4))
            append_record(out / "branches.jsonl", row)
            completed.append(row)
            print(json.dumps(dict(stage="fresh_branch_complete",index=index,role=role,commands=count,valid=row["valid"])),flush=True)
            env.close_env(clear_cache=True)
            env = None
        require_repeated_endpoints(completed)
        report.update(status="qualified_fresh_scene_AABA_engineering_only",branches_completed=4,
            exact_repeated_A_endpoints=3,distinct_physical_B=True,commands_per_branch=32)
    except BaseException as exc:
        report.update(status="failed_or_interrupted",error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        if env is not None and hasattr(env, "scene"):
            env.close_env(clear_cache=True)
        report.update(elapsed_seconds=time.monotonic()-started,branch_attempts=len(completed),
            core_source_sha256_after=source_identity(core)["sha256"],runtime_source_sha256_after=runtime_identity(runtime)["sha256"])
        (out / "summary.json").write_text(json.dumps(report,indent=2))
    if source["sha256"] != report["core_source_sha256_after"] or sim["sha256"] != report["runtime_source_sha256_after"]:
        raise RuntimeError("bound source changed")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
