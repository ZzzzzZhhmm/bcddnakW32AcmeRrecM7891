"""Run one task through the pinned official RMBench evaluator.

This launcher owns protocol validation, output capture, and policy exposure.
The simulator and task implementation are always imported from an external,
pinned RMBench checkout.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.benchmarks.rmbench import (  # noqa: E402
    RMBENCH_EPISODES_PER_TASK,
    RMBENCH_HF_DATASET_REVISION,
    RMBENCH_INSTRUCTION_TYPE,
    RMBENCH_TASK_CONFIG,
    RMBENCH_TASK_MANIFEST_SHA256,
    discover_new_result_file,
    parse_official_result,
    prepare_runtime_overlay,
    snapshot_result_files,
    task_by_name,
    validate_hf_revision_marker,
    validate_policy_source,
    validate_protocol_pins,
    validate_read_only_checkout,
    validate_official_seed_namespace,
)


def _resolve_path(value: Any, *, base: Path, required: bool = True) -> Path | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        if required:
            raise ValueError("A required path is null or empty")
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _assert_output_is_external_to_checkout(output: Path, checkout: Path) -> None:
    try:
        output.relative_to(checkout)
    except ValueError:
        return
    raise RuntimeError(
        f"Evaluation output must not be written inside the external RMBench checkout: {output}"
    )


def _format_override(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _task_file_from_override(
    overrides: dict[str, Any],
    key: str,
    *,
    task_name: str,
    relative_candidates: tuple[str, ...],
) -> Path:
    value = overrides.pop(key, None)
    if value is None:
        raise ValueError(f"EVALUATION.policy_overrides.{key} is required")
    path = Path(str(value)).expanduser().resolve()
    candidates = (
        [path / relative for relative in relative_candidates]
        if path.is_dir()
        else [path]
    )
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Task-bound {key} for {task_name!r} must resolve exactly one file; "
            f"checked: {[str(candidate) for candidate in candidates]}"
        )
    return existing[0]


def _append_pair(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    command.extend([f"--{key}", _format_override(value)])


def _policy_overrides(cfg: DictConfig) -> dict[str, Any]:
    policy_kind = str(cfg.EVALUATION.get("policy_kind", "warm"))
    if policy_kind == "warm":
        node = cfg.EVALUATION.policy_overrides
        field = "EVALUATION.policy_overrides"
    elif policy_kind == "fastwam_baseline":
        node = cfg.EVALUATION.baseline_policy_overrides
        field = "EVALUATION.baseline_policy_overrides"
    else:
        raise ValueError(
            "EVALUATION.policy_kind must be 'warm' or 'fastwam_baseline'"
        )
    raw = OmegaConf.to_container(node, resolve=True)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError(f"{field} must be a mapping")
    resolved: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        key_text = str(key)
        if key_text.endswith("_path") or key_text.endswith("_directory"):
            path = _resolve_path(value, base=PROJECT_ROOT)
            assert path is not None
            resolved[key_text] = str(path)
        else:
            resolved[key_text] = value
    return resolved


def _git_head_or_none(root: Path) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_single_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = payload["result"]
    with (output_dir / "result.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "task_name",
                "memory_regime",
                "episodes",
                "successes",
                "success_rate",
                "mean_reward",
                "seed",
            ]
        )
        writer.writerow(
            [
                result["task_name"],
                payload["task"]["memory_regime"],
                result["episodes"],
                result["successes"],
                result["success_rate"],
                result["mean_reward"],
                payload["seed"],
            ]
        )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_rmbench.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must be provided")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("EVALUATION.task_name must be provided to the single-task runner")

    task = task_by_name(str(cfg.EVALUATION.task_name))
    validate_protocol_pins(
        code_revision=str(cfg.EVALUATION.code_revision),
        hf_revision=str(cfg.EVALUATION.hf_dataset_revision),
        manifest_sha256=str(cfg.EVALUATION.task_manifest_sha256),
        task_config=str(cfg.EVALUATION.task_config),
        episodes_per_task=int(cfg.EVALUATION.eval_num_episodes),
    )
    if str(cfg.EVALUATION.instruction_type) != RMBENCH_INSTRUCTION_TYPE:
        raise RuntimeError(
            f"Official protocol requires instruction_type={RMBENCH_INSTRUCTION_TYPE!r}"
        )

    checkout = _resolve_path(cfg.EVALUATION.rmbench_root, base=PROJECT_ROOT)
    assert checkout is not None
    checkout_attestation = validate_read_only_checkout(
        checkout,
        expected_revision=str(cfg.EVALUATION.code_revision),
        require_push_disabled=bool(cfg.EVALUATION.require_push_disabled),
    )
    marker = _resolve_path(
        cfg.EVALUATION.hf_revision_marker,
        base=PROJECT_ROOT,
    )
    assert marker is not None
    hf_attestation = validate_hf_revision_marker(
        marker,
        expected_revision=str(cfg.EVALUATION.hf_dataset_revision),
    )

    checkpoint = _resolve_path(cfg.ckpt, base=PROJECT_ROOT)
    assert checkpoint is not None
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    checkpoint_sha256 = _sha256_file(checkpoint)
    policy_source = _resolve_path(cfg.EVALUATION.policy_source, base=PROJECT_ROOT)
    assert policy_source is not None
    policy_name = str(cfg.EVALUATION.policy_name)
    policy_source = validate_policy_source(
        policy_source,
        policy_name=policy_name,
    )

    output_root = _resolve_path(cfg.EVALUATION.output_dir, base=PROJECT_ROOT)
    assert output_root is not None
    _assert_output_is_external_to_checkout(output_root, checkout)
    task_output = output_root / task.name
    task_output.mkdir(parents=True, exist_ok=True)

    runtime_root = task_output / "official_runtime"
    runtime_attestation = prepare_runtime_overlay(checkout, runtime_root)
    official_base = (
        runtime_root
        / "eval_result"
        / task.name
        / policy_name
        / RMBENCH_TASK_CONFIG
    )
    before = snapshot_result_files(official_base)
    launch_log = task_output / "official_stdout.log"

    command = [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        str(policy_source / "deploy_policy.yml"),
        "--overrides",
    ]
    policy_kind = str(cfg.EVALUATION.get("policy_kind", "warm"))
    policy_overrides = _policy_overrides(cfg)
    common_overrides: dict[str, Any] = {
        "task_name": task.name,
        "task_config": RMBENCH_TASK_CONFIG,
        "ckpt_setting": str(checkpoint),
        "seed": int(cfg.seed),
        "policy_name": policy_name,
        "instruction_type": RMBENCH_INSTRUCTION_TYPE,
        "eval_num_episodes": RMBENCH_EPISODES_PER_TASK,
    }
    seed_protocol: Path | None = None
    if policy_kind == "warm":
        experiment_id = str(
            policy_overrides.pop("warm_experiment_id", None) or "manual"
        )
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", experiment_id):
            raise ValueError(f"Unsafe warm_experiment_id: {experiment_id!r}")
        online_contract = _task_file_from_override(
            policy_overrides,
            "warm_online_contract_path",
            task_name=task.name,
            relative_candidates=(
                f"{experiment_id}/{task.name}.json",
                f"{task.name}.json",
            ),
        )
        seed_protocol = _task_file_from_override(
            policy_overrides,
            "warm_initial_states_path",
            task_name=task.name,
            relative_candidates=(
                f"seeds/{task.name}.seed_protocol.npy",
                f"{task.name}.seed_protocol.npy",
            ),
        )
        task_definition = (checkout / "envs" / f"{task.name}.py").resolve()
        if not task_definition.is_file():
            raise FileNotFoundError(
                f"Pinned RMBench task definition not found: {task_definition}"
            )
        required_overrides = {
            **common_overrides,
            "warm_telemetry_path": str(
                (task_output / "warm_online_evidence.jsonl").resolve()
            ),
            "warm_online_contract_path": str(online_contract),
            "warm_initial_states_path": str(seed_protocol),
            "warm_task_definition_path": str(task_definition),
            "warm_experiment_id": experiment_id,
        }
    elif policy_kind == "fastwam_baseline":
        forbidden = sorted(key for key in policy_overrides if key.startswith("warm_"))
        if forbidden:
            raise ValueError(
                f"FastWAM baseline received WARM-only policy overrides: {forbidden}"
            )
        seed_root = _resolve_path(
            cfg.EVALUATION.get("seed_protocol_root"), base=PROJECT_ROOT
        )
        assert seed_root is not None
        seed_protocol = (seed_root / "seeds" / f"{task.name}.seed_protocol.npy").resolve()
        if not seed_protocol.is_file():
            raise FileNotFoundError(
                f"FastWAM baseline seed namespace protocol is missing: {seed_protocol}"
            )
        required_overrides = common_overrides
    else:  # guarded by _policy_overrides; retain a local fail-closed boundary.
        raise ValueError(f"unsupported EVALUATION.policy_kind: {policy_kind!r}")
    for key, value in required_overrides.items():
        _append_pair(command, key, value)
    for key, value in sorted(policy_overrides.items()):
        if key in required_overrides:
            raise RuntimeError(f"Reserved policy override cannot be replaced: {key}")
        _append_pair(command, key, value)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    python_path = [str(PROJECT_ROOT / "src"), str(policy_source.parent)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    started_at = datetime.now(timezone.utc).isoformat()
    with launch_log.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=str(runtime_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_stream.write(line)
            log_stream.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Official RMBench evaluator failed with code {return_code}; log: {launch_log}"
        )
    post_checkout_attestation = validate_read_only_checkout(
        checkout,
        expected_revision=str(cfg.EVALUATION.code_revision),
        require_push_disabled=bool(cfg.EVALUATION.require_push_disabled),
    )
    if _sha256_file(checkpoint) != checkpoint_sha256:
        raise RuntimeError("Policy checkpoint changed during RMBench evaluation")

    official_result = discover_new_result_file(official_base, before)
    official_log = official_result.with_name("eval_log.txt")
    parsed = parse_official_result(
        official_result,
        task_name=task.name,
        expected_episodes=RMBENCH_EPISODES_PER_TASK,
        log_file=official_log,
    )
    assert seed_protocol is not None
    seed_attestation = validate_official_seed_namespace(
        official_log,
        seed_protocol,
        expected_episodes=RMBENCH_EPISODES_PER_TASK,
        expected_root_seed=int(cfg.seed),
    )
    actual_seeds = seed_attestation.pop("actual_accepted_seeds")
    accepted_seed_path = task_output / "actual_accepted_seeds.npy"
    with accepted_seed_path.open("xb") as stream:
        np.save(stream, actual_seeds, allow_pickle=False)

    # Preserve the raw evidence under WARM-owned output storage.
    (task_output / "official_result.txt").write_text(
        official_result.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (task_output / "official_episode_log.txt").write_text(
        official_log.read_text(encoding="utf-8"), encoding="utf-8"
    )
    payload = {
        "schema_version": 1,
        "benchmark": "RMBench",
        "official_protocol": True,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": {
            "name": task.name,
            "memory_regime": task.memory_regime,
            "step_limit": task.step_limit,
        },
        "seed": int(cfg.seed),
        "gpu_id": int(cfg.gpu_id),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "policy_kind": policy_kind,
        "policy_name": policy_name,
        "action_generation": {
            "action_horizon": int(
                required_overrides.get(
                    "action_horizon", policy_overrides.get("action_horizon")
                )
            ),
            "replan_steps": int(
                required_overrides.get(
                    "replan_steps", policy_overrides.get("replan_steps")
                )
            ),
            "num_inference_steps": int(
                required_overrides.get(
                    "num_inference_steps",
                    policy_overrides.get("num_inference_steps"),
                )
            ),
        },
        "accepted_seed_attestation": {
            **seed_attestation,
            "actual_accepted_seed_file": str(accepted_seed_path),
        },
        "warm_git_revision": _git_head_or_none(PROJECT_ROOT),
        "protocol": {
            "task_manifest_sha256": RMBENCH_TASK_MANIFEST_SHA256,
            "code_revision": checkout_attestation["head"],
            "hf_dataset_revision": RMBENCH_HF_DATASET_REVISION,
            "task_config": RMBENCH_TASK_CONFIG,
            "instruction_type": RMBENCH_INSTRUCTION_TYPE,
            "episodes_per_task": RMBENCH_EPISODES_PER_TASK,
        },
        "checkout_attestation": checkout_attestation,
        "post_run_checkout_attestation": post_checkout_attestation,
        "hf_attestation": hf_attestation,
        "policy_source": str(policy_source),
        "runtime_overlay": runtime_attestation,
        "result": parsed.to_dict(),
    }
    _write_single_outputs(task_output, payload)
    OmegaConf.save(cfg, str(task_output / "resolved_eval_config.yaml"))
    print(f"RMBench task result saved to {task_output / 'result.json'}")


if __name__ == "__main__":
    main()
