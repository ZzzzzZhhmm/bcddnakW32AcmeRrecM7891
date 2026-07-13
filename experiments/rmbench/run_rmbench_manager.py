"""Parallel, fail-closed manager for the exact RMBench official/pilot suites."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.benchmarks.rmbench import (  # noqa: E402
    RMBENCH_EPISODES_PER_TASK,
    RMBENCH_TASKS,
    assert_exact_task_sequence,
    task_by_name,
    tasks_for_suite,
    validate_hf_revision_marker,
    validate_protocol_pins,
    validate_read_only_checkout,
    write_manifest,
)


SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "rmbench" / "eval_rmbench_single.py"
POLL_SECONDS = 2.0
TERMINATE_TIMEOUT_SECONDS = 15.0


def _resolve_path(value: Any, *, base: Path) -> Path:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        raise ValueError("A required path is null or empty")
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _blocked_override(raw: str) -> bool:
    key = raw.split("=", 1)[0].lstrip("+~")
    protected = {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.output_dir",
        "EVALUATION.task_config",
        "EVALUATION.eval_num_episodes",
        "EVALUATION.code_revision",
        "EVALUATION.hf_dataset_revision",
        "EVALUATION.task_manifest_sha256",
    }
    return key in protected or key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _worker_overrides() -> list[str]:
    return [item for item in HydraConfig.get().overrides.task if not _blocked_override(item)]


@dataclass
class Worker:
    task_name: str
    gpu_id: int
    process: subprocess.Popen[str]


def _load_result(path: Path, *, expected_task: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Worker result missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("result"), dict):
        raise RuntimeError(f"Malformed worker result: {path}")
    result = payload["result"]
    if result.get("task_name") != expected_task:
        raise RuntimeError(
            f"Worker result task mismatch: expected {expected_task}, got {result.get('task_name')}"
        )
    if result.get("episodes") != RMBENCH_EPISODES_PER_TASK:
        raise RuntimeError(f"Worker result is not a 100-rollout RMBench result: {path}")
    seed_attestation = payload.get("accepted_seed_attestation")
    if (
        not isinstance(seed_attestation, dict)
        or not isinstance(seed_attestation.get("actual_accepted_seed_sha256"), str)
        or len(seed_attestation["actual_accepted_seed_sha256"]) != 64
        or seed_attestation.get("not_accepted_seed_claim") is not True
    ):
        raise RuntimeError(f"Worker result has no valid accepted-seed evidence: {path}")
    return payload


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty list")
    return float(sum(values) / len(values))


def _write_summary(
    output: Path,
    *,
    suite: str,
    ordered_task_names: list[str],
    results: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for task_name in ordered_task_names:
        payload = results.get(task_name)
        task = task_by_name(task_name)
        result = payload["result"] if payload is not None else None
        rows.append(
            {
                "task_name": task_name,
                "memory_regime": task.memory_regime,
                "step_limit": task.step_limit,
                "episodes": result.get("episodes") if result else None,
                "successes": result.get("successes") if result else None,
                "success_rate": result.get("success_rate") if result else None,
                "mean_reward": result.get("mean_reward") if result else None,
                "episode_records_sha256": (
                    result.get("episode_records_sha256") if result else None
                ),
                "actual_accepted_seed_sha256": (
                    payload.get("accepted_seed_attestation", {}).get(
                        "actual_accepted_seed_sha256"
                    )
                    if payload is not None
                    else None
                ),
            }
        )

    complete_rows = [row for row in rows if row["success_rate"] is not None]
    aggregates: dict[str, Any] = {
        "complete_tasks": len(complete_rows),
        "expected_tasks": len(ordered_task_names),
        "macro_success_rate": None,
        "micro_success_rate": None,
        "mean_reward": None,
        "by_memory_regime": {},
    }
    if complete_rows:
        aggregates["macro_success_rate"] = _mean(
            [float(row["success_rate"]) for row in complete_rows]
        )
        total_episodes = sum(int(row["episodes"]) for row in complete_rows)
        total_successes = sum(int(row["successes"]) for row in complete_rows)
        aggregates["micro_success_rate"] = total_successes / float(total_episodes)
        aggregates["mean_reward"] = _mean(
            [float(row["mean_reward"]) for row in complete_rows]
        )
        for regime in ("M(1)", "M(n)"):
            subset = [row for row in complete_rows if row["memory_regime"] == regime]
            if subset:
                aggregates["by_memory_regime"][regime] = {
                    "tasks": len(subset),
                    "macro_success_rate": _mean(
                        [float(row["success_rate"]) for row in subset]
                    ),
                }

    is_official_score = (
        suite == "official9"
        and not failures
        and len(complete_rows) == len(RMBENCH_TASKS)
    )
    run_identity: dict[str, Any] | None = None
    if results:
        identities = []
        for task_name in ordered_task_names:
            worker = results.get(task_name)
            if worker is None:
                continue
            identities.append(
                {
                    "policy_kind": worker.get("policy_kind"),
                    "policy_name": worker.get("policy_name"),
                    "root_seed": worker.get("seed"),
                    "checkpoint_sha256": worker.get("checkpoint_sha256"),
                    "warm_git_revision": worker.get("warm_git_revision"),
                    "action_generation": worker.get("action_generation"),
                }
            )
        canonical = {
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in identities
        }
        if len(canonical) != 1:
            raise RuntimeError(
                "RMBench workers do not share one policy/checkpoint/sampler identity"
            )
        run_identity = identities[0]

    payload = {
        "schema_version": 1,
        "benchmark": "RMBench",
        "suite": suite,
        "is_official_nine_task_score": is_official_score,
        "task_order": ordered_task_names,
        "run_identity": run_identity,
        "per_task": rows,
        "aggregates": aggregates,
        "failures": failures,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_rmbench.yaml")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must be provided")
    if not SINGLE_ENTRY.is_file():
        raise FileNotFoundError(f"Single-task entrypoint missing: {SINGLE_ENTRY}")

    suite = str(cfg.EVALUATION.suite).strip().lower()
    if suite in {"official", "full"}:
        suite = "official9"
    elif suite == "pilot":
        suite = "pilot3"
    selected = tasks_for_suite(suite)
    task_names = [task.name for task in selected]
    assert_exact_task_sequence(task_names, suite=suite)
    if cfg.EVALUATION.task_name is not None:
        raise RuntimeError(
            "The manager does not accept arbitrary task subsets. Use "
            "EVALUATION.suite=official9 or pilot3; invoke eval_rmbench_single.py "
            "directly for debugging one task."
        )

    validate_protocol_pins(
        code_revision=str(cfg.EVALUATION.code_revision),
        hf_revision=str(cfg.EVALUATION.hf_dataset_revision),
        manifest_sha256=str(cfg.EVALUATION.task_manifest_sha256),
        task_config=str(cfg.EVALUATION.task_config),
        episodes_per_task=int(cfg.EVALUATION.eval_num_episodes),
    )
    checkout = _resolve_path(cfg.EVALUATION.rmbench_root, base=PROJECT_ROOT)
    validate_read_only_checkout(
        checkout,
        expected_revision=str(cfg.EVALUATION.code_revision),
        require_push_disabled=bool(cfg.EVALUATION.require_push_disabled),
    )
    marker = _resolve_path(cfg.EVALUATION.hf_revision_marker, base=PROJECT_ROOT)
    validate_hf_revision_marker(
        marker,
        expected_revision=str(cfg.EVALUATION.hf_dataset_revision),
    )
    checkpoint = _resolve_path(cfg.ckpt, base=PROJECT_ROOT)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output = _resolve_path(cfg.EVALUATION.output_dir, base=PROJECT_ROOT)
    try:
        output.relative_to(checkout)
    except ValueError:
        pass
    else:
        raise RuntimeError("RMBench outputs must be outside the external checkout")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"Refusing to mix an RMBench run with existing output files: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output / "protocol_manifest.json", suite=suite)

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    max_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if num_gpus <= 0 or max_per_gpu <= 0:
        raise ValueError("MULTIRUN.num_gpus and max_tasks_per_gpu must be positive")
    gpu_ids = list(range(num_gpus))
    pending = deque(task_names)
    running: list[Worker] = []
    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    manager_log = output / "manager.log"
    forwarded = _worker_overrides()

    def log(message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def gpu_load(gpu_id: int) -> int:
        return sum(
            worker.gpu_id == gpu_id and worker.process.poll() is None for worker in running
        )

    def launch(task_name: str, gpu_id: int) -> Worker:
        command = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={checkpoint}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.output_dir={output}",
            *forwarded,
        ]
        log(f"launch task={task_name} gpu={gpu_id}")
        return Worker(
            task_name=task_name,
            gpu_id=gpu_id,
            process=subprocess.Popen(command, cwd=str(PROJECT_ROOT), text=True),
        )

    def fill(gpu_id: int) -> None:
        while pending and gpu_load(gpu_id) < max_per_gpu:
            running.append(launch(pending.popleft(), gpu_id))

    def terminate_all() -> None:
        alive = [worker for worker in running if worker.process.poll() is None]
        for worker in alive:
            worker.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SECONDS
        for worker in alive:
            try:
                worker.process.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait()

    log(f"start suite={suite} tasks={task_names} gpus={gpu_ids}")
    for gpu_id in gpu_ids:
        fill(gpu_id)

    fatal_error: str | None = None
    while running and fatal_error is None:
        progressed = False
        for worker in list(running):
            code = worker.process.poll()
            if code is None:
                continue
            progressed = True
            running.remove(worker)
            if code != 0:
                fatal_error = f"worker failed task={worker.task_name} gpu={worker.gpu_id} code={code}"
                failures.append(
                    {
                        "task_name": worker.task_name,
                        "gpu_id": worker.gpu_id,
                        "return_code": code,
                        "reason": "worker_failed",
                    }
                )
                log(fatal_error)
                terminate_all()
                break
            try:
                payload = _load_result(
                    output / worker.task_name / "result.json",
                    expected_task=worker.task_name,
                )
            except Exception as exc:
                fatal_error = f"result validation failed task={worker.task_name}: {exc!r}"
                failures.append(
                    {
                        "task_name": worker.task_name,
                        "gpu_id": worker.gpu_id,
                        "return_code": code,
                        "reason": "invalid_result",
                        "error": repr(exc),
                    }
                )
                log(fatal_error)
                terminate_all()
                break
            results[worker.task_name] = payload
            log(
                f"done task={worker.task_name} gpu={worker.gpu_id} "
                f"success_rate={payload['result']['success_rate']:.4f}"
            )
            fill(worker.gpu_id)
        if not progressed and fatal_error is None:
            time.sleep(POLL_SECONDS)

    if fatal_error is not None:
        completed = set(results)
        already_failed = {entry["task_name"] for entry in failures}
        for task_name in task_names:
            if task_name not in completed and task_name not in already_failed:
                failures.append(
                    {
                        "task_name": task_name,
                        "gpu_id": None,
                        "return_code": None,
                        "reason": "aborted_after_peer_failure",
                    }
                )

    _write_summary(
        output,
        suite=suite,
        ordered_task_names=task_names,
        results=results,
        failures=failures,
    )
    log(f"summary written: {output / 'summary.json'}")
    if fatal_error is not None:
        raise RuntimeError(fatal_error)
    if set(results) != set(task_names):
        raise RuntimeError(
            f"Completed task set mismatch: expected {task_names}, got {sorted(results)}"
        )
    log("RMBench suite finished successfully")


if __name__ == "__main__":
    main()
