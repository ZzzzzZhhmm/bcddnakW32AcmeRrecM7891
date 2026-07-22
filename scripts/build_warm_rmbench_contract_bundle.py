#!/usr/bin/env python3
"""Build the immutable online-contract bundle for a formal RMBench matrix.

Every matrix cell receives one contract per selected official task.  The
builder composes the exact deployed policy configuration, projects it through
the same canonical helper used at inference, and delegates the cryptographic
artifact checks to :mod:`scripts.build_warm_online_contract`.

The seed protocol records the deterministic *candidate namespace* used by the
official evaluator.  RMBench filters candidates through an expert setup loop,
so these values are intentionally not presented as the as-yet-unknown accepted
rollout seeds; accepted seeds are captured from the official episode log.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from fastwam.benchmarks.rmbench import (
    RMBENCH_CODE_REVISION,
    RMBENCH_EPISODES_PER_TASK,
    RMBENCH_TASKS,
    RMBENCH_TASK_MANIFEST_SHA256,
    tasks_for_suite,
    validate_read_only_checkout,
)
from fastwam.benchmarks.rmbench_runtime import (
    build_rmbench_policy_runtime_projection,
)
from fastwam.memory.manifest import sha256_array, sha256_canonical_json, sha256_file
from fastwam.models.warm.online_contract import WarmOnlineRunContract
from scripts import build_warm_online_contract as online_builder
from scripts.run_warm_rmbench_matrix import (
    DEFAULT_MATRIX,
    Experiment,
    load_matrix,
    select_experiments,
)


BUNDLE_SCHEMA = "warm.rmbench-online-contract-bundle"
BUNDLE_SCHEMA_VERSION = 1
SEED_PROTOCOL_SCHEMA = "warm.rmbench-root-seed-namespace"
SEED_PROTOCOL_VERSION = 1


class RMBenchBundleError(RuntimeError):
    """Raised before an incomplete or ambiguous formal bundle is published."""


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--suite", choices=("official9", "pilot3"), default="official9")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Build only the named official task(s); repeat for multiple tasks.",
    )
    parser.add_argument("--rmbench-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--plan-only", action="store_true")

    parser.add_argument("--training-run-contract", type=Path)
    parser.add_argument("--validation-run-contract", type=Path)
    parser.add_argument("--warm-checkpoint", type=Path)
    parser.add_argument("--training-attestation", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--normalizer-contract", type=Path)
    parser.add_argument("--encoder-contract", type=Path)
    parser.add_argument("--camera-contract", type=Path)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "rmbench_3cam.yaml",
    )
    parser.add_argument("--dino-checkpoint", type=Path)
    parser.add_argument("--normalization-stats", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--vae-checkpoint", type=Path)
    parser.add_argument("--text-encoder", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--sim-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sim_rmbench.yaml",
    )
    parser.add_argument(
        "--evaluation-namespace", default="warm-rmbench-full-v1"
    )
    parser.add_argument("--top-k", type=_positive_int, default=32)
    parser.add_argument("--memory-sigma", type=_positive_float, default=0.2)
    parser.add_argument("--action-horizon", type=_positive_int, default=32)
    parser.add_argument("--action-dim", type=_positive_int, default=14)
    parser.add_argument("--replan-steps", type=_positive_int, default=10)
    parser.add_argument("--recent-event-capacity", type=_positive_int, default=6)
    parser.add_argument("--action-summary-capacity", type=_positive_int, default=2)
    parser.add_argument("--dino-device", default="cuda")
    parser.add_argument("--dino-batch-size", type=_positive_int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--sigma-shift", type=float)
    parser.add_argument("--text-cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--tiled", action="store_true")
    return parser


def _canonical_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
        )
        + ("\n" if indent is not None else "")
    ).encode("utf-8")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_bytes(_canonical_bytes(value, indent=2))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_npy_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def root_seed_namespace(root_seed: int) -> np.ndarray:
    """Return ``[ordinal, lower-bound candidate seed]`` for 100 rollouts.

    The evaluator starts at ``100000 * (1 + root_seed)``.  Expert setup
    failures can skip arbitrary candidates before a rollout is accepted, so
    column one is a deterministic namespace lower-bound sequence rather than
    a claim about accepted seeds.
    """

    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise RMBenchBundleError("root_seed must be a non-negative integer")
    start = 100_000 * (1 + root_seed)
    ordinal = np.arange(RMBENCH_EPISODES_PER_TASK, dtype=np.int64)
    return np.stack((ordinal, start + ordinal), axis=1)


def _required_path(args: argparse.Namespace, name: str) -> Path:
    value = getattr(args, name)
    if value is None:
        raise RMBenchBundleError(f"--{name.replace('_', '-')} is required to build")
    result = Path(value).expanduser().resolve()
    if not result.exists():
        raise RMBenchBundleError(f"required input does not exist: {name}={result}")
    return result


def _artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    names = (
        "training_run_contract",
        "validation_run_contract",
        "warm_checkpoint",
        "training_attestation",
        "base_checkpoint",
        "bank",
        "normalizer_contract",
        "encoder_contract",
        "camera_contract",
        "data_config",
        "dino_checkpoint",
        "normalization_stats",
        "catalog",
        "audit_report",
        "vae_checkpoint",
        "text_encoder",
        "tokenizer",
        "sim_config",
    )
    return {name: _required_path(args, name) for name in names}


def _runtime_args(
    args: argparse.Namespace,
    artifacts: Mapping[str, Path],
    *,
    experiment: Experiment,
    task_name: str,
    contract_path: Path,
    seed_protocol_path: Path,
    task_definition_path: Path,
    root_seed: int,
) -> dict[str, Any]:
    return {
        "sim_cfg_path": str(artifacts["sim_config"]),
        "sim_task": "rmbench_warm_online_3cam384_full",
        "task_name": task_name,
        "seed": root_seed,
        "ckpt_setting": str(artifacts["warm_checkpoint"]),
        "dataset_stats_path": str(artifacts["normalization_stats"]),
        "action_horizon": args.action_horizon,
        "replan_steps": args.replan_steps,
        "num_inference_steps": experiment.num_inference_steps,
        "sigma_shift": args.sigma_shift,
        "text_cfg_scale": args.text_cfg_scale,
        "negative_prompt": args.negative_prompt,
        "rand_device": args.rand_device,
        "tiled": bool(args.tiled),
        "mixed_precision": args.mixed_precision,
        "device": args.device,
        "warm_online_contract_path": str(contract_path),
        "warm_training_attestation_path": str(artifacts["training_attestation"]),
        "warm_training_run_contract_path": str(artifacts["training_run_contract"]),
        "warm_validation_run_contract_path": str(
            artifacts["validation_run_contract"]
        ),
        "warm_base_checkpoint_path": str(artifacts["base_checkpoint"]),
        "warm_bank_directory": str(artifacts["bank"]),
        "warm_normalizer_contract_path": str(artifacts["normalizer_contract"]),
        "warm_encoder_contract_path": str(artifacts["encoder_contract"]),
        "warm_camera_contract_path": str(artifacts["camera_contract"]),
        "warm_m1_data_config_path": str(artifacts["data_config"]),
        "warm_dino_checkpoint_path": str(artifacts["dino_checkpoint"]),
        "warm_catalog_path": str(artifacts["catalog"]),
        "warm_audit_report_path": str(artifacts["audit_report"]),
        "warm_initial_states_path": str(seed_protocol_path),
        "warm_task_definition_path": str(task_definition_path),
        "warm_vae_checkpoint_path": str(artifacts["vae_checkpoint"]),
        "warm_text_encoder_path": str(artifacts["text_encoder"]),
        "warm_tokenizer_path": str(artifacts["tokenizer"]),
        "warm_experiment_id": experiment.id,
        "warm_ablation_mode": experiment.ablation_mode,
        "warm_memory_corruption": experiment.memory_corruption,
        "warm_evaluation_namespace": args.evaluation_namespace,
        "warm_top_k": args.top_k,
        "warm_recent_event_capacity": args.recent_event_capacity,
        "warm_action_summary_capacity": args.action_summary_capacity,
        "warm_dino_device": args.dino_device,
        "warm_dino_batch_size": args.dino_batch_size,
    }


def _compose_projection(
    runtime_args: Mapping[str, Any],
    *,
    checkpoint: Path,
    stats: Path,
    task_name: str,
    task_id: int,
    compose_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if compose_fn is None:
        from experiments.rmbench.warm_policy.deploy_policy import (
            _compose_runtime_config as compose_fn,
        )
    cfg = compose_fn(
        runtime_args,
        checkpoint=checkpoint,
        stats=stats,
        task_name=task_name,
        task_id=task_id,
        allow_missing_contract_path=True,
    )
    try:
        from omegaconf import OmegaConf

        resolved = OmegaConf.to_container(cfg, resolve=True)
    except Exception as exc:  # pragma: no cover - server dependency failure
        raise RMBenchBundleError("cannot resolve the exact policy config") from exc
    if not isinstance(resolved, Mapping):
        raise RMBenchBundleError("composed policy config is not a mapping")
    return build_rmbench_policy_runtime_projection(resolved, runtime_args)


def _online_builder_argv(
    args: argparse.Namespace,
    artifacts: Mapping[str, Path],
    *,
    task_name: str,
    task_id: int,
    projection_path: Path,
    seed_protocol_path: Path,
    task_definition_path: Path,
    output_path: Path,
    root_seed: int,
) -> list[str]:
    return [
        "--benchmark-profile", "robotwin",
        "--resolved-config-binding", "rmbench_policy_runtime",
        "--training-run-contract", str(artifacts["training_run_contract"]),
        "--validation-run-contract", str(artifacts["validation_run_contract"]),
        "--warm-checkpoint", str(artifacts["warm_checkpoint"]),
        "--training-attestation", str(artifacts["training_attestation"]),
        "--bank", str(artifacts["bank"]),
        "--normalizer-contract", str(artifacts["normalizer_contract"]),
        "--encoder-contract", str(artifacts["encoder_contract"]),
        "--camera-contract", str(artifacts["camera_contract"]),
        "--data-config", str(artifacts["data_config"]),
        "--dino-checkpoint", str(artifacts["dino_checkpoint"]),
        "--normalization-stats", str(artifacts["normalization_stats"]),
        "--catalog", str(artifacts["catalog"]),
        "--audit-report", str(artifacts["audit_report"]),
        "--resolved-eval-config", str(projection_path),
        "--vae-checkpoint", str(artifacts["vae_checkpoint"]),
        "--text-encoder", str(artifacts["text_encoder"]),
        "--tokenizer", str(artifacts["tokenizer"]),
        "--evaluation-namespace", args.evaluation_namespace,
        "--task-suite", "rmbench",
        "--task-id", str(task_id),
        "--task-description", task_name,
        "--initial-states", str(seed_protocol_path),
        "--bddl", str(task_definition_path),
        "--root-seed", str(root_seed),
        "--top-k", str(args.top_k),
        "--source-policy", "fixed_context_top1",
        "--memory-sigma", str(args.memory_sigma),
        "--action-horizon", str(args.action_horizon),
        "--action-dim", str(args.action_dim),
        "--output", str(output_path),
    ]


def _load_contract(path: Path) -> WarmOnlineRunContract:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RMBenchBundleError(f"cannot read built online contract: {path}") from exc
    return WarmOnlineRunContract.from_dict(payload)


def _bundle_member(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise RMBenchBundleError(f"invalid {label} relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RMBenchBundleError(f"{label} escapes the bundle root") from exc
    if not path.is_file() or path.is_symlink():
        raise RMBenchBundleError(f"missing or symlinked {label}: {path}")
    return path


def validate_contract_bundle(
    root: str | Path,
    *,
    experiment_id: str,
    task_names: Sequence[str],
) -> dict[str, Any]:
    """Validate a complete immutable bundle before launching the simulator."""

    bundle = Path(root).expanduser().resolve()
    if not bundle.is_dir() or bundle.is_symlink():
        raise RMBenchBundleError(f"bundle root is not a real directory: {bundle}")
    if (bundle / ".incomplete.json").exists():
        raise RMBenchBundleError("bundle has an incomplete construction marker")
    manifest_path = bundle / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RMBenchBundleError("cannot read contract-bundle manifest") from exc
    if not isinstance(manifest, Mapping) or (
        manifest.get("schema") != BUNDLE_SCHEMA
        or manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("task_manifest_sha256") != RMBENCH_TASK_MANIFEST_SHA256
    ):
        raise RMBenchBundleError("contract-bundle manifest identity mismatch")
    experiments = manifest.get("experiments")
    tasks = manifest.get("tasks")
    cells = manifest.get("cells")
    seed_manifest = manifest.get("seed_protocol")
    if not all(
        isinstance(item, list) for item in (experiments, tasks, cells)
    ):
        raise RMBenchBundleError("contract-bundle manifest collections are malformed")
    if (
        manifest.get("cell_count") != len(experiments) * len(tasks)
        or len(cells) != len(experiments) * len(tasks)
    ):
        raise RMBenchBundleError("contract-bundle manifest is not a complete matrix")
    if not isinstance(seed_manifest, Mapping) or (
        seed_manifest.get("schema") != SEED_PROTOCOL_SCHEMA
        or seed_manifest.get("schema_version") != SEED_PROTOCOL_VERSION
        or seed_manifest.get("not_accepted_seed_claim") is not True
    ):
        raise RMBenchBundleError("seed namespace manifest is malformed")
    experiment = next(
        (
            item
            for item in experiments
            if isinstance(item, Mapping) and item.get("id") == experiment_id
        ),
        None,
    )
    if experiment is None:
        raise RMBenchBundleError(f"experiment is absent from bundle: {experiment_id}")
    expected_tasks = tuple(str(name) for name in task_names)
    if len(expected_tasks) != len(set(expected_tasks)) or not expected_tasks:
        raise RMBenchBundleError("requested task names must be unique and non-empty")
    manifest_task_names = {
        item.get("name") for item in tasks if isinstance(item, Mapping)
    }
    if not set(expected_tasks).issubset(manifest_task_names):
        raise RMBenchBundleError("requested tasks are absent from bundle manifest")

    seed_rows = seed_manifest.get("tasks")
    if not isinstance(seed_rows, list):
        raise RMBenchBundleError("seed namespace task rows are malformed")
    seed_by_task = {
        row.get("task"): row for row in seed_rows if isinstance(row, Mapping)
    }
    cell_by_task: dict[str, Mapping[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise RMBenchBundleError("contract-bundle cell is malformed")
        cell_experiment = cell.get("experiment")
        if isinstance(cell_experiment, Mapping) and cell_experiment.get("id") == experiment_id:
            task_name = str(cell.get("task"))
            if task_name in cell_by_task:
                raise RMBenchBundleError("duplicate experiment/task contract cell")
            cell_by_task[task_name] = cell

    records: list[dict[str, Any]] = []
    for task_name in expected_tasks:
        seed_row = seed_by_task.get(task_name)
        cell = cell_by_task.get(task_name)
        if not isinstance(seed_row, Mapping) or not isinstance(cell, Mapping):
            raise RMBenchBundleError(f"bundle is incomplete for task {task_name}")
        if seed_row.get("not_accepted_seed_claim") is not True:
            raise RMBenchBundleError("seed row falsely claims accepted rollout seeds")
        seed_path = _bundle_member(
            bundle, seed_row.get("relative_path"), label="seed protocol"
        )
        if seed_path != bundle / "seeds" / f"{task_name}.seed_protocol.npy":
            raise RMBenchBundleError("seed protocol does not use the fixed layout")
        seed = np.load(seed_path, allow_pickle=False)
        expected_seed = root_seed_namespace(int(manifest.get("root_seed")))
        if (
            seed.shape != (RMBENCH_EPISODES_PER_TASK, 2)
            or seed.dtype != np.int64
            or not np.array_equal(seed, expected_seed)
            or sha256_file(seed_path) != seed_row.get("file_sha256")
            or sha256_array(seed) != seed_row.get("array_sha256")
        ):
            raise RMBenchBundleError(f"seed namespace drift for task {task_name}")

        contract_path = _bundle_member(
            bundle, cell.get("contract_relpath"), label="online contract"
        )
        projection_path = _bundle_member(
            bundle,
            cell.get("runtime_projection_relpath"),
            label="runtime projection",
        )
        if contract_path != bundle / experiment_id / f"{task_name}.json":
            raise RMBenchBundleError("online contract does not use the fixed layout")
        if projection_path != (
            bundle / "runtime_projections" / experiment_id / f"{task_name}.json"
        ):
            raise RMBenchBundleError("runtime projection does not use the fixed layout")
        if sha256_file(contract_path) != cell.get("contract_file_sha256"):
            raise RMBenchBundleError(f"contract file drift for task {task_name}")
        if sha256_file(projection_path) != cell.get("runtime_projection_sha256"):
            raise RMBenchBundleError(f"runtime projection file drift for task {task_name}")
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        contract = _load_contract(contract_path)
        official_task_id = next(
            index
            for index, official in enumerate(RMBENCH_TASKS)
            if official.name == task_name
        )
        canonical_projection_sha = sha256_canonical_json(projection)
        if (
            contract.sha256 != cell.get("online_run_contract_sha256")
            or contract.resolved_eval_config_sha256 != canonical_projection_sha
            or cell.get("runtime_projection_canonical_sha256")
            != canonical_projection_sha
            or contract.task_suite != "rmbench"
            or contract.task_id != official_task_id
            or contract.task_description != task_name
            or contract.root_seed != int(manifest.get("root_seed"))
            or contract.initial_states_sha256 != sha256_array(seed)
            or contract.action_horizon != 32
            or contract.action_dim != 14
        ):
            raise RMBenchBundleError(f"semantic contract drift for task {task_name}")
        projected_experiment = projection.get("experiment")
        expected_experiment_projection = {
            "experiment_id": experiment.get("id"),
            "ablation_mode": experiment.get("ablation_mode"),
            "memory_corruption": experiment.get("memory_corruption"),
        }
        if not isinstance(projected_experiment, Mapping) or (
            dict(projected_experiment) != expected_experiment_projection
        ):
            raise RMBenchBundleError(f"experiment controls drift for task {task_name}")
        records.append(
            {
                "task": task_name,
                "contract_sha256": contract.sha256,
                "seed_namespace_sha256": sha256_array(seed),
            }
        )
    return {
        "bundle_root": str(bundle),
        "manifest_sha256": sha256_file(manifest_path),
        "experiment_id": experiment_id,
        "tasks": records,
    }


def _plan(
    args: argparse.Namespace,
    *,
    experiments: Sequence[Experiment],
    tasks: Sequence[Any],
    root_seed: int,
    matrix_sha256: str,
    checkout: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": BUNDLE_SCHEMA,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "suite": args.suite,
        "root_seed": root_seed,
        "matrix_sha256": matrix_sha256,
        "official_checkout": {
            "revision": checkout["head"],
            "tracked_clean": checkout["tracked_clean"],
            "push_url": checkout["push_url"],
        },
        "experiments": [asdict(item) for item in experiments],
        "tasks": [
            {
                "task_id": next(
                    index
                    for index, official in enumerate(RMBENCH_TASKS)
                    if official.name == task.name
                ),
                "name": task.name,
                "memory_regime": task.memory_regime,
                "step_limit": task.step_limit,
            }
            for task in tasks
        ],
        "cell_count": len(experiments) * len(tasks),
        "output_root": str(args.output_root.expanduser().resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix_path = args.matrix.expanduser().resolve()
    matrix = load_matrix(matrix_path)
    experiments = select_experiments(matrix, args.experiment)
    tasks = tasks_for_suite(args.suite)
    if args.task:
        requested = tuple(args.task)
        if len(requested) != len(set(requested)):
            raise RMBenchBundleError("--task values must be unique")
        known = {task.name: task for task in tasks}
        unknown = sorted(set(requested) - set(known))
        if unknown:
            raise RMBenchBundleError(f"unknown official RMBench tasks: {unknown}")
        tasks = tuple(known[name] for name in requested)
    checkout_root = args.rmbench_root.expanduser().resolve()
    checkout = validate_read_only_checkout(
        checkout_root,
        expected_revision=RMBENCH_CODE_REVISION,
        require_push_disabled=True,
    )
    plan = _plan(
        args,
        experiments=experiments,
        tasks=tasks,
        root_seed=matrix.root_seed,
        matrix_sha256=matrix.sha256,
        checkout=checkout,
    )
    if args.plan_only:
        print(json.dumps(plan, sort_keys=True, indent=2, ensure_ascii=False))
        return 0

    if args.action_horizon != 32 or args.action_dim != 14:
        raise RMBenchBundleError("formal RMBench contracts require H=32 and D=14")
    artifacts = _artifact_paths(args)
    output = args.output_root.expanduser().resolve()
    for forbidden in (PROJECT_ROOT, checkout_root):
        try:
            output.relative_to(forbidden)
        except ValueError:
            pass
        else:
            raise RMBenchBundleError(f"bundle output must be outside {forbidden}")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RMBenchBundleError(
            f"refusing to overwrite existing bundle root: {output}"
        ) from exc
    incomplete = output / ".incomplete.json"
    _write_json_atomic(
        incomplete,
        {
            **plan,
            "complete": False,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    seed_rows: list[dict[str, Any]] = []
    seed_array = root_seed_namespace(matrix.root_seed)
    for task in tasks:
        seed_path = output / "seeds" / f"{task.name}.seed_protocol.npy"
        _write_npy_atomic(seed_path, seed_array)
        seed_rows.append(
            {
                "task": task.name,
                "relative_path": seed_path.relative_to(output).as_posix(),
                "file_sha256": sha256_file(seed_path),
                "array_sha256": sha256_array(seed_array),
                "shape": list(seed_array.shape),
                "dtype": str(seed_array.dtype),
                "namespace_start": int(seed_array[0, 1]),
                "accepted_rollout_seed_claim": False,
                "not_accepted_seed_claim": True,
            }
        )

    cells: list[dict[str, Any]] = []
    for experiment in experiments:
        for task in tasks:
            task_id = next(
                index
                for index, official in enumerate(RMBENCH_TASKS)
                if official.name == task.name
            )
            contract_path = output / experiment.id / f"{task.name}.json"
            projection_path = (
                output
                / "runtime_projections"
                / experiment.id
                / f"{task.name}.json"
            )
            seed_path = output / "seeds" / f"{task.name}.seed_protocol.npy"
            task_definition = checkout_root / "envs" / f"{task.name}.py"
            if not task_definition.is_file():
                raise RMBenchBundleError(
                    f"pinned official task implementation is missing: {task_definition}"
                )
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_args = _runtime_args(
                args,
                artifacts,
                experiment=experiment,
                task_name=task.name,
                contract_path=contract_path,
                seed_protocol_path=seed_path,
                task_definition_path=task_definition,
                root_seed=matrix.root_seed,
            )
            projection = _compose_projection(
                runtime_args,
                checkpoint=artifacts["warm_checkpoint"],
                stats=artifacts["normalization_stats"],
                task_name=task.name,
                task_id=task_id,
            )
            _write_json_atomic(projection_path, projection)
            online_builder.main(
                _online_builder_argv(
                    args,
                    artifacts,
                    task_name=task.name,
                    task_id=task_id,
                    projection_path=projection_path,
                    seed_protocol_path=seed_path,
                    task_definition_path=task_definition,
                    output_path=contract_path,
                    root_seed=matrix.root_seed,
                )
            )
            contract = _load_contract(contract_path)
            cells.append(
                {
                    "experiment": asdict(experiment),
                    "task_id": task_id,
                    "task": task.name,
                    "task_definition_sha256": sha256_file(task_definition),
                    "runtime_projection_relpath": projection_path.relative_to(
                        output
                    ).as_posix(),
                    "runtime_projection_sha256": sha256_file(projection_path),
                    "runtime_projection_canonical_sha256": (
                        sha256_canonical_json(projection)
                    ),
                    "contract_relpath": contract_path.relative_to(output).as_posix(),
                    "contract_file_sha256": sha256_file(contract_path),
                    "online_run_contract_sha256": contract.sha256,
                }
            )

    manifest = {
        **plan,
        "complete": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_file_sha256": sha256_file(matrix_path),
        "task_manifest_sha256": RMBENCH_TASK_MANIFEST_SHA256,
        "seed_protocol": {
            "schema": SEED_PROTOCOL_SCHEMA,
            "schema_version": SEED_PROTOCOL_VERSION,
            "formula": "candidate_seed_lower_bound = 100000*(1+root_seed)+ordinal",
            "columns": ["rollout_ordinal", "candidate_namespace_lower_bound"],
            "accepted_rollout_seed_claim": False,
            "not_accepted_seed_claim": True,
            "reason": (
                "official expert setup can skip candidates; actual accepted seeds "
                "are captured from the official episode log"
            ),
            "tasks": seed_rows,
        },
        "cells": cells,
    }
    manifest_path = output / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    incomplete.unlink()
    print(
        json.dumps(
            {
                "bundle_root": str(output),
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "cells": len(cells),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
