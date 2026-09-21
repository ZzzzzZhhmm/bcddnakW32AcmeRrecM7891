#!/usr/bin/env python3
"""Run the closed WARM RMBench ablation/corruption/ODE matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = PROJECT_ROOT / "configs" / "ablation" / "rmbench_reproducible_matrix.json"
EVALUATOR = PROJECT_ROOT / "scripts" / "evaluate_warm_rmbench_server.sh"
SCHEMA = "warm.rmbench-experiment-matrix"
SCHEMA_VERSION = 1
ALLOWED_ABLATIONS = frozenset(
    {"context_only", "source_only_no_consequence", "full"}
)
ALLOWED_CORRUPTIONS = frozenset(
    {"clean", "wrong_event", "reversed_action", "phase_shift", "effect_mismatch"}
)
ALLOWED_ODE_STEPS = frozenset({2, 4, 8, 10})
EXPERIMENT_ID = re.compile(r"[a-z][a-z0-9_]{1,63}\Z")
REQUIRED_EXECUTION_ENV = (
    "WARM_ARTIFACT_ROOT",
    "RMBENCH_ROOT",
    "RMBENCH_HF_REVISION_MARKER",
    "FASTWAM_BASE_CHECKPOINT",
    "WARM_CHECKPOINT",
    "WARM_TRAINING_ATTESTATION",
    "WARM_RMBENCH_ONLINE_CONTRACT",
    "WARM_DINO_CHECKPOINT",
    "WARM_VAE_CHECKPOINT",
    "WARM_TEXT_ENCODER",
    "WARM_TOKENIZER",
)


class MatrixError(RuntimeError):
    """Raised before an incomplete or ambiguous formal matrix can launch."""


@dataclass(frozen=True, slots=True)
class Experiment:
    id: str
    ablation_mode: str
    memory_corruption: str
    num_inference_steps: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Experiment":
        expected = {
            "id",
            "ablation_mode",
            "memory_corruption",
            "num_inference_steps",
        }
        if set(value) != expected:
            raise MatrixError("experiment has unknown or missing fields")
        experiment = cls(
            id=value["id"],
            ablation_mode=value["ablation_mode"],
            memory_corruption=value["memory_corruption"],
            num_inference_steps=value["num_inference_steps"],
        )
        if not isinstance(experiment.id, str) or EXPERIMENT_ID.fullmatch(
            experiment.id
        ) is None:
            raise MatrixError(f"unsafe experiment id: {experiment.id!r}")
        if experiment.ablation_mode not in ALLOWED_ABLATIONS:
            raise MatrixError(f"unsupported ablation: {experiment.ablation_mode!r}")
        if experiment.memory_corruption not in ALLOWED_CORRUPTIONS:
            raise MatrixError(
                f"unsupported memory corruption: {experiment.memory_corruption!r}"
            )
        if (
            isinstance(experiment.num_inference_steps, bool)
            or experiment.num_inference_steps not in ALLOWED_ODE_STEPS
        ):
            raise MatrixError("num_inference_steps must be one of 2, 4, 8, 10")
        if (
            experiment.ablation_mode != "full"
            and experiment.memory_corruption != "clean"
        ):
            raise MatrixError("memory corruptions are defined only for full WARM")
        return experiment


@dataclass(frozen=True, slots=True)
class Matrix:
    suite: str
    root_seed: int
    experiments: tuple[Experiment, ...]
    sha256: str


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def load_matrix(path: Path) -> Matrix:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatrixError(f"cannot read experiment matrix: {path}") from exc
    if not isinstance(raw, Mapping):
        raise MatrixError("experiment matrix must be a mapping")
    expected = {
        "schema",
        "schema_version",
        "benchmark",
        "suite",
        "root_seed",
        "experiments",
    }
    if set(raw) != expected:
        raise MatrixError("experiment matrix has unknown or missing fields")
    if raw["schema"] != SCHEMA or raw["schema_version"] != SCHEMA_VERSION:
        raise MatrixError("experiment-matrix schema mismatch")
    if raw["benchmark"] != "RMBench" or raw["suite"] != "official9":
        raise MatrixError("formal matrix must target the official9 RMBench suite")
    root_seed = raw["root_seed"]
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise MatrixError("root_seed must be a non-negative integer")
    rows = raw["experiments"]
    if not isinstance(rows, list) or not rows:
        raise MatrixError("experiments must be a non-empty list")
    experiments = tuple(
        Experiment.from_mapping(row) if isinstance(row, Mapping) else _bad_row()
        for row in rows
    )
    ids = [item.id for item in experiments]
    if len(ids) != len(set(ids)):
        raise MatrixError("experiment ids must be unique")

    clean_modes = {
        item.ablation_mode
        for item in experiments
        if item.memory_corruption == "clean" and item.num_inference_steps == 10
    }
    if clean_modes != ALLOWED_ABLATIONS:
        raise MatrixError(
            "matrix must contain context-only, source-only/no-consequence, and full WARM"
        )
    observed_corruptions = {
        item.memory_corruption
        for item in experiments
        if item.ablation_mode == "full" and item.num_inference_steps == 10
    }
    if observed_corruptions != ALLOWED_CORRUPTIONS:
        raise MatrixError("matrix does not cover the closed corruption suite")
    observed_steps = {
        item.num_inference_steps
        for item in experiments
        if item.ablation_mode == "full" and item.memory_corruption == "clean"
    }
    if observed_steps != ALLOWED_ODE_STEPS:
        raise MatrixError("matrix must cover ODE steps 2, 4, 8, and 10")
    return Matrix(
        suite="official9",
        root_seed=root_seed,
        experiments=experiments,
        sha256=sha256(_canonical(raw)).hexdigest(),
    )


def _bad_row() -> Experiment:
    raise MatrixError("every experiment row must be a mapping")


def select_experiments(
    matrix: Matrix, requested: Sequence[str]
) -> tuple[Experiment, ...]:
    if not requested:
        return matrix.experiments
    if len(requested) != len(set(requested)):
        raise MatrixError("--experiment values must not repeat")
    by_id = {item.id: item for item in matrix.experiments}
    missing = [name for name in requested if name not in by_id]
    if missing:
        raise MatrixError(f"unknown experiment ids: {missing}")
    return tuple(by_id[name] for name in requested)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_environment() -> None:
    missing = [name for name in REQUIRED_EXECUTION_ENV if not os.environ.get(name)]
    if missing:
        raise MatrixError(f"missing formal execution environment: {missing}")


def _git(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _preflight_formal_inputs(
    experiments: Sequence[Experiment], *, suite: str
) -> None:
    for name in REQUIRED_EXECUTION_ENV:
        path = Path(os.path.expandvars(os.path.expanduser(os.environ[name]))).resolve()
        if not path.exists():
            raise MatrixError(f"required formal input does not exist: {name}={path}")
    bundle = Path(os.environ["WARM_RMBENCH_ONLINE_CONTRACT"]).resolve()
    if not bundle.is_dir():
        raise MatrixError("WARM_RMBENCH_ONLINE_CONTRACT must be a bundle directory")
    task_names = (
        (
            "observe_and_pickup",
            "rearrange_blocks",
            "put_back_block",
            "swap_blocks",
            "swap_T",
            "blocks_ranking_try",
            "press_button",
            "cover_blocks",
            "battery_try",
        )
        if suite == "official9"
        else ("put_back_block", "rearrange_blocks", "battery_try")
    )
    for task_name in task_names:
        seed_protocol = bundle / "seeds" / f"{task_name}.seed_protocol.npy"
        if not seed_protocol.is_file():
            raise MatrixError(f"missing seed protocol: {seed_protocol}")
        for experiment in experiments:
            contract = bundle / experiment.id / f"{task_name}.json"
            if not contract.is_file():
                raise MatrixError(f"missing experiment/task contract: {contract}")


def _write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    path.write_text(encoded, encoding="utf-8")


def _git_head() -> str:
    return _git("rev-parse", "HEAD")


def _validate_result(path: Path, *, suite: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise MatrixError(f"RMBench manager did not publish summary: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatrixError(f"invalid RMBench summary: {path}") from exc
    if not isinstance(payload, Mapping) or payload.get("suite") != suite:
        raise MatrixError(f"RMBench summary suite mismatch: {path}")
    expected_tasks = 9 if suite == "official9" else 3
    aggregates = payload.get("aggregates")
    if (
        not isinstance(aggregates, Mapping)
        or aggregates.get("complete_tasks") != expected_tasks
        or aggregates.get("expected_tasks") != expected_tasks
        or payload.get("failures") != []
    ):
        raise MatrixError(f"incomplete RMBench summary: {path}")
    if suite == "official9" and payload.get("is_official_nine_task_score") is not True:
        raise MatrixError(f"official9 summary is not marked official: {path}")
    per_task = payload.get("per_task")
    if not isinstance(per_task, list) or len(per_task) != expected_tasks:
        raise MatrixError(f"RMBench summary has incomplete per-task seed evidence: {path}")
    seed_hashes: dict[str, str] = {}
    for row in per_task:
        if not isinstance(row, Mapping):
            raise MatrixError(f"RMBench summary has malformed per-task row: {path}")
        task_name = row.get("task_name")
        seed_hash = row.get("actual_accepted_seed_sha256")
        if (
            not isinstance(task_name, str)
            or not isinstance(seed_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", seed_hash) is None
            or task_name in seed_hashes
        ):
            raise MatrixError(f"RMBench summary has invalid accepted-seed hash: {path}")
        seed_hashes[task_name] = seed_hash
    identity = payload.get("run_identity")
    if not isinstance(identity, Mapping):
        raise MatrixError(f"RMBench summary has no run identity: {path}")
    action_generation = identity.get("action_generation")
    if (
        not isinstance(identity.get("policy_kind"), str)
        or not isinstance(identity.get("policy_name"), str)
        or isinstance(identity.get("root_seed"), bool)
        or not isinstance(identity.get("root_seed"), int)
        or not isinstance(identity.get("checkpoint_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", identity["checkpoint_sha256"]) is None
        or not isinstance(action_generation, Mapping)
        or any(
            isinstance(action_generation.get(name), bool)
            or not isinstance(action_generation.get(name), int)
            or int(action_generation[name]) <= 0
            for name in ("action_horizon", "replan_steps", "num_inference_steps")
        )
    ):
        raise MatrixError(f"RMBench summary has malformed run identity: {path}")
    return payload


def _validate_fastwam_reference(
    summary: Mapping[str, Any], *, expected_root_seed: int
) -> None:
    identity = summary["run_identity"]
    action_generation = identity["action_generation"]
    if identity["policy_kind"] != "fastwam_baseline" or (
        identity["policy_name"] != "fastwam_policy"
    ):
        raise MatrixError(
            "accepted-seed reference must be a FastWAM baseline summary"
        )
    if identity["root_seed"] != expected_root_seed:
        raise MatrixError(
            "FastWAM reference root seed differs from the experiment matrix"
        )
    expected_generation = {
        "action_horizon": 32,
        "replan_steps": 10,
        "num_inference_steps": 10,
    }
    if dict(action_generation) != expected_generation:
        raise MatrixError(
            "FastWAM reference must use H=32, replan=10, and ODE=10"
        )


def _accepted_seed_hashes(summary: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row["task_name"]): str(row["actual_accepted_seed_sha256"])
        for row in summary["per_task"]
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Run only this registered experiment ID; repeat to preserve order.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--accepted-seed-reference",
        type=Path,
        help=(
            "Completed same-suite FastWAM baseline summary whose per-task actual "
            "accepted seeds must match every WARM matrix cell."
        ),
    )
    parser.add_argument("--suite", choices=("official9", "pilot3"), default="official9")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix = load_matrix(args.matrix.expanduser().resolve())
    selected = select_experiments(matrix, args.experiment)
    plan = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "matrix_sha256": matrix.sha256,
        "suite": args.suite,
        "root_seed": matrix.root_seed,
        "complete_matrix": len(selected) == len(matrix.experiments),
        "experiments": [asdict(item) for item in selected],
    }
    if args.plan_only:
        print(json.dumps(plan, sort_keys=True, indent=2))
        return 0

    _require_environment()
    _preflight_formal_inputs(selected, suite=args.suite)
    if not EVALUATOR.is_file():
        raise MatrixError(f"server evaluator missing: {EVALUATOR}")
    output_value = args.output_root or (
        Path(os.environ["WARM_EVAL_MATRIX_ROOT"])
        if os.environ.get("WARM_EVAL_MATRIX_ROOT")
        else None
    )
    if output_value is None:
        raise MatrixError("--output-root or WARM_EVAL_MATRIX_ROOT is required")
    reference_value = args.accepted_seed_reference or (
        Path(os.environ["WARM_RMBENCH_ACCEPTED_SEED_REFERENCE"])
        if os.environ.get("WARM_RMBENCH_ACCEPTED_SEED_REFERENCE")
        else None
    )
    if reference_value is None:
        raise MatrixError(
            "--accepted-seed-reference or "
            "WARM_RMBENCH_ACCEPTED_SEED_REFERENCE is required"
        )
    reference_path = reference_value.expanduser().resolve()
    reference_summary = _validate_result(reference_path, suite=args.suite)
    _validate_fastwam_reference(
        reference_summary, expected_root_seed=matrix.root_seed
    )
    output = output_value.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise MatrixError(f"immutable matrix output already exists: {output}")
    for forbidden in (PROJECT_ROOT, Path(os.environ["RMBENCH_ROOT"]).resolve()):
        try:
            output.relative_to(forbidden)
        except ValueError:
            pass
        else:
            raise MatrixError(f"matrix output must be outside {forbidden}")
    output.mkdir(parents=True)
    plan["warm_git_revision"] = _git_head()
    plan["accepted_seed_reference"] = {
        "path": str(reference_path),
        "sha256": _file_sha256(reference_path),
        "checkpoint_sha256": reference_summary["run_identity"][
            "checkpoint_sha256"
        ],
        "macro_success_rate": reference_summary["aggregates"][
            "macro_success_rate"
        ],
        "micro_success_rate": reference_summary["aggregates"][
            "micro_success_rate"
        ],
    }
    plan["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output / "matrix_plan.json", plan)

    records: list[dict[str, Any]] = []
    reference_seed_hashes: dict[str, str] | None = _accepted_seed_hashes(
        reference_summary
    )
    reference_seed_experiment: str | None = "fastwam_baseline"
    for experiment in selected:
        experiment_output = output / experiment.id
        env = os.environ.copy()
        env.update(
            {
                "WARM_EVAL_ROOT": str(experiment_output),
                "WARM_EXPERIMENT_ID": experiment.id,
                "WARM_ABLATION_MODE": experiment.ablation_mode,
                "WARM_MEMORY_CORRUPTION": experiment.memory_corruption,
                "WARM_NUM_INFERENCE_STEPS": str(experiment.num_inference_steps),
                "WARM_RMBENCH_SUITE": args.suite,
                "WARM_ROOT_SEED": str(matrix.root_seed),
            }
        )
        subprocess.run(
            ["bash", str(EVALUATOR)],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
        )
        summary_path = experiment_output / "summary.json"
        summary = _validate_result(summary_path, suite=args.suite)
        accepted_seed_hashes = _accepted_seed_hashes(summary)
        if reference_seed_hashes is None:
            reference_seed_hashes = accepted_seed_hashes
            reference_seed_experiment = experiment.id
        elif accepted_seed_hashes != reference_seed_hashes:
            raise MatrixError(
                "official accepted environment seeds drifted across matrix cells: "
                f"reference={reference_seed_experiment}, current={experiment.id}"
            )
        records.append(
            {
                **asdict(experiment),
                "summary_relpath": summary_path.relative_to(output).as_posix(),
                "summary_sha256": _file_sha256(summary_path),
                "macro_success_rate": summary["aggregates"]["macro_success_rate"],
                "micro_success_rate": summary["aggregates"]["micro_success_rate"],
                "macro_success_rate_delta_vs_fastwam": (
                    summary["aggregates"]["macro_success_rate"]
                    - reference_summary["aggregates"]["macro_success_rate"]
                ),
                "micro_success_rate_delta_vs_fastwam": (
                    summary["aggregates"]["micro_success_rate"]
                    - reference_summary["aggregates"]["micro_success_rate"]
                ),
                "accepted_seed_set_sha256": sha256(
                    _canonical(accepted_seed_hashes)
                ).hexdigest(),
            }
        )
        _write_json(
            output / "matrix_results.in_progress.json",
            {**plan, "completed": records},
        )

    result = {
        **plan,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": records,
    }
    _write_json(output / "matrix_results.json", result)
    (output / "matrix_results.in_progress.json").unlink(missing_ok=True)
    print(f"RMBench matrix complete: {output / 'matrix_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
