#!/usr/bin/env python3
"""Build one closed-world contract for a WARM online rollout.

The online contract is deliberately separate from the stride-one training
candidate-cache contract.  It binds the exact checkpoint, train event bank,
frozen online encoder, evaluation configuration, task identity, seed namespace,
and simulator initialization artifacts used by one rollout job.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from fastwam.datasets.lerobot.audit import load_audit_report
from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog
from fastwam.memory.action_contract import validate_action_space_contract
from fastwam.memory.bank_contract import validate_warm_v1_bank
from fastwam.memory.candidate_cache import canonical_event_bank_content_hash
from fastwam.memory.event_bank import EventBank, MANIFEST_FILENAME
from fastwam.memory.manifest import (
    sha256_array,
    sha256_canonical_json,
    sha256_file,
    sha256_path_tree,
)
from fastwam.models.warm.online_contract import (
    ONLINE_RETRIEVAL_IMPLEMENTATION,
    WarmOnlineRunContract,
)
from fastwam.models.warm.source_contract import WarmSourceRunContract
from fastwam.models.warm.training_attestation import (
    TrainingAttestationError,
    verify_training_attestation,
)
from fastwam.benchmarks.rmbench_runtime import (
    RMBENCH_RUNTIME_PROJECTION_SCHEMA,
    RMBENCH_RUNTIME_PROJECTION_VERSION,
)
from fastwam.memory.online_retrieval import (
    validate_online_camera_contract,
    validate_online_encoder_contract,
)
from fastwam.memory.processor_contract import (
    ProcessorContractError,
    extract_m1_processor_recipe,
    load_m1_data_config,
)
from fastwam.utils.artifact_claim import artifact_claim


SUMMARY_SCHEMA = "warm.online-run-contract-build-summary"
SUMMARY_SCHEMA_VERSION = 1


class OnlineContractBuildError(RuntimeError):
    """Raised when online rollout artifacts cannot be bound safely."""


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a contract-bound WARM online rollout job."
    )
    parser.add_argument(
        "--benchmark-profile",
        choices=("libero", "robotwin"),
        default="libero",
        help="Exact camera/action processor profile used by the rollout.",
    )
    parser.add_argument(
        "--resolved-config-binding",
        choices=("full_eval_config", "rmbench_policy_runtime"),
        default="full_eval_config",
        help=(
            "LIBERO keeps the legacy full resolved-config digest. RMBench "
            "binds the shared policy-runtime projection, excluding runner-only "
            "outputs and telemetry paths."
        ),
    )
    parser.add_argument("--training-run-contract", required=True, type=Path)
    parser.add_argument("--validation-run-contract", required=True, type=Path)
    parser.add_argument("--warm-checkpoint", required=True, type=Path)
    parser.add_argument("--training-attestation", required=True, type=Path)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--normalizer-contract", required=True, type=Path)
    parser.add_argument("--encoder-contract", required=True, type=Path)
    parser.add_argument("--camera-contract", required=True, type=Path)
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--dino-checkpoint", required=True, type=Path)
    parser.add_argument("--normalization-stats", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--resolved-eval-config", required=True, type=Path)
    parser.add_argument("--vae-checkpoint", required=True, type=Path)
    parser.add_argument("--text-encoder", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--evaluation-namespace", required=True)
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--task-id", required=True, type=_nonnegative_int)
    parser.add_argument("--task-description", required=True)
    parser.add_argument(
        "--initial-states",
        required=True,
        type=Path,
        help=(
            "Exact initialization .npy artifact. LIBERO uses its initial-state "
            "array; RMBench uses the deterministic official seed schedule."
        ),
    )
    parser.add_argument(
        "--bddl",
        required=True,
        type=Path,
        help=(
            "Exact task-definition file. LIBERO supplies BDDL; RMBench supplies "
            "the pinned envs/<task>.py module."
        ),
    )
    parser.add_argument("--root-seed", required=True, type=_nonnegative_int)
    parser.add_argument("--top-k", required=True, type=_positive_int)
    parser.add_argument(
        "--source-policy",
        required=True,
        choices=("gaussian_null", "fixed_context_top1"),
    )
    parser.add_argument("--memory-sigma", required=True, type=_positive_float)
    parser.add_argument("--action-horizon", required=True, type=_positive_int)
    parser.add_argument("--action-dim", required=True, type=_positive_int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OnlineContractBuildError(f"cannot read {label} JSON at {path}") from exc
    if not isinstance(value, Mapping):
        raise OnlineContractBuildError(f"{label} must be a JSON object")
    return value


def _read_resolved_config(path: Path) -> Mapping[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        try:
            from omegaconf import OmegaConf

            value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        except Exception as exc:  # pragma: no cover - depends on server extras
            raise OnlineContractBuildError(
                "resolved eval config must be valid JSON or OmegaConf YAML"
            ) from exc
    if not isinstance(value, Mapping):
        raise OnlineContractBuildError("resolved eval config must decode to a mapping")
    # Round-trip through canonical JSON so OmegaConf containers cannot carry
    # non-portable objects into the contract digest.
    try:
        normalized = json.loads(
            json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=True)
        )
    except (TypeError, ValueError) as exc:
        raise OnlineContractBuildError(
            "resolved eval config must contain finite JSON-compatible values"
        ) from exc
    return normalized


def _config_value(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            dotted = ".".join(path)
            raise OnlineContractBuildError(
                f"resolved eval config is missing required field {dotted}"
            )
        current = current[key]
    return current


def _resolved_path_value(value: Any, *, label: str) -> Path:
    if value is None or not str(value).strip():
        raise OnlineContractBuildError(
            f"resolved eval config {label} must name a concrete path"
        )
    return Path(
        os.path.expanduser(os.path.expandvars(str(value)))
    ).resolve()


def _validate_resolved_config(
    value: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    """Reject a hash-only config binding that contradicts explicit fields."""

    if args.resolved_config_binding == "rmbench_policy_runtime":
        _validate_rmbench_policy_runtime_projection(value, args)
        return

    warm_online = _config_value(value, "EVALUATION", "warm_online")
    if not isinstance(warm_online, Mapping):
        raise OnlineContractBuildError(
            "resolved eval config EVALUATION.warm_online must be a mapping"
        )
    mode = str(warm_online.get("mode", "source_only"))
    if mode not in {"source_only", "full_retrospection"}:
        raise OnlineContractBuildError(
            "resolved eval config warm_online.mode must be source_only or "
            "full_retrospection"
        )
    if mode == "full_retrospection" and args.source_policy != "fixed_context_top1":
        raise OnlineContractBuildError(
            "full_retrospection online contracts require fixed_context_top1"
        )

    comparisons = (
        (("seed",), args.root_seed, "root seed"),
        (("EVALUATION", "task_suite_name"), args.task_suite, "task suite"),
        (("EVALUATION", "task_id"), args.task_id, "task id"),
        (
            ("EVALUATION", "warm_online", "evaluation_namespace"),
            args.evaluation_namespace,
            "evaluation namespace",
        ),
        (("EVALUATION", "warm_online", "top_k"), args.top_k, "retrieval top-k"),
        (("model", "source_policy"), args.source_policy, "source policy"),
        (("model", "memory_sigma"), args.memory_sigma, "memory sigma"),
    )
    for path, expected, label in comparisons:
        actual = _config_value(value, *path)
        if isinstance(expected, float):
            try:
                agrees = float(actual) == float(expected)
            except (TypeError, ValueError):
                agrees = False
        else:
            agrees = actual == expected
        if not agrees:
            raise OnlineContractBuildError(
                f"resolved eval config {label} disagrees with explicit contract field: "
                f"{actual!r} != {expected!r}"
            )

    configured_horizon = _config_value(value, "EVALUATION", "action_horizon")
    if configured_horizon is None:
        configured_horizon = int(_config_value(value, "data", "train", "num_frames")) - 1
    if int(configured_horizon) != args.action_horizon:
        raise OnlineContractBuildError(
            "resolved eval config action horizon disagrees with explicit contract field"
        )
    configured_dim = int(
        _config_value(value, "data", "train", "processor", "action_output_dim")
    )
    if configured_dim != args.action_dim:
        raise OnlineContractBuildError(
            "resolved eval config action dimension disagrees with explicit contract field"
        )

    if _config_value(value, "EVALUATION", "warm_online", "enabled") is not True:
        raise OnlineContractBuildError(
            "resolved eval config must enable formal WARM online evaluation"
        )
    if _config_value(value, "EVALUATION", "visualize_future_video") is not False:
        raise OnlineContractBuildError(
            "formal WARM online evaluation must disable future-video inference"
        )
    if _config_value(value, "EVALUATION", "use_action_ensembler") is not False:
        raise OnlineContractBuildError(
            "formal M2.1 online evaluation must disable action ensembling"
        )

    if mode == "source_only":
        # The M2.1 report is produced only after this online contract exists.
        # Its planned path is nevertheless part of the resolved-config digest;
        # the pair builder later requires a passing report at exactly this path.
        _resolved_path_value(
            _config_value(
                value, "EVALUATION", "warm_online", "parity_report_path"
            ),
            label="planned online parity report",
        )
        _resolved_path_value(
            _config_value(
                value, "EVALUATION", "warm_online", "pair_contract_path"
            ),
            label="planned online pair contract",
        )

    path_bindings = (
        (("ckpt",), args.warm_checkpoint, "WARM checkpoint"),
        (
            ("model", "run_contract_path"),
            args.training_run_contract,
            "training run contract",
        ),
        (
            ("model", "validation_run_contract_path"),
            args.validation_run_contract,
            "validation run contract",
        ),
        (
            ("EVALUATION", "warm_online", "contract_path"),
            args.output,
            "online run contract output",
        ),
        (
            ("EVALUATION", "warm_online", "training_attestation_path"),
            args.training_attestation,
            "training attestation",
        ),
        (
            ("EVALUATION", "warm_online", "training_run_contract_path"),
            args.training_run_contract,
            "online training run contract",
        ),
        (
            ("EVALUATION", "warm_online", "validation_run_contract_path"),
            args.validation_run_contract,
            "online validation run contract",
        ),
        (
            ("EVALUATION", "warm_online", "bank_directory"),
            args.bank,
            "event bank",
        ),
        (
            ("EVALUATION", "warm_online", "normalizer_contract_path"),
            args.normalizer_contract,
            "normalizer contract",
        ),
        (
            ("EVALUATION", "warm_online", "encoder_contract_path"),
            args.encoder_contract,
            "encoder contract",
        ),
        (
            ("EVALUATION", "warm_online", "camera_contract_path"),
            args.camera_contract,
            "camera contract",
        ),
        (
            ("EVALUATION", "warm_online", "m1_data_config_path"),
            args.data_config,
            "M1 data config",
        ),
        (
            ("EVALUATION", "warm_online", "dino_checkpoint_path"),
            args.dino_checkpoint,
            "DINO checkpoint",
        ),
        (
            ("EVALUATION", "warm_online", "catalog_path"),
            args.catalog,
            "episode catalog",
        ),
        (
            ("EVALUATION", "warm_online", "audit_report_path"),
            args.audit_report,
            "audit report",
        ),
        (
            ("EVALUATION", "dataset_stats_path"),
            args.normalization_stats,
            "normalization statistics",
        ),
        (
            ("data", "train", "pretrained_norm_stats"),
            args.normalization_stats,
            "processor normalization statistics",
        ),
    )
    for config_path, expected_path, label in path_bindings:
        actual = _resolved_path_value(
            _config_value(value, *config_path), label=label
        )
        expected = Path(expected_path).expanduser().resolve()
        if actual != expected:
            raise OnlineContractBuildError(
                f"resolved eval config {label} path disagrees with the bound "
                f"artifact: {actual} != {expected}"
            )

    model_base = _resolved_path_value(
        _config_value(value, "model", "base_checkpoint_path"),
        label="model base checkpoint",
    )
    online_base = _resolved_path_value(
        _config_value(
            value, "EVALUATION", "warm_online", "base_checkpoint_path"
        ),
        label="online base checkpoint",
    )
    if model_base != online_base or not model_base.is_file():
        raise OnlineContractBuildError(
            "resolved model/online base checkpoint paths must name the same file"
        )

    encoder_contract = _read_json_object(
        Path(args.encoder_contract).expanduser().resolve(),
        label="encoder contract",
    )
    _, _, _, compute_device = validate_online_encoder_contract(encoder_contract)
    configured_dino_device = _config_value(
        value, "EVALUATION", "warm_online", "dino_device"
    )
    if configured_dino_device != compute_device:
        raise OnlineContractBuildError(
            "resolved eval config DINO device disagrees with the feature-encoder "
            f"contract: {configured_dino_device!r} != {compute_device!r}"
        )


def _projection_path(value: Any, *, label: str) -> Path:
    path = _resolved_path_value(value, label=label)
    if not path.is_absolute():  # pragma: no cover - resolve above is absolute
        raise OnlineContractBuildError(f"RMBench projection {label} must be absolute")
    return path


def _validate_rmbench_policy_runtime_projection(
    value: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    """Validate the canonical projection shared with the RMBench policy."""

    if args.benchmark_profile != "robotwin":
        raise OnlineContractBuildError(
            "rmbench_policy_runtime requires --benchmark-profile robotwin"
        )
    if (
        value.get("schema") != RMBENCH_RUNTIME_PROJECTION_SCHEMA
        or value.get("schema_version") != RMBENCH_RUNTIME_PROJECTION_VERSION
        or value.get("benchmark_profile") != "robotwin"
    ):
        raise OnlineContractBuildError("RMBench policy-runtime projection schema mismatch")
    task = value.get("task")
    experiment = value.get("experiment")
    action = value.get("action_generation")
    retrieval = value.get("retrieval")
    artifacts = value.get("artifact_paths")
    processor_recipe = value.get("processor_recipe")
    model_behavior = value.get("model_behavior")
    if not all(
        isinstance(item, Mapping)
        for item in (
            task,
            experiment,
            action,
            retrieval,
            artifacts,
            processor_recipe,
            model_behavior,
        )
    ):
        raise OnlineContractBuildError(
            "RMBench policy-runtime projection is missing a required mapping"
        )
    assert isinstance(task, Mapping)
    assert isinstance(action, Mapping)
    assert isinstance(retrieval, Mapping)
    assert isinstance(artifacts, Mapping)
    expected_scalars = (
        (task.get("suite"), args.task_suite, "task suite"),
        (task.get("task_id"), args.task_id, "task id"),
        (task.get("task_description"), args.task_description, "task description"),
        (task.get("root_seed"), args.root_seed, "root seed"),
        (action.get("action_horizon"), args.action_horizon, "action horizon"),
        (action.get("source_policy"), args.source_policy, "source policy"),
        (action.get("memory_sigma"), args.memory_sigma, "memory sigma"),
        (retrieval.get("top_k"), args.top_k, "retrieval top-k"),
    )
    for actual, expected, label in expected_scalars:
        if isinstance(expected, float):
            try:
                agrees = float(actual) == float(expected)
            except (TypeError, ValueError):
                agrees = False
        else:
            agrees = actual == expected
        if not agrees:
            raise OnlineContractBuildError(
                f"RMBench policy-runtime {label} disagrees with contract field: "
                f"{actual!r} != {expected!r}"
            )
    namespace = retrieval.get("evaluation_namespace")
    if namespace != args.evaluation_namespace:
        raise OnlineContractBuildError(
            "RMBench policy-runtime evaluation namespace disagrees with contract"
        )
    if retrieval.get("enabled") is not True or retrieval.get("mode") != "full_retrospection":
        raise OnlineContractBuildError(
            "RMBench policy-runtime must enable full_retrospection"
        )

    expected_paths = {
        "online_contract": args.output,
        "training_attestation": args.training_attestation,
        "training_run_contract": args.training_run_contract,
        "validation_run_contract": args.validation_run_contract,
        "event_bank": args.bank,
        "normalizer_contract": args.normalizer_contract,
        "encoder_contract": args.encoder_contract,
        "camera_contract": args.camera_contract,
        "m1_data_config": args.data_config,
        "dino_checkpoint": args.dino_checkpoint,
        "catalog": args.catalog,
        "audit_report": args.audit_report,
        "seed_protocol": args.initial_states,
        "task_definition": args.bddl,
        "vae_checkpoint": args.vae_checkpoint,
        "text_encoder": args.text_encoder,
        "tokenizer": args.tokenizer,
        "warm_checkpoint": args.warm_checkpoint,
        "normalization_stats": args.normalization_stats,
    }
    for name, expected in expected_paths.items():
        actual = _projection_path(artifacts.get(name), label=f"artifact {name}")
        if actual != Path(expected).expanduser().resolve():
            raise OnlineContractBuildError(
                f"RMBench policy-runtime artifact {name} path disagrees with contract"
            )

    base_checkpoint = _projection_path(
        artifacts.get("base_checkpoint"), label="artifact base_checkpoint"
    )
    if not base_checkpoint.is_file():
        raise OnlineContractBuildError(
            "RMBench policy-runtime base checkpoint does not exist"
        )
    if model_behavior.get("base_checkpoint_path") is not None and (
        _projection_path(
            model_behavior.get("base_checkpoint_path"),
            label="model base checkpoint",
        )
        != base_checkpoint
    ):
        raise OnlineContractBuildError(
            "RMBench model behavior and artifact base checkpoint paths disagree"
        )


def _initial_states_digest(path: Path) -> str:
    if path.suffix.lower() != ".npy":
        raise OnlineContractBuildError("--initial-states must name a .npy file")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise OnlineContractBuildError(f"cannot load initial states at {path}") from exc
    if not isinstance(array, np.ndarray) or array.ndim < 2 or array.shape[0] <= 0:
        raise OnlineContractBuildError(
            "initial states must be a non-empty numeric array with rank >= 2"
        )
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise OnlineContractBuildError("initial states must be finite numeric values")
    return sha256_array(np.ascontiguousarray(array))


def _git_identity(repository: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise OnlineContractBuildError("cannot inspect WARM Git identity") from exc
    return commit, bool(status.strip())


def _write_atomic(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"online contract already exists at {path}")
    encoded = (
        json.dumps(
            dict(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Preserve the no-overwrite contract even against a writer that
            # does not cooperate with our advisory artifact lock.
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def _assert_file_digest(path: Path, expected: str, *, label: str) -> None:
    if sha256_file(path) != expected:
        raise OnlineContractBuildError(f"{label} changed during contract build")


def _build_contract(args: argparse.Namespace) -> WarmOnlineRunContract:
    source_path = args.training_run_contract.expanduser().resolve()
    validation_source_path = args.validation_run_contract.expanduser().resolve()
    warm_checkpoint = args.warm_checkpoint.expanduser().resolve()
    training_attestation_path = args.training_attestation.expanduser().resolve()
    bank_path = args.bank.expanduser().resolve()
    normalizer_path = args.normalizer_contract.expanduser().resolve()
    encoder_path = args.encoder_contract.expanduser().resolve()
    camera_path = args.camera_contract.expanduser().resolve()
    data_config_path = args.data_config.expanduser().resolve()
    dino_path = args.dino_checkpoint.expanduser().resolve()
    stats_path = args.normalization_stats.expanduser().resolve()
    catalog_path = args.catalog.expanduser().resolve()
    audit_path = args.audit_report.expanduser().resolve()
    config_path = args.resolved_eval_config.expanduser().resolve()
    vae_path = args.vae_checkpoint.expanduser().resolve()
    text_path = args.text_encoder.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    initial_states_path = args.initial_states.expanduser().resolve()
    bddl_path = args.bddl.expanduser().resolve()

    for label, path in (
        ("training run contract", source_path),
        ("validation run contract", validation_source_path),
        ("WARM checkpoint", warm_checkpoint),
        ("training attestation", training_attestation_path),
        ("normalizer contract", normalizer_path),
        ("encoder contract", encoder_path),
        ("camera contract", camera_path),
        ("M1 data config", data_config_path),
        ("normalization stats", stats_path),
        ("catalog", catalog_path),
        ("audit report", audit_path),
        ("resolved eval config", config_path),
        ("VAE checkpoint", vae_path),
        ("initial states", initial_states_path),
        ("BDDL", bddl_path),
    ):
        if not path.is_file():
            raise OnlineContractBuildError(f"{label} is not a regular file: {path}")
    if not bank_path.is_dir():
        raise OnlineContractBuildError(f"event bank is not a directory: {bank_path}")
    if not dino_path.exists():
        raise OnlineContractBuildError(f"DINO checkpoint does not exist: {dino_path}")
    if not text_path.exists():
        raise OnlineContractBuildError(f"text encoder does not exist: {text_path}")
    if not tokenizer_path.exists():
        raise OnlineContractBuildError(f"tokenizer does not exist: {tokenizer_path}")

    source = WarmSourceRunContract.from_dict(
        _read_json_object(source_path, label="training run contract")
    )
    if source.query_split != "train":
        raise OnlineContractBuildError(
            "online retrieval must be bound to the train event-bank/source "
            "contract, not a held-out dev query contract"
        )
    validation_source = WarmSourceRunContract.from_dict(
        _read_json_object(
            validation_source_path, label="validation run contract"
        )
    )
    if validation_source.query_split != "dev":
        raise OnlineContractBuildError(
            "online evaluation requires a dev validation run contract"
        )
    shared_source_fields = (
        "bank_manifest_sha256",
        "bank_content_sha256",
        "catalog_sha256",
        "audit_sha256",
        "normalization_stats_sha256",
        "action_space_contract_sha256",
        "base_checkpoint_sha256",
        "global_sample_stride",
        "action_horizon",
        "action_dim",
    )
    if any(
        getattr(source, field) != getattr(validation_source, field)
        for field in shared_source_fields
    ):
        raise OnlineContractBuildError(
            "train/dev source contracts disagree on shared artifacts"
        )
    if (
        source.candidate_manifest_sha256
        == validation_source.candidate_manifest_sha256
        or source.query_corpus_sha256 == validation_source.query_corpus_sha256
    ):
        raise OnlineContractBuildError(
            "validation contract must bind an independent dev candidate/query corpus"
        )
    training_attestation_file_sha = sha256_file(training_attestation_path)
    try:
        training_attestation = verify_training_attestation(
            warm_checkpoint, training_attestation_path
        )
    except (OSError, TrainingAttestationError) as exc:
        raise OnlineContractBuildError(
            "WARM checkpoint has no valid immutable training attestation"
        ) from exc
    if sha256_file(training_attestation_path) != training_attestation_file_sha:
        raise OnlineContractBuildError(
            "training attestation changed while it was verified"
        )
    if training_attestation.source_policy != args.source_policy:
        raise OnlineContractBuildError(
            "training attestation source policy disagrees with the online policy"
        )
    if training_attestation.train_source_contract_sha256 != source.sha256:
        raise OnlineContractBuildError(
            "training attestation does not bind the supplied train source contract"
        )
    if training_attestation.dev_source_contract_sha256 != validation_source.sha256:
        raise OnlineContractBuildError(
            "training attestation does not bind the supplied dev source contract"
        )
    if training_attestation.base_checkpoint_sha256 != source.base_checkpoint_sha256:
        raise OnlineContractBuildError(
            "training attestation base checkpoint disagrees with the source contract"
        )
    bank = EventBank.load(bank_path)
    summary = validate_warm_v1_bank(
        bank,
        expected_action_horizon=args.action_horizon,
        expected_action_dim=args.action_dim,
    )
    if bank.manifest is None:
        raise OnlineContractBuildError("loaded event bank has no manifest")
    manifest = bank.manifest
    bank_manifest_path = bank_path / MANIFEST_FILENAME
    bank_manifest_sha = sha256_file(bank_manifest_path)
    bank_content_sha = canonical_event_bank_content_hash(manifest.content_hashes)
    if bank_manifest_sha != source.bank_manifest_sha256:
        raise OnlineContractBuildError("event bank does not match training contract")
    if bank_content_sha != source.bank_content_sha256:
        raise OnlineContractBuildError("event-bank content does not match training contract")
    if (summary.action_horizon, summary.action_dim) != (
        source.action_horizon,
        source.action_dim,
    ):
        raise OnlineContractBuildError("online action shape differs from training contract")

    normalizer_sha = sha256_file(normalizer_path)
    encoder_sha = sha256_file(encoder_path)
    camera_sha = sha256_file(camera_path)
    if manifest.action_normalizer.get("file_sha256") != normalizer_sha:
        raise OnlineContractBuildError("normalizer contract does not match event bank")
    if manifest.encoder.get("file_sha256") != encoder_sha:
        raise OnlineContractBuildError("encoder contract does not match event bank")
    if manifest.camera_layout.get("file_sha256") != camera_sha:
        raise OnlineContractBuildError("camera contract does not match event bank")
    encoder_contract = _read_json_object(encoder_path, label="encoder contract")
    camera_contract = _read_json_object(camera_path, label="camera contract")
    normalizer_contract = _read_json_object(
        normalizer_path, label="normalizer contract"
    )
    if manifest.encoder.get("contract") != encoder_contract:
        raise OnlineContractBuildError(
            "encoder contract payload differs from the event-bank manifest"
        )
    if manifest.camera_layout.get("contract") != camera_contract:
        raise OnlineContractBuildError(
            "camera contract payload differs from the event-bank manifest"
        )
    validate_online_encoder_contract(encoder_contract)
    validate_online_camera_contract(
        camera_contract, benchmark_profile=args.benchmark_profile
    )
    encoder_runtime = encoder_contract.get("runtime")
    if not isinstance(encoder_runtime, Mapping):
        raise OnlineContractBuildError(
            "encoder contract has no numerical runtime fingerprint"
        )
    encoder_runtime_sha = sha256_canonical_json(dict(encoder_runtime))
    try:
        m1_processor_recipe, data_config_sha = load_m1_data_config(
            data_config_path, profile=args.benchmark_profile
        )
    except ProcessorContractError as exc:
        raise OnlineContractBuildError(
            "M1 data config does not contain the exact factual processor recipe"
        ) from exc
    dino = encoder_contract.get("dino")
    if not isinstance(dino, Mapping):
        raise OnlineContractBuildError("encoder contract has no DINO mapping")
    dino_sha, dino_count = sha256_path_tree(dino_path)
    if dino.get("checkpoint_tree_sha256") != dino_sha or int(
        dino.get("checkpoint_file_count", -1)
    ) != dino_count:
        raise OnlineContractBuildError("DINO checkpoint differs from encoder contract")

    action_mapping = manifest.action_normalizer.get("contract")
    if not isinstance(action_mapping, Mapping):
        raise OnlineContractBuildError("event bank has no action-space contract")
    if action_mapping != normalizer_contract:
        raise OnlineContractBuildError(
            "normalizer contract payload differs from the event-bank manifest"
        )
    action_contract = validate_action_space_contract(action_mapping)
    action_contract_sha = sha256_canonical_json(action_contract.to_dict())
    stats_sha = sha256_file(stats_path)
    if stats_sha != source.normalization_stats_sha256:
        raise OnlineContractBuildError("normalization stats differ from training contract")
    if stats_sha != action_contract.normalization_stats_sha256:
        raise OnlineContractBuildError("normalization stats differ from action contract")
    if action_contract_sha != source.action_space_contract_sha256:
        raise OnlineContractBuildError("action-space contract differs from training contract")
    if action_contract.action_dim != args.action_dim:
        raise OnlineContractBuildError("action dimension differs from action contract")

    catalog = EpisodeCatalog.load(catalog_path)
    audit = load_audit_report(audit_path)
    catalog_sha = catalog.content_sha256
    audit_sha = audit.report_sha256
    if audit.catalog_sha256 != catalog_sha:
        raise OnlineContractBuildError(
            "audit report is bound to a different episode catalog"
        )
    if catalog_sha != source.catalog_sha256 or audit_sha != source.audit_sha256:
        raise OnlineContractBuildError("catalog/audit differ from training contract")
    data_binding = manifest.provenance.get("data_binding")
    if data_binding != {
        "catalog_sha256": catalog_sha,
        "audit_report_sha256": audit_sha,
        "split": "train",
    }:
        raise OnlineContractBuildError(
            "event bank is not bound to the supplied audited train catalog"
        )

    resolved_config = _read_resolved_config(config_path)
    _validate_resolved_config(resolved_config, args)
    if args.resolved_config_binding == "rmbench_policy_runtime":
        resolved_processor_recipe = resolved_config.get("processor_recipe")
        retrieval_projection = resolved_config.get("retrieval")
        if not isinstance(retrieval_projection, Mapping) or (
            retrieval_projection.get("dino_device") != compute_device
        ):
            raise OnlineContractBuildError(
                "RMBench policy-runtime DINO device differs from encoder contract"
            )
    else:
        try:
            resolved_processor_recipe = extract_m1_processor_recipe(
                resolved_config, profile=args.benchmark_profile
            )
        except ProcessorContractError as exc:
            raise OnlineContractBuildError(
                "resolved rollout processor is not the exact M1 processor recipe"
            ) from exc
    if resolved_processor_recipe != m1_processor_recipe:
        raise OnlineContractBuildError(
            "resolved rollout processor recipe differs from the M1 data config"
        )
    if encoder_contract.get("data_config_sha256") != data_config_sha:
        raise OnlineContractBuildError(
            "M1 data config does not match the feature-encoder contract"
        )
    if args.resolved_config_binding == "rmbench_policy_runtime":
        base_checkpoint_path = _projection_path(
            _config_value(resolved_config, "artifact_paths", "base_checkpoint"),
            label="online base checkpoint",
        )
    else:
        base_checkpoint_path = _resolved_path_value(
            _config_value(
                resolved_config,
                "EVALUATION",
                "warm_online",
                "base_checkpoint_path",
            ),
            label="online base checkpoint",
        )
    if sha256_file(base_checkpoint_path) != source.base_checkpoint_sha256:
        raise OnlineContractBuildError(
            "resolved base checkpoint does not match the training run contract"
        )
    repository = Path(__file__).resolve().parent.parent
    git_commit, git_dirty = _git_identity(repository)
    if git_dirty:
        raise OnlineContractBuildError(
            "formal online rollout contract requires a clean Git tree"
        )
    if training_attestation.git_commit != git_commit:
        raise OnlineContractBuildError(
            "training attestation and online contract must bind the same Git commit"
        )

    return WarmOnlineRunContract(
        training_run_contract_sha256=source.sha256,
        validation_run_contract_sha256=validation_source.sha256,
        warm_checkpoint_sha256=sha256_file(warm_checkpoint),
        training_attestation_sha256=training_attestation_file_sha,
        shared_training_recipe_sha256=training_attestation.shared_recipe_sha256,
        training_runtime_sha256=training_attestation.training_runtime_sha256,
        bank_manifest_sha256=bank_manifest_sha,
        bank_content_sha256=bank_content_sha,
        encoder_contract_sha256=encoder_sha,
        encoder_runtime_sha256=encoder_runtime_sha,
        camera_contract_sha256=camera_sha,
        m1_data_config_sha256=data_config_sha,
        dino_checkpoint_tree_sha256=dino_sha,
        dino_checkpoint_file_count=dino_count,
        normalization_stats_sha256=stats_sha,
        action_space_contract_sha256=action_contract_sha,
        catalog_sha256=catalog_sha,
        audit_sha256=audit_sha,
        resolved_eval_config_sha256=sha256_canonical_json(resolved_config),
        vae_checkpoint_sha256=sha256_file(vae_path),
        text_encoder_tree_sha256=sha256_path_tree(text_path)[0],
        tokenizer_tree_sha256=sha256_path_tree(tokenizer_path)[0],
        evaluation_namespace_sha256=sha256_canonical_json(
            {"evaluation_namespace": args.evaluation_namespace}
        ),
        task_suite=args.task_suite,
        task_id=args.task_id,
        task_description=args.task_description,
        root_seed=args.root_seed,
        initial_states_sha256=_initial_states_digest(initial_states_path),
        bddl_sha256=sha256_file(bddl_path),
        retrieval_implementation=ONLINE_RETRIEVAL_IMPLEMENTATION,
        top_k=args.top_k,
        source_policy=args.source_policy,
        memory_sigma=args.memory_sigma,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        git_commit=git_commit,
        git_dirty=False,
    )


def _assert_contract_inputs_unchanged(
    args: argparse.Namespace,
    contract: WarmOnlineRunContract,
) -> None:
    """Re-read every identity-bearing artifact inside the publication claim."""

    source_path = args.training_run_contract.expanduser().resolve()
    source = WarmSourceRunContract.from_dict(
        _read_json_object(source_path, label="training run contract")
    )
    if source.sha256 != contract.training_run_contract_sha256:
        raise OnlineContractBuildError("training run contract changed during build")
    validation_source = WarmSourceRunContract.from_dict(
        _read_json_object(
            args.validation_run_contract.expanduser().resolve(),
            label="validation run contract",
        )
    )
    if validation_source.sha256 != contract.validation_run_contract_sha256:
        raise OnlineContractBuildError(
            "validation run contract changed during build"
        )

    direct_files = (
        (args.warm_checkpoint, contract.warm_checkpoint_sha256, "WARM checkpoint"),
        (
            args.training_attestation,
            contract.training_attestation_sha256,
            "training attestation",
        ),
        (args.encoder_contract, contract.encoder_contract_sha256, "encoder contract"),
        (args.camera_contract, contract.camera_contract_sha256, "camera contract"),
        (
            args.normalization_stats,
            contract.normalization_stats_sha256,
            "normalization stats",
        ),
        (args.vae_checkpoint, contract.vae_checkpoint_sha256, "VAE checkpoint"),
        (args.bddl, contract.bddl_sha256, "BDDL"),
    )
    for raw_path, expected, label in direct_files:
        _assert_file_digest(
            raw_path.expanduser().resolve(), expected, label=label
        )

    try:
        attestation = verify_training_attestation(
            args.warm_checkpoint.expanduser().resolve(),
            args.training_attestation.expanduser().resolve(),
        )
    except (OSError, TrainingAttestationError) as exc:
        raise OnlineContractBuildError(
            "training attestation became invalid during contract build"
        ) from exc
    if (
        attestation.shared_recipe_sha256 != contract.shared_training_recipe_sha256
        or attestation.training_runtime_sha256 != contract.training_runtime_sha256
        or attestation.source_policy != contract.source_policy
        or attestation.train_source_contract_sha256
        != contract.training_run_contract_sha256
        or attestation.dev_source_contract_sha256
        != contract.validation_run_contract_sha256
        or attestation.base_checkpoint_sha256 != source.base_checkpoint_sha256
        or attestation.git_commit != contract.git_commit
    ):
        raise OnlineContractBuildError(
            "training attestation facts changed during contract build"
        )

    catalog = EpisodeCatalog.load(args.catalog.expanduser().resolve())
    audit = load_audit_report(args.audit_report.expanduser().resolve())
    if catalog.content_sha256 != contract.catalog_sha256:
        raise OnlineContractBuildError("catalog changed during contract build")
    if (
        audit.report_sha256 != contract.audit_sha256
        or audit.catalog_sha256 != catalog.content_sha256
    ):
        raise OnlineContractBuildError("audit report changed during contract build")

    config = _read_resolved_config(args.resolved_eval_config.expanduser().resolve())
    _validate_resolved_config(config, args)
    if sha256_canonical_json(config) != contract.resolved_eval_config_sha256:
        raise OnlineContractBuildError("resolved eval config changed during build")
    if args.resolved_config_binding == "rmbench_policy_runtime":
        base_checkpoint_path = _projection_path(
            _config_value(config, "artifact_paths", "base_checkpoint"),
            label="online base checkpoint",
        )
    else:
        base_checkpoint_path = _resolved_path_value(
            _config_value(
                config,
                "EVALUATION",
                "warm_online",
                "base_checkpoint_path",
            ),
            label="online base checkpoint",
        )
    if sha256_file(base_checkpoint_path) != source.base_checkpoint_sha256:
        raise OnlineContractBuildError("base checkpoint changed during contract build")
    if _initial_states_digest(
        args.initial_states.expanduser().resolve()
    ) != contract.initial_states_sha256:
        raise OnlineContractBuildError("initial states changed during build")

    dino_sha, dino_count = sha256_path_tree(
        args.dino_checkpoint.expanduser().resolve()
    )
    if (
        dino_sha != contract.dino_checkpoint_tree_sha256
        or dino_count != contract.dino_checkpoint_file_count
    ):
        raise OnlineContractBuildError("DINO checkpoint changed during build")
    if sha256_path_tree(args.text_encoder.expanduser().resolve())[0] != (
        contract.text_encoder_tree_sha256
    ):
        raise OnlineContractBuildError("text encoder changed during build")
    if sha256_path_tree(args.tokenizer.expanduser().resolve())[0] != (
        contract.tokenizer_tree_sha256
    ):
        raise OnlineContractBuildError("tokenizer changed during build")
    if sha256_file(args.data_config.expanduser().resolve()) != (
        contract.m1_data_config_sha256
    ):
        raise OnlineContractBuildError("M1 data config changed during build")

    bank_path = args.bank.expanduser().resolve()
    if sha256_file(bank_path / MANIFEST_FILENAME) != contract.bank_manifest_sha256:
        raise OnlineContractBuildError("event-bank manifest changed during build")
    bank = EventBank.load(bank_path)
    if bank.manifest is None or canonical_event_bank_content_hash(
        bank.manifest.content_hashes
    ) != contract.bank_content_sha256:
        raise OnlineContractBuildError("event-bank content changed during build")
    if bank.manifest.action_normalizer.get("file_sha256") != sha256_file(
        args.normalizer_contract.expanduser().resolve()
    ):
        raise OnlineContractBuildError("normalizer contract changed during build")

    git_commit, git_dirty = _git_identity(Path(__file__).resolve().parent.parent)
    if git_dirty or git_commit != contract.git_commit:
        raise OnlineContractBuildError("Git identity changed during contract build")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    direct_inputs = {
        Path(getattr(args, field)).expanduser().resolve()
        for field in (
            "training_run_contract",
            "validation_run_contract",
            "warm_checkpoint",
            "training_attestation",
            "bank",
            "normalizer_contract",
            "encoder_contract",
            "camera_contract",
            "data_config",
            "dino_checkpoint",
            "normalization_stats",
            "catalog",
            "audit_report",
            "resolved_eval_config",
            "vae_checkpoint",
            "text_encoder",
            "tokenizer",
            "initial_states",
            "bddl",
        )
    }
    if output in direct_inputs:
        raise OnlineContractBuildError(
            "--output must not overwrite an identity-bearing input file"
        )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"online contract already exists at {output}")
    forbidden_roots = (
        args.bank.expanduser().resolve(),
        args.dino_checkpoint.expanduser().resolve(),
        args.text_encoder.expanduser().resolve(),
        args.tokenizer.expanduser().resolve(),
    )
    for root in forbidden_roots:
        if root.is_dir():
            try:
                output.relative_to(root)
            except ValueError:
                pass
            else:
                raise OnlineContractBuildError(
                    "--output must be outside immutable artifact directories"
                )

    contract = _build_contract(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.warm-artifact.lock"
    with artifact_claim(lock_path, purpose=f"publish WARM online contract: {output}"):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"online contract already exists at {output}")
        _assert_contract_inputs_unchanged(args, contract)
        _write_atomic(output, contract.to_dict(), overwrite=args.overwrite)

    print(
        json.dumps(
            {
                "schema": SUMMARY_SCHEMA,
                "version": SUMMARY_SCHEMA_VERSION,
                "output": str(output),
                "online_run_contract_sha256": contract.sha256,
                "contract": contract.to_dict(),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
