"""Canonical model-behaviour projection for formal RMBench WARM runs.

The official runner owns timestamps, output directories, telemetry paths and
worker placement.  None of those values changes the policy.  Hashing the
entire Hydra tree would therefore make a pre-built online contract impossible
to reuse for the exact same policy on another worker.  This module defines the
single projection shared by the contract builder and the deployed policy.

The projection deliberately keeps paths to immutable input artifacts.  Their
contents are independently hashed by :class:`WarmOnlineRunContract`; retaining
the normalized paths additionally prevents a runtime from silently swapping a
different input location after the contract was built.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

from fastwam.memory.processor_contract import (
    ProcessorContractError,
    extract_m1_processor_recipe,
)


RMBENCH_RUNTIME_PROJECTION_SCHEMA = "warm.rmbench-policy-runtime"
RMBENCH_RUNTIME_PROJECTION_VERSION = 1
RMBENCH_ABLATION_MODES = frozenset(
    {"context_only", "source_only_no_consequence", "full"}
)
RMBENCH_MEMORY_CORRUPTIONS = frozenset(
    {"clean", "wrong_event", "reversed_action", "phase_shift", "effect_mismatch"}
)
RMBENCH_ODE_STEPS = frozenset({2, 4, 8, 10})


class RMBenchRuntimeProjectionError(ValueError):
    """Raised when a runtime cannot be represented unambiguously."""


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RMBenchRuntimeProjectionError(f"{field} must be a mapping")
    return value


def _get(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise RMBenchRuntimeProjectionError(
                f"resolved config is missing {'.'.join(path)}"
            )
        current = current[key]
    return current


def _json_value(value: Any, *, field: str) -> Any:
    """Return a detached, finite JSON value with deterministic key types."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise RMBenchRuntimeProjectionError(
            f"{field} must contain only finite JSON-compatible values"
        ) from exc


def _absolute_path(value: Any, *, field: str) -> str:
    if value is None or not str(value).strip():
        raise RMBenchRuntimeProjectionError(f"{field} is required")
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    if not path.is_absolute():
        raise RMBenchRuntimeProjectionError(
            f"{field} must be absolute in a formal RMBench runtime"
        )
    return str(path.resolve())


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise RMBenchRuntimeProjectionError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RMBenchRuntimeProjectionError(f"{field} must be an integer") from exc
    if result < minimum:
        raise RMBenchRuntimeProjectionError(f"{field} must be >= {minimum}")
    return result


def _float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise RMBenchRuntimeProjectionError(f"{field} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RMBenchRuntimeProjectionError(f"{field} must be a number") from exc
    if not (float("-inf") < result < float("inf")):
        raise RMBenchRuntimeProjectionError(f"{field} must be finite")
    return result


def _optional_float(value: Any, *, field: str) -> float | None:
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "none", "null"}):
        return None
    return _float(value, field=field)


def _bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    raise RMBenchRuntimeProjectionError(f"{field} must be boolean")


def _same(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise RMBenchRuntimeProjectionError(
            f"runtime argument {field} disagrees with resolved config: "
            f"{actual!r} != {expected!r}"
        )


def build_rmbench_policy_runtime_projection(
    resolved_config: Mapping[str, Any],
    runtime_args: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a resolved policy configuration onto behaviour-bearing fields.

    ``runtime_args`` is the exact mapping received by the official policy.  It
    is used both to bind inputs that do not live in Hydra (the seed protocol,
    task implementation and explicitly loaded model artifacts) and to
    cross-check duplicated controls.  The returned mapping never contains
    output directories, telemetry paths, timestamps, GPU worker IDs or runner
    scheduling controls.
    """

    cfg = _mapping(resolved_config, field="resolved_config")
    args = _mapping(runtime_args, field="runtime_args")
    evaluation = _mapping(_get(cfg, "EVALUATION"), field="EVALUATION")
    online = _mapping(
        _get(cfg, "EVALUATION", "warm_online"),
        field="EVALUATION.warm_online",
    )
    model = _mapping(_get(cfg, "model"), field="model")

    suite = str(_get(cfg, "EVALUATION", "task_suite_name"))
    task_id = _integer(_get(cfg, "EVALUATION", "task_id"), field="task_id")
    task_description = str(_get(cfg, "EVALUATION", "task_description"))
    root_seed = _integer(_get(cfg, "seed"), field="seed")
    if suite != "rmbench" or not task_description:
        raise RMBenchRuntimeProjectionError(
            "formal RMBench runtime requires suite='rmbench' and a task description"
        )
    _same(str(args.get("task_name")), task_description, field="task_name")
    _same(_integer(args.get("seed"), field="seed"), root_seed, field="seed")

    experiment_id = str(online.get("experiment_id", ""))
    ablation_mode = str(online.get("ablation_mode", ""))
    memory_corruption = str(online.get("memory_corruption", ""))
    ode_steps = _integer(
        evaluation.get("num_inference_steps"),
        field="EVALUATION.num_inference_steps",
        minimum=1,
    )
    if not experiment_id or experiment_id.strip() != experiment_id:
        raise RMBenchRuntimeProjectionError("experiment_id must be normalized")
    if ablation_mode not in RMBENCH_ABLATION_MODES:
        raise RMBenchRuntimeProjectionError("unsupported RMBench ablation_mode")
    if memory_corruption not in RMBENCH_MEMORY_CORRUPTIONS:
        raise RMBenchRuntimeProjectionError("unsupported RMBench memory_corruption")
    if ode_steps not in RMBENCH_ODE_STEPS:
        raise RMBenchRuntimeProjectionError("RMBench ODE steps must be 2, 4, 8, or 10")
    if ablation_mode != "full" and memory_corruption != "clean":
        raise RMBenchRuntimeProjectionError(
            "memory corruption is defined only for full WARM"
        )
    duplicated = (
        (str(args.get("warm_experiment_id")), experiment_id, "warm_experiment_id"),
        (str(args.get("warm_ablation_mode")), ablation_mode, "warm_ablation_mode"),
        (
            str(args.get("warm_memory_corruption")),
            memory_corruption,
            "warm_memory_corruption",
        ),
        (
            _integer(args.get("num_inference_steps"), field="num_inference_steps", minimum=1),
            ode_steps,
            "num_inference_steps",
        ),
        (
            _integer(args.get("action_horizon"), field="action_horizon", minimum=1),
            _integer(evaluation.get("action_horizon"), field="action_horizon", minimum=1),
            "action_horizon",
        ),
        (
            _integer(args.get("replan_steps"), field="replan_steps", minimum=1),
            _integer(evaluation.get("replan_steps"), field="replan_steps", minimum=1),
            "replan_steps",
        ),
    )
    for actual, expected, field in duplicated:
        _same(actual, expected, field=field)
    _same(
        _integer(online.get("num_inference_steps"), field="online.num_inference_steps", minimum=1),
        ode_steps,
        field="online.num_inference_steps",
    )

    artifact_args = {
        "online_contract": "warm_online_contract_path",
        "training_attestation": "warm_training_attestation_path",
        "training_run_contract": "warm_training_run_contract_path",
        "validation_run_contract": "warm_validation_run_contract_path",
        "base_checkpoint": "warm_base_checkpoint_path",
        "event_bank": "warm_bank_directory",
        "normalizer_contract": "warm_normalizer_contract_path",
        "encoder_contract": "warm_encoder_contract_path",
        "camera_contract": "warm_camera_contract_path",
        "m1_data_config": "warm_m1_data_config_path",
        "dino_checkpoint": "warm_dino_checkpoint_path",
        "catalog": "warm_catalog_path",
        "audit_report": "warm_audit_report_path",
        "seed_protocol": "warm_initial_states_path",
        "task_definition": "warm_task_definition_path",
        "vae_checkpoint": "warm_vae_checkpoint_path",
        "text_encoder": "warm_text_encoder_path",
        "tokenizer": "warm_tokenizer_path",
    }
    artifact_paths = {
        name: _absolute_path(args.get(argument), field=argument)
        for name, argument in artifact_args.items()
    }
    artifact_paths["warm_checkpoint"] = _absolute_path(
        args.get("ckpt_setting"), field="ckpt_setting"
    )
    artifact_paths["normalization_stats"] = _absolute_path(
        args.get("dataset_stats_path"), field="dataset_stats_path"
    )

    cfg_artifact_paths = {
        "online_contract": online.get("contract_path"),
        "training_attestation": online.get("training_attestation_path"),
        "training_run_contract": online.get("training_run_contract_path"),
        "validation_run_contract": online.get("validation_run_contract_path"),
        "base_checkpoint": online.get("base_checkpoint_path"),
        "event_bank": online.get("bank_directory"),
        "normalizer_contract": online.get("normalizer_contract_path"),
        "encoder_contract": online.get("encoder_contract_path"),
        "camera_contract": online.get("camera_contract_path"),
        "m1_data_config": online.get("m1_data_config_path"),
        "dino_checkpoint": online.get("dino_checkpoint_path"),
        "catalog": online.get("catalog_path"),
        "audit_report": online.get("audit_report_path"),
        "warm_checkpoint": cfg.get("ckpt"),
        "normalization_stats": evaluation.get("dataset_stats_path"),
    }
    for name, configured in cfg_artifact_paths.items():
        _same(
            _absolute_path(configured, field=f"config artifact {name}"),
            artifact_paths[name],
            field=f"artifact.{name}",
        )

    try:
        processor_recipe = extract_m1_processor_recipe(cfg, profile="robotwin")
    except ProcessorContractError as exc:
        raise RMBenchRuntimeProjectionError(
            "resolved RMBench processor is not the exact Robotwin profile"
        ) from exc

    sampler = {
        "num_inference_steps": ode_steps,
        "sigma_shift": _optional_float(
            evaluation.get("sigma_shift"), field="sigma_shift"
        ),
        "text_cfg_scale": _float(
            evaluation.get("text_cfg_scale"), field="text_cfg_scale"
        ),
        "negative_prompt": str(evaluation.get("negative_prompt", "")),
        "rand_device": str(evaluation.get("rand_device")),
        "tiled": _bool(evaluation.get("tiled"), field="tiled"),
    }
    for arg_name, expected in (
        ("sigma_shift", sampler["sigma_shift"]),
        ("text_cfg_scale", sampler["text_cfg_scale"]),
        ("negative_prompt", sampler["negative_prompt"]),
        ("rand_device", sampler["rand_device"]),
        ("tiled", sampler["tiled"]),
    ):
        raw = args.get(arg_name)
        if arg_name == "sigma_shift":
            actual = _optional_float(raw, field=arg_name)
        elif arg_name == "text_cfg_scale":
            actual = _float(raw, field=arg_name)
        elif arg_name == "tiled":
            actual = _bool(raw, field=arg_name)
        else:
            actual = str(raw or "")
        _same(actual, expected, field=arg_name)

    top_k = _integer(online.get("top_k"), field="warm_online.top_k", minimum=1)
    _same(
        _integer(args.get("warm_top_k"), field="warm_top_k", minimum=1),
        top_k,
        field="warm_top_k",
    )
    namespace = str(online.get("evaluation_namespace", ""))
    _same(
        str(args.get("warm_evaluation_namespace")),
        namespace,
        field="warm_evaluation_namespace",
    )

    projection = {
        "schema": RMBENCH_RUNTIME_PROJECTION_SCHEMA,
        "schema_version": RMBENCH_RUNTIME_PROJECTION_VERSION,
        "benchmark_profile": "robotwin",
        "task": {
            "suite": suite,
            "task_id": task_id,
            "task_description": task_description,
            "root_seed": root_seed,
        },
        "experiment": {
            "experiment_id": experiment_id,
            "ablation_mode": ablation_mode,
            "memory_corruption": memory_corruption,
        },
        "action_generation": {
            "action_horizon": _integer(
                evaluation.get("action_horizon"), field="action_horizon", minimum=1
            ),
            "replan_steps": _integer(
                evaluation.get("replan_steps"), field="replan_steps", minimum=1
            ),
            "source_policy": str(model.get("source_policy")),
            "memory_sigma": _float(model.get("memory_sigma"), field="memory_sigma"),
            "sampler": sampler,
        },
        "retrieval": {
            "enabled": _bool(online.get("enabled"), field="warm_online.enabled"),
            "mode": str(online.get("mode")),
            "evaluation_namespace": namespace,
            "top_k": top_k,
            "dino_device": str(online.get("dino_device")),
            "dino_batch_size": _integer(
                online.get("dino_batch_size"), field="dino_batch_size", minimum=1
            ),
        },
        "numerics": {
            "mixed_precision": str(args.get("mixed_precision")),
            "device": str(args.get("device")),
        },
        "artifact_paths": artifact_paths,
        "processor_recipe": processor_recipe,
        "model_behavior": _json_value(model, field="model"),
    }
    if projection["retrieval"]["mode"] != "full_retrospection":
        raise RMBenchRuntimeProjectionError(
            "formal RMBench projection requires full_retrospection mode"
        )
    if not projection["retrieval"]["enabled"]:
        raise RMBenchRuntimeProjectionError("formal RMBench retrieval must be enabled")
    if _bool(evaluation.get("visualize_future_video"), field="visualize_future_video"):
        raise RMBenchRuntimeProjectionError("future-video inference must be disabled")
    if _bool(evaluation.get("use_action_ensembler"), field="use_action_ensembler"):
        raise RMBenchRuntimeProjectionError("action ensembling must be disabled")
    return _json_value(projection, field="RMBench runtime projection")


def rmbench_policy_runtime_sha256(
    resolved_config: Mapping[str, Any],
    runtime_args: Mapping[str, Any],
) -> str:
    """Return the canonical digest stored in the online run contract."""

    projection = build_rmbench_policy_runtime_projection(
        resolved_config, runtime_args
    )
    encoded = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "RMBENCH_ABLATION_MODES",
    "RMBENCH_MEMORY_CORRUPTIONS",
    "RMBENCH_ODE_STEPS",
    "RMBENCH_RUNTIME_PROJECTION_SCHEMA",
    "RMBENCH_RUNTIME_PROJECTION_VERSION",
    "RMBenchRuntimeProjectionError",
    "build_rmbench_policy_runtime_projection",
    "rmbench_policy_runtime_sha256",
]
