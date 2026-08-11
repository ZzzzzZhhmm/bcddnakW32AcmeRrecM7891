"""Closed-world provenance for one formally published WARM checkpoint.

The attestation is deliberately produced by the live trainer, not by a
post-hoc command.  It binds the checkpoint bytes to the resolved training
recipe and the distributed optimizer geometry that actually created them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from fastwam.memory.manifest import sha256_file


TRAINING_ATTESTATION_SCHEMA = "warm.training-attestation"
TRAINING_ATTESTATION_VERSION = 3

# These are the only resolved-config values intentionally removed from the
# fixed/null shared recipe.  W&B routing/enablement remains bound; only its
# human-facing run labels may differ.
V1_SHARED_RECIPE_IGNORED_PATHS = (
    "/output_dir",
    "/model/source_policy",
    "/wandb/workspace",
    "/wandb/project",
    "/wandb/name",
    "/wandb/group",
)

# V2 additionally excludes control-plane values that legitimately change at a
# restart while leaving the mathematical training recipe unchanged.  Their
# exact values remain bound by ``resolved_train_config_sha256``; excluding them
# only permits fixed/null fairness comparison and parent-recipe validation
# across independently named or differently checkpointed continuation jobs.
SHARED_RECIPE_IGNORED_PATHS = V1_SHARED_RECIPE_IGNORED_PATHS + (
    "/resume",
    "/run_steps",
    "/log_every",
    "/save_every",
    "/eval_every",
)

_SOURCE_POLICIES = frozenset(
    {"fixed_context_top1", "gaussian_null", "oracle_action_top1"}
)
_PRECISIONS = frozenset({"no", "fp16", "bf16"})
_SCHEDULERS = frozenset({"cosine", "constant"})
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_FIELDS_V1 = frozenset(
    {
        "schema",
        "version",
        "checkpoint_sha256",
        "checkpoint_step",
        "source_policy",
        "resolved_train_config_sha256",
        "shared_recipe_sha256",
        "shared_recipe_ignored_paths",
        "root_seed",
        "actual_global_step",
        "actual_max_steps",
        "optimizer_name",
        "optimizer_learning_rate",
        "optimizer_weight_decay",
        "optimizer_beta1",
        "optimizer_beta2",
        "optimizer_epsilon",
        "optimizer_amsgrad",
        "optimizer_wrapper_chain",
        "scheduler_type",
        "scheduler_total_steps",
        "scheduler_warmup_steps",
        "scheduler_min_learning_rate",
        "scheduler_wrapper_chain",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "world_size",
        "effective_batch_size",
        "mixed_precision",
        "training_runtime",
        "train_source_contract_sha256",
        "dev_source_contract_sha256",
        "base_checkpoint_sha256",
        "git_commit",
    }
)
_RESUME_LINEAGE_FIELDS = frozenset(
    {
        "parent_checkpoint_sha256",
        "parent_training_attestation_sha256",
        "resume_state_sha256",
        "resume_step",
    }
)
_FORK_LINEAGE_FIELDS = frozenset(
    {
        "parent_checkpoint_sha256",
        "parent_training_attestation_sha256",
        "fork_manifest_sha256",
        "fork_reason",
        "fork_parent_git_commit",
        "fork_parent_resolved_train_config_sha256",
    }
)
_FIELDS_V2 = _FIELDS_V1 | _RESUME_LINEAGE_FIELDS
_FIELDS_V3 = _FIELDS_V2 | frozenset(
    {
        "lineage_kind",
        "fork_manifest_sha256",
        "fork_reason",
        "fork_parent_git_commit",
        "fork_parent_resolved_train_config_sha256",
    }
)
_STEP_TAG = re.compile(r"step_(\d{6,})")

_TRAINING_RUNTIME_FIELDS = frozenset(
    {
        "python_version",
        "platform",
        "accelerate_version",
        "deepspeed_version",
        "torch_version",
        "torch_cuda_version",
        "cudnn_version",
        "cuda_available",
        "gpu_count",
        "gpu_devices",
        "distributed_type",
        "deepspeed_config",
        "deepspeed_zero_stage",
        "float32_matmul_precision",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "deterministic_algorithms_enabled",
        "cublas_workspace_config",
    }
)
_SUPPORTED_DISTRIBUTED_TYPES = frozenset({"no", "multi_gpu", "deepspeed"})


class TrainingAttestationError(ValueError):
    """Raised when formal WARM training provenance is incomplete or mutable."""


def _canonical_json_bytes(value: object, *, field: str) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        # Round-tripping rejects custom Mapping/list subclasses and guarantees
        # that the bytes consumed by every hash describe ordinary JSON only.
        json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TrainingAttestationError(
            f"{field} must contain only finite canonical JSON values"
        ) from error
    return encoded


def _sha256_json(value: object, *, field: str) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, field=field)).hexdigest()


def _digest(value: object, field: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingAttestationError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return value


def _integer(
    value: object,
    field: str,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result < minimum:
        raise TrainingAttestationError(f"{field} must be >= {minimum}")
    return result


def _finite_float(
    value: object,
    field: str,
    *,
    minimum: float,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingAttestationError(f"{field} must be finite")
    invalid = result <= minimum if strict_minimum else result < minimum
    if invalid:
        operator = ">" if strict_minimum else ">="
        raise TrainingAttestationError(f"{field} must be {operator} {minimum}")
    return result


def _plain_json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("resolved training config must be a mapping")
    encoded = _canonical_json_bytes(dict(value), field="resolved training config")
    result = json.loads(encoded)
    if not isinstance(result, dict):  # pragma: no cover - protected by Mapping
        raise TypeError("resolved training config must encode a JSON object")
    return result


def _normalized_string(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise TrainingAttestationError(
            f"{field} must be a normalized non-empty string"
        )
    return value


def _wrapper_chain(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise TrainingAttestationError(f"{field} must be a non-empty class list")
    result = tuple(_normalized_string(item, field) for item in value)
    return result


def _canonical_training_runtime(value: object) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("training_runtime must be a mapping")
    runtime = _plain_json_object(value)
    actual = set(runtime)
    if actual != _TRAINING_RUNTIME_FIELDS:
        raise TrainingAttestationError(
            "invalid training_runtime fields; "
            f"missing={sorted(_TRAINING_RUNTIME_FIELDS - actual)}, "
            f"extra={sorted(actual - _TRAINING_RUNTIME_FIELDS)}"
        )
    for field in (
        "python_version",
        "platform",
        "accelerate_version",
        "torch_version",
        "distributed_type",
        "float32_matmul_precision",
    ):
        runtime[field] = _normalized_string(runtime[field], f"training_runtime.{field}")
    for field in ("deepspeed_version", "torch_cuda_version"):
        item = runtime[field]
        if item is not None:
            runtime[field] = _normalized_string(
                item, f"training_runtime.{field}"
            )
    workspace = runtime["cublas_workspace_config"]
    if workspace is not None:
        runtime["cublas_workspace_config"] = _normalized_string(
            workspace, "training_runtime.cublas_workspace_config"
        )
    for field in (
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "deterministic_algorithms_enabled",
    ):
        if not isinstance(runtime[field], bool):
            raise TypeError(f"training_runtime.{field} must be a boolean")
    cudnn = runtime["cudnn_version"]
    if cudnn is not None:
        runtime["cudnn_version"] = _nonnegative_runtime_int(
            cudnn, "training_runtime.cudnn_version"
        )
    if not isinstance(runtime["cuda_available"], bool):
        raise TypeError("training_runtime.cuda_available must be a boolean")
    runtime["gpu_count"] = _nonnegative_runtime_int(
        runtime["gpu_count"], "training_runtime.gpu_count"
    )
    devices = runtime["gpu_devices"]
    if not isinstance(devices, list):
        raise TypeError("training_runtime.gpu_devices must be a list")
    normalized_devices: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        if not isinstance(device, Mapping) or set(device) != {
            "index",
            "name",
            "capability",
        }:
            raise TrainingAttestationError(
                f"training_runtime.gpu_devices[{index}] has invalid fields"
            )
        capability = device["capability"]
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in capability)
        ):
            raise TypeError(
                f"training_runtime.gpu_devices[{index}].capability must be [major, minor]"
            )
        normalized_devices.append(
            {
                "index": _nonnegative_runtime_int(
                    device["index"],
                    f"training_runtime.gpu_devices[{index}].index",
                ),
                "name": _normalized_string(
                    device["name"],
                    f"training_runtime.gpu_devices[{index}].name",
                ),
                "capability": [int(capability[0]), int(capability[1])],
            }
        )
    if runtime["gpu_count"] != len(normalized_devices):
        raise TrainingAttestationError(
            "training_runtime.gpu_count does not match gpu_devices"
        )
    if runtime["cuda_available"] != (runtime["gpu_count"] > 0):
        raise TrainingAttestationError(
            "training_runtime CUDA availability disagrees with gpu_count"
        )
    runtime["gpu_devices"] = normalized_devices

    distributed = runtime["distributed_type"].lower()
    if distributed not in _SUPPORTED_DISTRIBUTED_TYPES:
        raise TrainingAttestationError(
            f"unsupported accelerator distributed_type {distributed!r}"
        )
    runtime["distributed_type"] = distributed
    deepspeed_config = runtime["deepspeed_config"]
    zero_stage = runtime["deepspeed_zero_stage"]
    if distributed == "deepspeed":
        if runtime["deepspeed_version"] is None:
            raise TrainingAttestationError(
                "DeepSpeed distributed training requires a bound deepspeed version"
            )
        if not isinstance(deepspeed_config, Mapping):
            raise TrainingAttestationError(
                "DeepSpeed distributed training requires a canonical plugin config"
            )
        deepspeed_config = _plain_json_object(deepspeed_config)
        zero_stage = _nonnegative_runtime_int(
            zero_stage, "training_runtime.deepspeed_zero_stage"
        )
        if zero_stage > 3:
            raise TrainingAttestationError("DeepSpeed ZeRO stage must be in [0, 3]")
        zero = deepspeed_config.get("zero_optimization")
        if not isinstance(zero, Mapping) or zero.get("stage") != zero_stage:
            raise TrainingAttestationError(
                "DeepSpeed config zero_optimization.stage disagrees with the bound stage"
            )
        runtime["deepspeed_config"] = deepspeed_config
        runtime["deepspeed_zero_stage"] = zero_stage
    elif deepspeed_config is not None or zero_stage is not None:
        raise TrainingAttestationError(
            "non-DeepSpeed training must not claim a DeepSpeed plugin config or stage"
        )

    encoded = _canonical_json_bytes(runtime, field="training_runtime")
    return encoded.decode("utf-8"), json.loads(encoded)


def _nonnegative_runtime_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingAttestationError(f"{field} must be a non-negative integer")
    return int(value)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_training_runtime(accelerator: object) -> dict[str, Any]:
    """Capture the post-``prepare`` Accelerate/DeepSpeed numerical runtime."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - trainer always imports torch
        raise TrainingAttestationError("PyTorch is required for training attestation") from error

    raw_distributed = getattr(accelerator, "distributed_type", None)
    distributed = getattr(raw_distributed, "value", raw_distributed)
    distributed = str(distributed).strip().lower()
    if distributed.startswith("distributedtype."):
        distributed = distributed.split(".", 1)[1]
    state = getattr(accelerator, "state", None)
    plugin = getattr(state, "deepspeed_plugin", None)
    deepspeed_config: dict[str, Any] | None = None
    zero_stage: int | None = None
    if distributed == "deepspeed":
        raw_config = getattr(plugin, "deepspeed_config", None)
        if not isinstance(raw_config, Mapping):
            raise TrainingAttestationError(
                "Accelerate DeepSpeed mode has no canonical deepspeed_config"
            )
        raw_config = dict(raw_config)
        steps_per_print = raw_config.get("steps_per_print")
        if isinstance(steps_per_print, float) and math.isinf(steps_per_print):
            # Accelerate's DeepSpeedPlugin unconditionally injects float("inf")
            # here to silence DeepSpeed stdout logging. It is a logging cadence
            # rather than a numerical training fact and cannot be represented
            # in canonical JSON, so it is excluded from the attested config.
            del raw_config["steps_per_print"]
        deepspeed_config = _plain_json_object(raw_config)
        zero = deepspeed_config.get("zero_optimization")
        if not isinstance(zero, Mapping):
            raise TrainingAttestationError(
                "DeepSpeed config has no zero_optimization mapping"
            )
        zero_stage = _nonnegative_runtime_int(
            zero.get("stage"), "DeepSpeed zero_optimization.stage"
        )
    elif plugin is not None:
        raise TrainingAttestationError(
            "Accelerate exposes a DeepSpeed plugin outside DeepSpeed distributed mode"
        )

    cuda_available = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    devices = []
    for index in range(gpu_count):
        capability = torch.cuda.get_device_capability(index)
        devices.append(
            {
                "index": index,
                "name": str(torch.cuda.get_device_name(index)),
                "capability": [int(capability[0]), int(capability[1])],
            }
        )
    accelerate_version = _package_version("accelerate")
    if accelerate_version is None:
        raise TrainingAttestationError(
            "cannot resolve the installed Accelerate package version"
        )
    runtime = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "accelerate_version": accelerate_version,
        "deepspeed_version": _package_version("deepspeed"),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "cudnn_version": (
            None if torch.backends.cudnn.version() is None else int(torch.backends.cudnn.version())
        ),
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "gpu_devices": devices,
        "distributed_type": distributed,
        "deepspeed_config": deepspeed_config,
        "deepspeed_zero_stage": zero_stage,
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    return _canonical_training_runtime(runtime)[1]


def _qualified_class(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def capture_actual_optimizer_facts(optimizer: object) -> dict[str, Any]:
    """Resolve the optimizer returned by ``accelerator.prepare`` fail-closed."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - trainer always imports torch
        raise TrainingAttestationError("PyTorch is required for optimizer attestation") from error

    chain: list[str] = []
    current = optimizer
    seen: set[int] = set()
    while not isinstance(current, torch.optim.AdamW):
        if id(current) in seen:
            raise TrainingAttestationError("post-prepare optimizer wrapper cycle")
        seen.add(id(current))
        chain.append(_qualified_class(current))
        nested = getattr(current, "optimizer", None)
        if nested is None:
            raise TrainingAttestationError(
                "unsupported post-prepare optimizer wrapper " + _qualified_class(current)
            )
        current = nested
    chain.append(_qualified_class(current))
    defaults = dict(current.defaults)
    if bool(defaults.get("maximize", False)):
        raise TrainingAttestationError(
            "formal WARM attestation does not admit AdamW maximize=True"
        )
    betas = defaults.get("betas")
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise TrainingAttestationError("actual AdamW defaults lack two betas")
    required_group_values = {
        "weight_decay": defaults.get("weight_decay"),
        "betas": tuple(betas),
        "eps": defaults.get("eps"),
        "amsgrad": bool(defaults.get("amsgrad", False)),
    }
    if not current.param_groups:
        raise TrainingAttestationError("actual AdamW has no parameter groups")
    for index, group in enumerate(current.param_groups):
        for key, expected in required_group_values.items():
            actual = tuple(group.get(key)) if key == "betas" else group.get(key)
            if actual != expected:
                raise TrainingAttestationError(
                    f"AdamW parameter group {index} changes {key}"
                )
    return {
        "name": "AdamW",
        "learning_rate": defaults.get("lr"),
        "weight_decay": defaults.get("weight_decay"),
        "betas": [betas[0], betas[1]],
        "epsilon": defaults.get("eps"),
        "amsgrad": bool(defaults.get("amsgrad", False)),
        "wrapper_chain": chain,
    }


def capture_actual_scheduler_chain(
    scheduler: object,
    *,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int,
    minimum_learning_rate: float,
) -> tuple[str, ...]:
    """Validate the post-``prepare`` LR scheduler and return its wrapper chain."""

    try:
        from torch.optim.lr_scheduler import (
            ConstantLR,
            CosineAnnealingLR,
            LinearLR,
            SequentialLR,
        )
    except ImportError as error:  # pragma: no cover
        raise TrainingAttestationError("PyTorch scheduler classes are unavailable") from error

    canonical_type = str(scheduler_type).strip().lower()
    if canonical_type not in _SCHEDULERS:
        raise TrainingAttestationError(
            f"unsupported scheduler_type {canonical_type!r}"
        )
    chain: list[str] = []
    current = scheduler
    seen: set[int] = set()
    known = (SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR)
    while not isinstance(current, known):
        if id(current) in seen:
            raise TrainingAttestationError("post-prepare scheduler wrapper cycle")
        seen.add(id(current))
        chain.append(_qualified_class(current))
        nested = getattr(current, "scheduler", None)
        if nested is None:
            raise TrainingAttestationError(
                "unsupported post-prepare scheduler wrapper " + _qualified_class(current)
            )
        current = nested
    chain.append(_qualified_class(current))

    total = _integer(total_steps, "scheduler_total_steps", minimum=1)
    warmup = _integer(warmup_steps, "scheduler_warmup_steps", minimum=0)
    remaining = max(total - warmup, 1)
    main_expected = CosineAnnealingLR if canonical_type == "cosine" else ConstantLR
    main = current
    if warmup > 0:
        if not isinstance(current, SequentialLR):
            raise TrainingAttestationError(
                "post-prepare scheduler must resolve to SequentialLR with warmup"
            )
        children = list(getattr(current, "_schedulers", ()))
        if len(children) != 2 or not isinstance(children[0], LinearLR):
            raise TrainingAttestationError(
                "SequentialLR must contain LinearLR followed by the configured scheduler"
            )
        milestones = list(getattr(current, "_milestones", ()))
        if milestones != [warmup]:
            raise TrainingAttestationError(
                "SequentialLR warmup milestone differs from the configured recipe"
            )
        main = children[1]
    elif isinstance(current, SequentialLR):
        raise TrainingAttestationError(
            "post-prepare scheduler unexpectedly contains a warmup sequence"
        )
    if not isinstance(main, main_expected):
        raise TrainingAttestationError(
            "post-prepare scheduler implementation disagrees with scheduler_type"
        )
    if isinstance(main, CosineAnnealingLR):
        if int(main.T_max) != remaining or float(main.eta_min) != float(
            minimum_learning_rate
        ):
            raise TrainingAttestationError(
                "CosineAnnealingLR parameters differ from the configured recipe"
            )
    elif int(getattr(main, "total_iters", -1)) != remaining:
        raise TrainingAttestationError(
            "ConstantLR total_iters differs from the configured recipe"
        )
    return tuple(chain)


def training_config_hashes(
    resolved_config: Mapping[str, Any],
) -> tuple[str, str]:
    """Return full-config and policy-independent shared-recipe identities."""

    full = _plain_json_object(resolved_config)
    shared = json.loads(_canonical_json_bytes(full, field="resolved training config"))

    shared.pop("output_dir", None)
    for control_plane_field in (
        "resume",
        "run_steps",
        "log_every",
        "save_every",
        "eval_every",
    ):
        shared.pop(control_plane_field, None)
    model = shared.get("model")
    if isinstance(model, dict):
        model.pop("source_policy", None)
    wandb = shared.get("wandb")
    if isinstance(wandb, dict):
        wandb.pop("workspace", None)
        wandb.pop("project", None)
        wandb.pop("name", None)
        wandb.pop("group", None)

    return (
        _sha256_json(full, field="resolved training config"),
        _sha256_json(shared, field="shared training recipe"),
    )


def clean_git_commit(repository: str | Path) -> str:
    """Return HEAD only when the complete repository worktree is clean."""

    root = Path(repository).expanduser().resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise TrainingAttestationError(
            f"cannot inspect WARM Git provenance: repository={root}"
        ) from error
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise TrainingAttestationError(
            "Git returned an invalid commit SHA: "
            f"repository={root}, head={commit!r}"
        )
    if status.strip():
        entries = status.splitlines()
        preview = "; ".join(entries[:12])
        if len(entries) > 12:
            preview += f"; ... ({len(entries) - 12} more)"
        raise TrainingAttestationError(
            "formal WARM checkpoint publication requires a clean Git "
            f"worktree: repository={root}, head={commit}, changes={preview}"
        )
    return commit


@dataclass(frozen=True, slots=True)
class WarmTrainingAttestation:
    """Strict, canonical identity of one trained WARM weights checkpoint."""

    checkpoint_sha256: str
    checkpoint_step: int
    source_policy: str
    resolved_train_config_sha256: str
    shared_recipe_sha256: str
    root_seed: int
    actual_global_step: int
    actual_max_steps: int
    optimizer_name: str
    optimizer_learning_rate: float
    optimizer_weight_decay: float
    optimizer_beta1: float
    optimizer_beta2: float
    optimizer_epsilon: float
    optimizer_amsgrad: bool
    optimizer_wrapper_chain: tuple[str, ...]
    scheduler_type: str
    scheduler_total_steps: int
    scheduler_warmup_steps: int
    scheduler_min_learning_rate: float
    scheduler_wrapper_chain: tuple[str, ...]
    per_device_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    effective_batch_size: int
    mixed_precision: str
    training_runtime_json: str
    train_source_contract_sha256: str
    dev_source_contract_sha256: str
    base_checkpoint_sha256: str
    git_commit: str
    parent_checkpoint_sha256: str | None = None
    parent_training_attestation_sha256: str | None = None
    resume_state_sha256: str | None = None
    resume_step: int | None = None
    lineage_kind: str = "none"
    fork_manifest_sha256: str | None = None
    fork_reason: str | None = None
    fork_parent_git_commit: str | None = None
    fork_parent_resolved_train_config_sha256: str | None = None
    shared_recipe_ignored_paths: tuple[str, ...] = SHARED_RECIPE_IGNORED_PATHS
    schema: str = TRAINING_ATTESTATION_SCHEMA
    version: int = TRAINING_ATTESTATION_VERSION

    def __post_init__(self) -> None:
        if self.schema != TRAINING_ATTESTATION_SCHEMA:
            raise TrainingAttestationError(
                f"unsupported training-attestation schema {self.schema!r}"
            )
        version = _integer(self.version, "version", minimum=1)
        if version not in (1, 2, TRAINING_ATTESTATION_VERSION):
            raise TrainingAttestationError(
                f"unsupported training-attestation version {version}"
            )
        for field in (
            "checkpoint_sha256",
            "resolved_train_config_sha256",
            "shared_recipe_sha256",
            "train_source_contract_sha256",
            "base_checkpoint_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(
            self,
            "dev_source_contract_sha256",
            _digest(self.dev_source_contract_sha256, "dev_source_contract_sha256"),
        )
        for field in (
            "parent_checkpoint_sha256",
            "parent_training_attestation_sha256",
            "resume_state_sha256",
            "fork_manifest_sha256",
            "fork_parent_resolved_train_config_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _digest(getattr(self, field), field, optional=True),
            )
        if self.source_policy not in _SOURCE_POLICIES:
            raise TrainingAttestationError(
                f"unsupported source_policy {self.source_policy!r}"
            )
        if self.optimizer_name != "AdamW":
            raise TrainingAttestationError("optimizer_name must be 'AdamW'")
        if self.scheduler_type not in _SCHEDULERS:
            raise TrainingAttestationError(
                f"unsupported scheduler_type {self.scheduler_type!r}"
            )
        if self.mixed_precision not in _PRECISIONS:
            raise TrainingAttestationError(
                f"unsupported mixed_precision {self.mixed_precision!r}"
            )
        if not isinstance(self.optimizer_amsgrad, bool):
            raise TypeError("optimizer_amsgrad must be a boolean")
        object.__setattr__(
            self,
            "optimizer_wrapper_chain",
            _wrapper_chain(self.optimizer_wrapper_chain, "optimizer_wrapper_chain"),
        )
        object.__setattr__(
            self,
            "scheduler_wrapper_chain",
            _wrapper_chain(self.scheduler_wrapper_chain, "scheduler_wrapper_chain"),
        )
        try:
            runtime_value = json.loads(self.training_runtime_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise TrainingAttestationError(
                "training_runtime_json must contain canonical JSON"
            ) from error
        runtime_json, _ = _canonical_training_runtime(runtime_value)
        if runtime_json != self.training_runtime_json:
            raise TrainingAttestationError("training_runtime_json is not canonical")
        expected_ignored_paths = (
            V1_SHARED_RECIPE_IGNORED_PATHS
            if version == 1
            else SHARED_RECIPE_IGNORED_PATHS
        )
        if tuple(self.shared_recipe_ignored_paths) != expected_ignored_paths:
            raise TrainingAttestationError(
                "shared_recipe_ignored_paths does not match the attestation version"
            )
        if _GIT_COMMIT.fullmatch(self.git_commit) is None:
            raise TrainingAttestationError(
                "git_commit must be a lowercase 40-character commit SHA"
            )
        if self.fork_parent_git_commit is not None and _GIT_COMMIT.fullmatch(
            self.fork_parent_git_commit
        ) is None:
            raise TrainingAttestationError(
                "fork_parent_git_commit must be a lowercase 40-character commit SHA"
            )

        for field, minimum in (
            ("checkpoint_step", 0),
            ("root_seed", 0),
            ("actual_global_step", 0),
            ("actual_max_steps", 1),
            ("scheduler_total_steps", 1),
            ("scheduler_warmup_steps", 0),
            ("per_device_batch_size", 1),
            ("gradient_accumulation_steps", 1),
            ("world_size", 1),
            ("effective_batch_size", 1),
        ):
            object.__setattr__(
                self,
                field,
                _integer(getattr(self, field), field, minimum=minimum),
            )

        for field, minimum, strict in (
            ("optimizer_learning_rate", 0.0, True),
            ("optimizer_weight_decay", 0.0, False),
            ("optimizer_beta1", 0.0, False),
            ("optimizer_beta2", 0.0, False),
            ("optimizer_epsilon", 0.0, True),
            ("scheduler_min_learning_rate", 0.0, False),
        ):
            object.__setattr__(
                self,
                field,
                _finite_float(
                    getattr(self, field),
                    field,
                    minimum=minimum,
                    strict_minimum=strict,
                ),
            )
        if self.optimizer_beta1 >= 1.0 or self.optimizer_beta2 >= 1.0:
            raise TrainingAttestationError("AdamW beta values must be < 1")
        if self.checkpoint_step != self.actual_global_step:
            raise TrainingAttestationError(
                "checkpoint_step must equal actual_global_step"
            )
        if self.actual_global_step > self.actual_max_steps:
            raise TrainingAttestationError(
                "actual_global_step cannot exceed actual_max_steps"
            )
        if self.scheduler_total_steps != self.actual_max_steps:
            raise TrainingAttestationError(
                "scheduler_total_steps must equal actual_max_steps"
            )
        if self.scheduler_warmup_steps >= self.scheduler_total_steps:
            raise TrainingAttestationError(
                "scheduler_warmup_steps must be less than scheduler_total_steps"
            )
        expected_batch = (
            self.per_device_batch_size
            * self.gradient_accumulation_steps
            * self.world_size
        )
        if self.effective_batch_size != expected_batch:
            raise TrainingAttestationError(
                "effective_batch_size does not match per-device batch, "
                "gradient accumulation, and world size"
            )
        resume_lineage = (
            self.parent_checkpoint_sha256,
            self.parent_training_attestation_sha256,
            self.resume_state_sha256,
            self.resume_step,
        )
        fork_lineage = (
            self.parent_checkpoint_sha256,
            self.parent_training_attestation_sha256,
            self.fork_manifest_sha256,
            self.fork_reason,
            self.fork_parent_git_commit,
            self.fork_parent_resolved_train_config_sha256,
        )
        if version == 1:
            if any(item is not None for item in resume_lineage + fork_lineage[2:]):
                raise TrainingAttestationError(
                    "v1 training attestations cannot contain lineage"
                )
        elif version == 2:
            if self.lineage_kind != "none" or any(
                item is not None for item in fork_lineage[2:]
            ):
                raise TrainingAttestationError(
                    "v2 training attestations cannot contain fork lineage"
                )
            if any(item is not None for item in resume_lineage) and not all(
                item is not None for item in resume_lineage
            ):
                raise TrainingAttestationError(
                    "resume lineage fields must be either all null or all populated"
                )
        else:
            if self.lineage_kind not in {"none", "resume", "fork"}:
                raise TrainingAttestationError(
                    "lineage_kind must be one of ['none', 'resume', 'fork']"
                )
            if self.lineage_kind == "none":
                if any(item is not None for item in resume_lineage + fork_lineage[2:]):
                    raise TrainingAttestationError(
                        "fresh training cannot contain parent lineage"
                    )
            elif self.lineage_kind == "resume":
                if not all(item is not None for item in resume_lineage):
                    raise TrainingAttestationError(
                        "resume lineage fields must all be populated"
                    )
                if any(item is not None for item in fork_lineage[2:]):
                    raise TrainingAttestationError(
                        "resume lineage cannot contain fork fields"
                    )
            else:
                if not all(item is not None for item in fork_lineage):
                    raise TrainingAttestationError(
                        "fork lineage fields must all be populated"
                    )
                if self.resume_state_sha256 is not None or self.resume_step is not None:
                    raise TrainingAttestationError(
                        "fork lineage cannot contain resume state or step"
                    )
                if not isinstance(self.fork_reason, str) or not self.fork_reason.strip():
                    raise TrainingAttestationError("fork_reason must be non-empty")
                if len(self.fork_reason) > 512:
                    raise TrainingAttestationError("fork_reason is too long")
        if self.resume_step is not None:
            object.__setattr__(
                self,
                "resume_step",
                _integer(self.resume_step, "resume_step", minimum=0),
            )
            if self.resume_step > self.checkpoint_step:
                raise TrainingAttestationError(
                    "resume_step cannot exceed checkpoint_step"
                )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "version": self.version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_step": self.checkpoint_step,
            "source_policy": self.source_policy,
            "resolved_train_config_sha256": self.resolved_train_config_sha256,
            "shared_recipe_sha256": self.shared_recipe_sha256,
            "shared_recipe_ignored_paths": list(
                self.shared_recipe_ignored_paths
            ),
            "root_seed": self.root_seed,
            "actual_global_step": self.actual_global_step,
            "actual_max_steps": self.actual_max_steps,
            "optimizer_name": self.optimizer_name,
            "optimizer_learning_rate": self.optimizer_learning_rate,
            "optimizer_weight_decay": self.optimizer_weight_decay,
            "optimizer_beta1": self.optimizer_beta1,
            "optimizer_beta2": self.optimizer_beta2,
            "optimizer_epsilon": self.optimizer_epsilon,
            "optimizer_amsgrad": self.optimizer_amsgrad,
            "optimizer_wrapper_chain": list(self.optimizer_wrapper_chain),
            "scheduler_type": self.scheduler_type,
            "scheduler_total_steps": self.scheduler_total_steps,
            "scheduler_warmup_steps": self.scheduler_warmup_steps,
            "scheduler_min_learning_rate": self.scheduler_min_learning_rate,
            "scheduler_wrapper_chain": list(self.scheduler_wrapper_chain),
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "world_size": self.world_size,
            "effective_batch_size": self.effective_batch_size,
            "mixed_precision": self.mixed_precision,
            "training_runtime": self.training_runtime,
            "train_source_contract_sha256": (
                self.train_source_contract_sha256
            ),
            "dev_source_contract_sha256": self.dev_source_contract_sha256,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "git_commit": self.git_commit,
        }
        if self.version >= 2:
            value.update(
                {
                    "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
                    "parent_training_attestation_sha256": (
                        self.parent_training_attestation_sha256
                    ),
                    "resume_state_sha256": self.resume_state_sha256,
                    "resume_step": self.resume_step,
                }
            )
        if self.version >= 3:
            value.update(
                {
                    "lineage_kind": self.lineage_kind,
                    "fork_manifest_sha256": self.fork_manifest_sha256,
                    "fork_reason": self.fork_reason,
                    "fork_parent_git_commit": self.fork_parent_git_commit,
                    "fork_parent_resolved_train_config_sha256": (
                        self.fork_parent_resolved_train_config_sha256
                    ),
                }
            )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WarmTrainingAttestation":
        if not isinstance(value, Mapping):
            raise TypeError("training attestation must be a mapping")
        raw_version = value.get("version")
        version = _integer(raw_version, "version", minimum=1)
        expected_fields = (
            _FIELDS_V1
            if version == 1
            else (_FIELDS_V2 if version == 2 else _FIELDS_V3)
        )
        if version not in (1, 2, TRAINING_ATTESTATION_VERSION):
            raise TrainingAttestationError(
                f"unsupported training-attestation version {version}"
            )
        actual = set(value)
        if actual != expected_fields:
            raise TrainingAttestationError(
                "invalid training-attestation fields; "
                f"missing={sorted(expected_fields - actual)}, "
                f"extra={sorted(actual - expected_fields)}"
            )
        payload = dict(value)
        if version == 1:
            payload.update(
                {
                    "parent_checkpoint_sha256": None,
                    "parent_training_attestation_sha256": None,
                    "resume_state_sha256": None,
                    "resume_step": None,
                }
            )
        if version < 3:
            payload.update(
                {
                    "lineage_kind": "none",
                    "fork_manifest_sha256": None,
                    "fork_reason": None,
                    "fork_parent_git_commit": None,
                    "fork_parent_resolved_train_config_sha256": None,
                }
            )
        paths = payload["shared_recipe_ignored_paths"]
        if not isinstance(paths, list) or any(
            not isinstance(item, str) for item in paths
        ):
            raise TypeError("shared_recipe_ignored_paths must be a string list")
        payload["shared_recipe_ignored_paths"] = tuple(paths)
        payload["optimizer_wrapper_chain"] = _wrapper_chain(
            payload["optimizer_wrapper_chain"], "optimizer_wrapper_chain"
        )
        payload["scheduler_wrapper_chain"] = _wrapper_chain(
            payload["scheduler_wrapper_chain"], "scheduler_wrapper_chain"
        )
        runtime_json, _ = _canonical_training_runtime(payload.pop("training_runtime"))
        payload["training_runtime_json"] = runtime_json
        return cls(**payload)

    @property
    def training_runtime(self) -> dict[str, Any]:
        return json.loads(self.training_runtime_json)

    @property
    def training_runtime_sha256(self) -> str:
        return hashlib.sha256(self.training_runtime_json.encode("utf-8")).hexdigest()

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict(), field="training attestation")

    def encode(self) -> bytes:
        return _canonical_json_bytes(self.to_dict(), field="training attestation") + b"\n"


@dataclass(frozen=True, slots=True)
class WarmTrainingRunContext:
    """Immutable trainer facts used to build per-checkpoint attestations."""

    source_policy: str
    resolved_train_config_sha256: str
    shared_recipe_sha256: str
    root_seed: int
    actual_max_steps: int
    optimizer_learning_rate: float
    optimizer_weight_decay: float
    optimizer_beta1: float
    optimizer_beta2: float
    optimizer_epsilon: float
    optimizer_amsgrad: bool
    optimizer_wrapper_chain: tuple[str, ...]
    scheduler_type: str
    scheduler_total_steps: int
    scheduler_warmup_steps: int
    scheduler_min_learning_rate: float
    scheduler_wrapper_chain: tuple[str, ...]
    per_device_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    effective_batch_size: int
    mixed_precision: str
    training_runtime_json: str
    train_source_contract_sha256: str
    dev_source_contract_sha256: str
    base_checkpoint_sha256: str
    git_commit: str
    repository_root: Path
    parent_checkpoint_sha256: str | None = None
    parent_training_attestation_sha256: str | None = None
    resume_state_sha256: str | None = None
    resume_step: int | None = None
    lineage_kind: str = "none"
    fork_manifest_sha256: str | None = None
    fork_reason: str | None = None
    fork_parent_git_commit: str | None = None
    fork_parent_resolved_train_config_sha256: str | None = None

    @classmethod
    def create(
        cls,
        *,
        resolved_config: Mapping[str, Any],
        source_metadata: Mapping[str, Any],
        root_seed: int,
        actual_max_steps: int,
        optimizer_facts: Mapping[str, Any],
        scheduler_type: str,
        scheduler_warmup_steps: int,
        scheduler_min_learning_rate: float,
        per_device_batch_size: int,
        gradient_accumulation_steps: int,
        world_size: int,
        mixed_precision: str,
        scheduler_wrapper_chain: tuple[str, ...] | list[str],
        training_runtime: Mapping[str, Any],
        repository_root: str | Path,
    ) -> "WarmTrainingRunContext":
        required_source = {
            "source_policy",
            "train_source_contract_sha256",
            "dev_source_contract_sha256",
            "base_checkpoint_sha256",
        }
        if not isinstance(source_metadata, Mapping) or set(source_metadata) != required_source:
            actual = set(source_metadata) if isinstance(source_metadata, Mapping) else set()
            raise TrainingAttestationError(
                "invalid WARM source attestation metadata; "
                f"missing={sorted(required_source - actual)}, "
                f"extra={sorted(actual - required_source)}"
            )
        config_sha, shared_sha = training_config_hashes(resolved_config)
        root = Path(repository_root).expanduser().resolve()
        commit = clean_git_commit(root)
        required_optimizer = {
            "name",
            "learning_rate",
            "weight_decay",
            "betas",
            "epsilon",
            "amsgrad",
            "wrapper_chain",
        }
        if not isinstance(optimizer_facts, Mapping) or set(optimizer_facts) != required_optimizer:
            actual_optimizer = (
                set(optimizer_facts)
                if isinstance(optimizer_facts, Mapping)
                else set()
            )
            raise TrainingAttestationError(
                "invalid post-prepare optimizer facts; "
                f"missing={sorted(required_optimizer - actual_optimizer)}, "
                f"extra={sorted(actual_optimizer - required_optimizer)}"
            )
        if optimizer_facts["name"] != "AdamW":
            raise TrainingAttestationError("post-prepare optimizer must resolve to AdamW")
        betas = optimizer_facts.get("betas")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise TrainingAttestationError("actual AdamW defaults lack two betas")
        runtime_json, runtime = _canonical_training_runtime(training_runtime)
        if runtime["distributed_type"] != "deepspeed" and runtime["deepspeed_config"] is not None:
            raise TrainingAttestationError("invalid non-DeepSpeed runtime config")
        processes = _integer(world_size, "world_size", minimum=1)
        batch = _integer(
            per_device_batch_size, "per_device_batch_size", minimum=1
        )
        accumulation = _integer(
            gradient_accumulation_steps,
            "gradient_accumulation_steps",
            minimum=1,
        )
        policy = str(source_metadata["source_policy"])
        if policy not in _SOURCE_POLICIES:
            raise TrainingAttestationError(
                f"unsupported source_policy {policy!r}"
            )
        train_contract = _digest(
            source_metadata["train_source_contract_sha256"],
            "train_source_contract_sha256",
        )
        dev_contract = _digest(
            source_metadata["dev_source_contract_sha256"],
            "dev_source_contract_sha256",
        )
        base_checkpoint = _digest(
            source_metadata["base_checkpoint_sha256"],
            "base_checkpoint_sha256",
        )
        canonical_scheduler = str(scheduler_type).strip().lower()
        if canonical_scheduler not in _SCHEDULERS:
            raise TrainingAttestationError(
                f"unsupported scheduler_type {canonical_scheduler!r}"
            )
        canonical_precision = str(mixed_precision).strip().lower()
        if canonical_precision not in _PRECISIONS:
            raise TrainingAttestationError(
                f"unsupported mixed_precision {canonical_precision!r}"
            )
        context = cls(
            source_policy=policy,
            resolved_train_config_sha256=config_sha,
            shared_recipe_sha256=shared_sha,
            root_seed=_integer(root_seed, "root_seed", minimum=0),
            actual_max_steps=_integer(
                actual_max_steps, "actual_max_steps", minimum=1
            ),
            optimizer_learning_rate=_finite_float(
                optimizer_facts.get("learning_rate"),
                "optimizer_learning_rate",
                minimum=0.0,
                strict_minimum=True,
            ),
            optimizer_weight_decay=_finite_float(
                optimizer_facts.get("weight_decay"),
                "optimizer_weight_decay",
                minimum=0.0,
            ),
            optimizer_beta1=_finite_float(
                betas[0], "optimizer_beta1", minimum=0.0
            ),
            optimizer_beta2=_finite_float(
                betas[1], "optimizer_beta2", minimum=0.0
            ),
            optimizer_epsilon=_finite_float(
                optimizer_facts.get("epsilon"),
                "optimizer_epsilon",
                minimum=0.0,
                strict_minimum=True,
            ),
            optimizer_amsgrad=bool(optimizer_facts.get("amsgrad", False)),
            optimizer_wrapper_chain=_wrapper_chain(
                optimizer_facts.get("wrapper_chain"), "optimizer_wrapper_chain"
            ),
            scheduler_type=canonical_scheduler,
            scheduler_total_steps=_integer(
                actual_max_steps, "scheduler_total_steps", minimum=1
            ),
            scheduler_warmup_steps=_integer(
                scheduler_warmup_steps,
                "scheduler_warmup_steps",
                minimum=0,
            ),
            scheduler_min_learning_rate=_finite_float(
                scheduler_min_learning_rate,
                "scheduler_min_learning_rate",
                minimum=0.0,
            ),
            scheduler_wrapper_chain=_wrapper_chain(
                scheduler_wrapper_chain, "scheduler_wrapper_chain"
            ),
            per_device_batch_size=batch,
            gradient_accumulation_steps=accumulation,
            world_size=processes,
            effective_batch_size=batch * accumulation * processes,
            mixed_precision=canonical_precision,
            training_runtime_json=runtime_json,
            train_source_contract_sha256=str(train_contract),
            dev_source_contract_sha256=str(dev_contract),
            base_checkpoint_sha256=str(base_checkpoint),
            git_commit=commit,
            repository_root=root,
        )
        # Exercise the same strict final schema before the first optimizer
        # update, so an incomplete recipe cannot fail only at checkpoint time.
        context.build(checkpoint_sha256="0" * 64, actual_global_step=0)
        return context

    def with_resume_lineage(
        self, lineage: Mapping[str, Any]
    ) -> "WarmTrainingRunContext":
        """Bind a fully verified parent state to all subsequently saved weights."""

        if not isinstance(lineage, Mapping) or set(lineage) != _RESUME_LINEAGE_FIELDS:
            actual = set(lineage) if isinstance(lineage, Mapping) else set()
            raise TrainingAttestationError(
                "invalid formal resume lineage; "
                f"missing={sorted(_RESUME_LINEAGE_FIELDS - actual)}, "
                f"extra={sorted(actual - _RESUME_LINEAGE_FIELDS)}"
            )
        step = _integer(lineage["resume_step"], "resume_step", minimum=0)
        result = replace(
            self,
            parent_checkpoint_sha256=str(
                _digest(
                    lineage["parent_checkpoint_sha256"],
                    "parent_checkpoint_sha256",
                )
            ),
            parent_training_attestation_sha256=str(
                _digest(
                    lineage["parent_training_attestation_sha256"],
                    "parent_training_attestation_sha256",
                )
            ),
            resume_state_sha256=str(
                _digest(lineage["resume_state_sha256"], "resume_state_sha256")
            ),
            resume_step=step,
            lineage_kind="resume",
        )
        result.build(checkpoint_sha256="0" * 64, actual_global_step=step)
        return result

    def with_fork_lineage(
        self, lineage: Mapping[str, Any]
    ) -> "WarmTrainingRunContext":
        """Bind a verified weights-only initialization to child checkpoints."""

        if not isinstance(lineage, Mapping) or set(lineage) != _FORK_LINEAGE_FIELDS:
            actual = set(lineage) if isinstance(lineage, Mapping) else set()
            raise TrainingAttestationError(
                "invalid formal fork lineage; "
                f"missing={sorted(_FORK_LINEAGE_FIELDS - actual)}, "
                f"extra={sorted(actual - _FORK_LINEAGE_FIELDS)}"
            )
        reason = lineage["fork_reason"]
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 512:
            raise TrainingAttestationError(
                "fork_reason must be a non-empty string of at most 512 characters"
            )
        parent_commit = str(lineage["fork_parent_git_commit"])
        if _GIT_COMMIT.fullmatch(parent_commit) is None:
            raise TrainingAttestationError("invalid fork parent Git commit")
        result = replace(
            self,
            parent_checkpoint_sha256=str(
                _digest(
                    lineage["parent_checkpoint_sha256"],
                    "parent_checkpoint_sha256",
                )
            ),
            parent_training_attestation_sha256=str(
                _digest(
                    lineage["parent_training_attestation_sha256"],
                    "parent_training_attestation_sha256",
                )
            ),
            lineage_kind="fork",
            fork_manifest_sha256=str(
                _digest(lineage["fork_manifest_sha256"], "fork_manifest_sha256")
            ),
            fork_reason=reason.strip(),
            fork_parent_git_commit=parent_commit,
            fork_parent_resolved_train_config_sha256=str(
                _digest(
                    lineage["fork_parent_resolved_train_config_sha256"],
                    "fork_parent_resolved_train_config_sha256",
                )
            ),
        )
        result.build(checkpoint_sha256="0" * 64, actual_global_step=0)
        return result

    def build(
        self, *, checkpoint_sha256: str, actual_global_step: int
    ) -> WarmTrainingAttestation:
        step = _integer(actual_global_step, "actual_global_step", minimum=0)
        return WarmTrainingAttestation(
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_step=step,
            source_policy=self.source_policy,
            resolved_train_config_sha256=self.resolved_train_config_sha256,
            shared_recipe_sha256=self.shared_recipe_sha256,
            root_seed=self.root_seed,
            actual_global_step=step,
            actual_max_steps=self.actual_max_steps,
            optimizer_name="AdamW",
            optimizer_learning_rate=self.optimizer_learning_rate,
            optimizer_weight_decay=self.optimizer_weight_decay,
            optimizer_beta1=self.optimizer_beta1,
            optimizer_beta2=self.optimizer_beta2,
            optimizer_epsilon=self.optimizer_epsilon,
            optimizer_amsgrad=self.optimizer_amsgrad,
            optimizer_wrapper_chain=self.optimizer_wrapper_chain,
            scheduler_type=self.scheduler_type,
            scheduler_total_steps=self.scheduler_total_steps,
            scheduler_warmup_steps=self.scheduler_warmup_steps,
            scheduler_min_learning_rate=self.scheduler_min_learning_rate,
            scheduler_wrapper_chain=self.scheduler_wrapper_chain,
            per_device_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            world_size=self.world_size,
            effective_batch_size=self.effective_batch_size,
            mixed_precision=self.mixed_precision,
            training_runtime_json=self.training_runtime_json,
            train_source_contract_sha256=self.train_source_contract_sha256,
            dev_source_contract_sha256=self.dev_source_contract_sha256,
            base_checkpoint_sha256=self.base_checkpoint_sha256,
            git_commit=self.git_commit,
            parent_checkpoint_sha256=self.parent_checkpoint_sha256,
            parent_training_attestation_sha256=(
                self.parent_training_attestation_sha256
            ),
            resume_state_sha256=self.resume_state_sha256,
            resume_step=self.resume_step,
            lineage_kind=self.lineage_kind,
            fork_manifest_sha256=self.fork_manifest_sha256,
            fork_reason=self.fork_reason,
            fork_parent_git_commit=self.fork_parent_git_commit,
            fork_parent_resolved_train_config_sha256=(
                self.fork_parent_resolved_train_config_sha256
            ),
        )


def sha256_training_state_tree(state_directory: str | Path) -> str:
    """Hash an immutable view of every regular file in a resume state tree.

    Relative paths and file digests are length-delimited, making the identity
    independent of the server mount point and unambiguous across filenames.
    Symlinks and special files are rejected so a resumed state cannot change
    meaning between validation and Accelerate's read.
    """

    root = Path(state_directory).expanduser().resolve()
    if not root.is_dir():
        raise TrainingAttestationError(
            f"formal WARM resume state is not a directory: {root}"
        )
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise TrainingAttestationError(
                f"formal WARM resume state contains a symlink: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise TrainingAttestationError(
                f"formal WARM resume state contains a special file: {path}"
            )
        files.append(path)
    if not files:
        raise TrainingAttestationError("formal WARM resume state is empty")

    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        before = path.stat()
        file_digest = bytes.fromhex(sha256_file(path))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise TrainingAttestationError(
                f"resume-state file changed while it was hashed: {path}"
            )
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(before.st_size.to_bytes(16, "big"))
        digest.update(file_digest)
    return digest.hexdigest()


def prepare_formal_resume_lineage(
    state_directory: str | Path,
    *,
    current_context: WarmTrainingRunContext,
) -> dict[str, Any]:
    """Verify a full-state parent checkpoint and return its V2 lineage.

    V1 parent weights are accepted as a one-way migration path because those
    checkpoints predate lineage fields.  Every subsequent V2 checkpoint binds
    both the verified parent weights/attestation and the exact DeepSpeed state
    tree that was loaded.
    """

    state = Path(state_directory).expanduser().resolve()
    if not state.is_dir():
        raise TrainingAttestationError(
            "formal WARM continuation requires an Accelerate/DeepSpeed state directory"
        )
    match = _STEP_TAG.fullmatch(state.name)
    if match is None:
        raise TrainingAttestationError(
            "formal WARM resume directory must be named step_<global_step>"
        )
    resume_step = _integer(int(match.group(1)), "resume_step", minimum=0)
    state_file = state / "trainer_state.json"
    if not state_file.is_file():
        raise TrainingAttestationError(
            f"formal WARM resume is missing {state_file}"
        )
    try:
        trainer_state = json.loads(state_file.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingAttestationError(
            "formal WARM trainer_state.json is not valid UTF-8 JSON"
        ) from error
    if not isinstance(trainer_state, dict):
        raise TrainingAttestationError("trainer_state.json must contain an object")
    state_step = _integer(
        trainer_state.get("global_step"), "trainer_state.global_step", minimum=0
    )
    if state_step != resume_step:
        raise TrainingAttestationError(
            "resume directory step disagrees with trainer_state.global_step"
        )

    if state.parent.name != "state" or state.parent.parent.name != "checkpoints":
        raise TrainingAttestationError(
            "formal WARM resume must use checkpoints/state/step_<global_step>"
        )
    weights = state.parent.parent / "weights" / f"step_{resume_step:06d}.pt"
    if not weights.is_file():
        raise TrainingAttestationError(
            f"formal WARM resume has no matching weights checkpoint: {weights}"
        )
    sidecar = training_attestation_path(weights)
    parent = verify_training_attestation(weights, sidecar)
    if parent.checkpoint_step != resume_step:
        raise TrainingAttestationError(
            "parent weight attestation step disagrees with resume state"
        )

    parent_facts = {
        "source_policy": parent.source_policy,
        "root_seed": parent.root_seed,
        "actual_max_steps": parent.actual_max_steps,
        "optimizer_learning_rate": parent.optimizer_learning_rate,
        "optimizer_weight_decay": parent.optimizer_weight_decay,
        "optimizer_beta1": parent.optimizer_beta1,
        "optimizer_beta2": parent.optimizer_beta2,
        "optimizer_epsilon": parent.optimizer_epsilon,
        "optimizer_amsgrad": parent.optimizer_amsgrad,
        "optimizer_wrapper_chain": parent.optimizer_wrapper_chain,
        "scheduler_type": parent.scheduler_type,
        "scheduler_total_steps": parent.scheduler_total_steps,
        "scheduler_warmup_steps": parent.scheduler_warmup_steps,
        "scheduler_min_learning_rate": parent.scheduler_min_learning_rate,
        "scheduler_wrapper_chain": parent.scheduler_wrapper_chain,
        "per_device_batch_size": parent.per_device_batch_size,
        "gradient_accumulation_steps": parent.gradient_accumulation_steps,
        "world_size": parent.world_size,
        "effective_batch_size": parent.effective_batch_size,
        "mixed_precision": parent.mixed_precision,
        "training_runtime_json": parent.training_runtime_json,
        "train_source_contract_sha256": parent.train_source_contract_sha256,
        "dev_source_contract_sha256": parent.dev_source_contract_sha256,
        "base_checkpoint_sha256": parent.base_checkpoint_sha256,
    }
    current_facts = {key: getattr(current_context, key) for key in parent_facts}
    differing = sorted(
        key for key in parent_facts if parent_facts[key] != current_facts[key]
    )
    if differing:
        raise TrainingAttestationError(
            "resume parent contradicts the live training recipe/runtime: "
            + ", ".join(differing)
        )
    if (
        parent.version >= 2
        and parent.shared_recipe_sha256 != current_context.shared_recipe_sha256
    ):
        raise TrainingAttestationError(
            "resume parent shared training recipe differs from the continuation"
        )

    return {
        "parent_checkpoint_sha256": parent.checkpoint_sha256,
        "parent_training_attestation_sha256": sha256_file(sidecar),
        "resume_state_sha256": sha256_training_state_tree(state),
        "resume_step": resume_step,
    }


def training_attestation_path(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.with_suffix(".training.json")


def _write_exclusive_atomic(path: Path, payload: bytes) -> None:
    """Publish complete bytes atomically without replacing any prior claim."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"formal WARM artifact already exists: {path}")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link is an atomic O_EXCL-style publication on
        # both NTFS and Linux filesystems: it cannot replace an existing step.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_training_attestation(
    checkpoint_path: str | Path,
    *,
    context: WarmTrainingRunContext,
    actual_global_step: int,
) -> tuple[Path, WarmTrainingAttestation]:
    """Atomically publish an attestation with checkpoint/Git TOCTOU checks."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    before = sha256_file(checkpoint)
    if clean_git_commit(context.repository_root) != context.git_commit:
        raise TrainingAttestationError(
            "Git commit changed after formal WARM training started"
        )
    attestation = context.build(
        checkpoint_sha256=before,
        actual_global_step=actual_global_step,
    )

    # Validate canonical bytes before they become visible.
    payload = attestation.encode()
    decoded = json.loads(payload)
    if WarmTrainingAttestation.from_dict(decoded) != attestation:
        raise TrainingAttestationError(
            "training attestation did not round-trip canonically"
        )
    if sha256_file(checkpoint) != before:
        raise TrainingAttestationError(
            "WARM checkpoint changed while its attestation was prepared"
        )
    if clean_git_commit(context.repository_root) != context.git_commit:
        raise TrainingAttestationError(
            "Git state changed while the attestation was prepared"
        )

    output = training_attestation_path(checkpoint)
    _write_exclusive_atomic(output, payload)
    try:
        if sha256_file(checkpoint) != before:
            raise TrainingAttestationError(
                "WARM checkpoint changed while its attestation was published"
            )
        if clean_git_commit(context.repository_root) != context.git_commit:
            raise TrainingAttestationError(
                "Git state changed while the attestation was published"
            )
        verified = verify_training_attestation(checkpoint, output)
        if verified != attestation:
            raise TrainingAttestationError(
                "published training attestation changed during verification"
            )
        expected_file_sha256 = hashlib.sha256(payload).hexdigest()
        if sha256_file(output) != expected_file_sha256:
            raise TrainingAttestationError(
                "training attestation bytes changed after publication"
            )
    except Exception:
        # Never leave a formally named attestation behind after a failed final
        # consistency check.  Consumers also independently rehash checkpoint.
        output.unlink(missing_ok=True)
        raise
    return output, attestation


def load_training_attestation(path: str | Path) -> WarmTrainingAttestation:
    payload = Path(path).read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingAttestationError(
            "training attestation is not valid UTF-8 JSON"
        ) from error
    attestation = WarmTrainingAttestation.from_dict(value)
    if payload != attestation.encode():
        raise TrainingAttestationError(
            "training attestation is not canonical JSON"
        )
    return attestation


def verify_training_attestation(
    checkpoint_path: str | Path,
    attestation_path: str | Path | None = None,
) -> WarmTrainingAttestation:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    attestation_file = (
        training_attestation_path(checkpoint)
        if attestation_path is None
        else Path(attestation_path).expanduser().resolve()
    )
    value = load_training_attestation(attestation_file)
    before = sha256_file(checkpoint)
    if before != value.checkpoint_sha256:
        raise TrainingAttestationError(
            "training attestation checkpoint SHA-256 does not match"
        )
    after = sha256_file(checkpoint)
    if after != before:
        raise TrainingAttestationError(
            "WARM checkpoint changed while its attestation was verified"
        )
    return value


__all__ = [
    "SHARED_RECIPE_IGNORED_PATHS",
    "V1_SHARED_RECIPE_IGNORED_PATHS",
    "TRAINING_ATTESTATION_SCHEMA",
    "TRAINING_ATTESTATION_VERSION",
    "TrainingAttestationError",
    "WarmTrainingAttestation",
    "WarmTrainingRunContext",
    "capture_actual_optimizer_facts",
    "capture_actual_scheduler_chain",
    "capture_training_runtime",
    "clean_git_commit",
    "load_training_attestation",
    "publish_training_attestation",
    "prepare_formal_resume_lineage",
    "sha256_training_state_tree",
    "training_attestation_path",
    "training_config_hashes",
    "verify_training_attestation",
]
