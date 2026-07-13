"""Torch contracts for M2 source-only WARM.

M2 deliberately contains no learned consequence scorer.  A deterministic
policy selects either the explicit Gaussian null component or one event from a
precomputed, leave-episode-out candidate row.  The selected model-space action
is then used as the mean of the FastWAM flow source::

    source_null = epsilon
    source_memory = mu + memory_sigma * epsilon

The caller must draw ``epsilon`` using FastWAM's original RNG path *before*
calling this module.  Keeping selection and Gaussian generation separate is
what makes the null path reproducible and prevents a categorical draw from
changing the action-flow timestep or source noise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Protocol

import torch


NULL_COMPONENT = 0
SourcePolicy = Literal[
    "gaussian_null",
    "fixed_context_top1",
    "oracle_action_top1",
]
SourcePhase = Literal["train", "offline", "infer", "rollout"]


class SourceTransportError(ValueError):
    """Raised when a source-selection or source-tensor contract is invalid."""


class FlowScheduler(Protocol):
    """The FastWAM scheduler methods used by source-only training."""

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor: ...

    def training_target(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class ActionSourceContext:
    """Deterministic candidate payload passed to the model source hook.

    ``component_indices`` uses the paper/runtime convention: zero is the null
    component and memory candidate ``j`` is component ``j + 1``.  Candidate
    actions are already normalized FastWAM model-space actions.  They must not
    be denormalized, interpolated, or averaged by this module.
    """

    component_indices: torch.Tensor
    candidate_means: torch.Tensor | None
    candidate_valid_mask: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class ActionSourceOutput:
    """Resolved action source and selection telemetry."""

    source: torch.Tensor
    base_gaussian: torch.Tensor
    component_indices: torch.Tensor
    memory_mask: torch.Tensor
    selected_means: torch.Tensor | None
    memory_sigma: float
    # Complete WARM may append compact gist/action tokens to Action DiT's raw
    # conditioning sequence.  M2 leaves these fields unset and preserves its
    # exact source-only behavior.
    conditioning_tokens: torch.Tensor | None = None
    auxiliary_loss: torch.Tensor | None = None
    auxiliary_metrics: Mapping[str, torch.Tensor] | None = None
    source_gate: torch.Tensor | None = None

    @property
    def selected_candidate_indices(self) -> torch.Tensor:
        """Return zero-based candidate indices, with ``-1`` for null rows."""

        return self.component_indices - 1


@dataclass(frozen=True, slots=True)
class ActionFlowPair:
    """Scheduler inputs/targets for a source-to-action flow sample."""

    noisy_action: torch.Tensor
    target_velocity: torch.Tensor
    source_action: torch.Tensor


@dataclass(frozen=True, slots=True)
class SourceGeometry:
    """Per-sample source-to-target geometry in model action space."""

    rms: torch.Tensor
    l2: torch.Tensor

    @property
    def mean_rms(self) -> torch.Tensor:
        return self.rms.mean()

    @property
    def mean_l2(self) -> torch.Tensor:
        return self.l2.mean()


def _require_tensor(value: object, field: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    return value


def _require_action_tensor(value: object, field: str) -> torch.Tensor:
    tensor = _require_tensor(value, field)
    if tensor.ndim != 3:
        raise SourceTransportError(
            f"{field} must have shape [B,H,D], got {tuple(tensor.shape)}"
        )
    if any(int(size) <= 0 for size in tensor.shape):
        raise SourceTransportError(
            f"{field} must have non-empty [B,H,D] dimensions, got {tuple(tensor.shape)}"
        )
    if not tensor.is_floating_point():
        raise TypeError(f"{field} must have a floating dtype, got {tensor.dtype}")
    return tensor


def _require_bool_matrix(value: object, field: str) -> torch.Tensor:
    tensor = _require_tensor(value, field)
    if tensor.ndim != 2:
        raise SourceTransportError(
            f"{field} must have shape [B,K], got {tuple(tensor.shape)}"
        )
    if tensor.dtype != torch.bool:
        raise TypeError(f"{field} must have dtype bool, got {tensor.dtype}")
    return tensor


def _require_component_vector(
    value: object,
    *,
    batch_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    tensor = _require_tensor(value, "component_indices")
    if tensor.dtype != torch.long:
        raise TypeError(
            f"component_indices must have dtype int64, got {tensor.dtype}"
        )
    if tensor.shape != (batch_size,):
        raise SourceTransportError(
            "component_indices must have shape "
            f"{(batch_size,)}, got {tuple(tensor.shape)}"
        )
    if device is not None and tensor.device != device:
        raise SourceTransportError(
            "component_indices must be on the same device as the action source: "
            f"{tensor.device} != {device}"
        )
    if bool(torch.any(tensor < 0).item()):
        raise SourceTransportError("component_indices must be non-negative")
    return tensor


def _finite_positive(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise SourceTransportError(f"{field} must be a finite positive number")
    return result


def _validate_phase(phase: object) -> SourcePhase:
    if phase not in ("train", "offline", "infer", "rollout"):
        raise SourceTransportError(
            "phase must be one of 'train', 'offline', 'infer', or 'rollout'"
        )
    return phase  # type: ignore[return-value]


def select_source_components(
    candidate_valid_mask: torch.Tensor,
    *,
    policy: SourcePolicy,
    phase: SourcePhase,
    oracle_candidate_indices: torch.Tensor | None = None,
    memory_enabled_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select explicit null/memory components without reading action payloads.

    ``fixed_context_top1`` means cache rank zero exactly.  It does not skip an
    invalid first slot to use a later slot.  ``oracle_action_top1`` consumes a
    precomputed zero-based oracle index and is rejected for inference/rollout;
    it exists only as an upper-bound experiment, never as a deployable policy.
    An oracle index of ``-1`` denotes null.
    """

    valid = _require_bool_matrix(candidate_valid_mask, "candidate_valid_mask")
    phase = _validate_phase(phase)
    if policy not in (
        "gaussian_null",
        "fixed_context_top1",
        "oracle_action_top1",
    ):
        raise SourceTransportError(f"unsupported source policy {policy!r}")

    batch_size, num_candidates = valid.shape
    enabled = torch.ones((batch_size,), dtype=torch.bool, device=valid.device)
    if memory_enabled_mask is not None:
        enabled_input = _require_tensor(memory_enabled_mask, "memory_enabled_mask")
        if enabled_input.dtype != torch.bool:
            raise TypeError(
                "memory_enabled_mask must have dtype bool, "
                f"got {enabled_input.dtype}"
            )
        if enabled_input.shape != (batch_size,):
            raise SourceTransportError(
                "memory_enabled_mask must have shape "
                f"{(batch_size,)}, got {tuple(enabled_input.shape)}"
            )
        if enabled_input.device != valid.device:
            raise SourceTransportError(
                "memory_enabled_mask and candidate_valid_mask must share a device"
            )
        enabled = enabled_input

    components = torch.zeros((batch_size,), dtype=torch.long, device=valid.device)
    if policy == "gaussian_null" or num_candidates == 0:
        return components

    if policy == "fixed_context_top1":
        use_memory = enabled & valid[:, 0]
        components[use_memory] = 1
        return components

    if phase in ("infer", "rollout"):
        raise SourceTransportError(
            "oracle_action_top1 is an offline/training upper bound and is forbidden "
            f"during {phase}"
        )
    if oracle_candidate_indices is None:
        raise SourceTransportError(
            "oracle_action_top1 requires precomputed oracle_candidate_indices"
        )
    oracle = _require_tensor(oracle_candidate_indices, "oracle_candidate_indices")
    if oracle.dtype != torch.long:
        raise TypeError(
            f"oracle_candidate_indices must have dtype int64, got {oracle.dtype}"
        )
    if oracle.shape != (batch_size,):
        raise SourceTransportError(
            "oracle_candidate_indices must have shape "
            f"{(batch_size,)}, got {tuple(oracle.shape)}"
        )
    if oracle.device != valid.device:
        raise SourceTransportError(
            "oracle_candidate_indices and candidate_valid_mask must share a device"
        )
    if bool(torch.any(oracle < -1).item()) or bool(
        torch.any(oracle >= num_candidates).item()
    ):
        raise SourceTransportError(
            f"oracle_candidate_indices must lie in [-1, {num_candidates})"
        )

    memory_rows = enabled & (oracle >= 0)
    if bool(memory_rows.any().item()):
        rows = torch.nonzero(memory_rows, as_tuple=False).squeeze(1)
        positions = oracle[rows]
        if not bool(valid[rows, positions].all().item()):
            raise SourceTransportError(
                "oracle_candidate_indices selects an invalid candidate slot"
            )
        components[rows] = positions + 1
    return components


def resolve_action_source(
    base_gaussian: torch.Tensor,
    context: ActionSourceContext,
    *,
    memory_sigma: float,
) -> ActionSourceOutput:
    """Resolve a null or memory source using one already-drawn Gaussian.

    The all-null fast path returns ``base_gaussian`` itself and deliberately
    does not inspect ``candidate_means`` or ``candidate_valid_mask``.  This is a
    behavioral contract, not merely an optimization: fallback must be safe even
    if long-term payload storage is absent or unavailable.
    """

    gaussian = _require_action_tensor(base_gaussian, "base_gaussian")
    if not isinstance(context, ActionSourceContext):
        raise TypeError("context must be ActionSourceContext")
    components = _require_component_vector(
        context.component_indices,
        batch_size=gaussian.shape[0],
        device=gaussian.device,
    )
    sigma = _finite_positive(memory_sigma, "memory_sigma")
    memory_mask = components != NULL_COMPONENT

    if not bool(memory_mask.any().item()):
        return ActionSourceOutput(
            source=gaussian,
            base_gaussian=gaussian,
            component_indices=components.detach(),
            memory_mask=memory_mask,
            selected_means=None,
            memory_sigma=sigma,
        )

    means = _require_tensor(context.candidate_means, "candidate_means")
    valid = _require_bool_matrix(
        context.candidate_valid_mask, "candidate_valid_mask"
    )
    batch_size, horizon, action_dim = gaussian.shape
    if means.ndim != 4:
        raise SourceTransportError(
            "candidate_means must have shape [B,K,H,D], "
            f"got {tuple(means.shape)}"
        )
    if means.shape[0] != batch_size or means.shape[2:] != (horizon, action_dim):
        raise SourceTransportError(
            "candidate_means must have shape "
            f"[B,K,{horizon},{action_dim}], got {tuple(means.shape)}"
        )
    if means.dtype != gaussian.dtype:
        raise TypeError(
            "candidate_means and base_gaussian must have the same dtype: "
            f"{means.dtype} != {gaussian.dtype}"
        )
    if means.device != gaussian.device or valid.device != gaussian.device:
        raise SourceTransportError(
            "candidate_means, candidate_valid_mask, and base_gaussian must share a device"
        )
    if valid.shape != means.shape[:2]:
        raise SourceTransportError(
            "candidate_valid_mask must match candidate_means [B,K]: "
            f"{tuple(valid.shape)} != {tuple(means.shape[:2])}"
        )
    if not means.is_floating_point():
        raise TypeError(
            f"candidate_means must have a floating dtype, got {means.dtype}"
        )

    rows = torch.nonzero(memory_mask, as_tuple=False).squeeze(1)
    candidate_positions = components[rows] - 1
    if bool(torch.any(candidate_positions >= means.shape[1]).item()):
        raise SourceTransportError(
            "component_indices refers to a candidate outside candidate_means"
        )
    if not bool(valid[rows, candidate_positions].all().item()):
        raise SourceTransportError(
            "component_indices selects an invalid candidate slot"
        )

    selected_memory_means = means[rows, candidate_positions].detach()
    if not bool(torch.isfinite(selected_memory_means).all().item()):
        raise SourceTransportError("selected candidate means must be finite")
    if not bool(torch.isfinite(gaussian).all().item()):
        raise SourceTransportError("base_gaussian must be finite")

    source = gaussian.clone()
    source[rows] = selected_memory_means + sigma * gaussian[rows]
    source = source.detach()
    if not bool(torch.isfinite(source).all().item()):
        raise SourceTransportError("resolved action source must be finite")

    selected_means = torch.zeros_like(gaussian)
    selected_means[rows] = selected_memory_means
    return ActionSourceOutput(
        source=source,
        base_gaussian=gaussian,
        component_indices=components.detach(),
        memory_mask=memory_mask,
        selected_means=selected_means.detach(),
        memory_sigma=sigma,
    )


def build_action_flow_pair(
    action: torch.Tensor,
    source_action: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: FlowScheduler,
) -> ActionFlowPair:
    """Use FastWAM's unchanged scheduler to build one source-to-data pair."""

    target = _require_action_tensor(action, "action")
    source = _require_action_tensor(source_action, "source_action")
    if source.shape != target.shape:
        raise SourceTransportError(
            f"source_action shape {tuple(source.shape)} != action shape {tuple(target.shape)}"
        )
    if source.dtype != target.dtype:
        raise TypeError(
            f"source_action dtype {source.dtype} != action dtype {target.dtype}"
        )
    if source.device != target.device:
        raise SourceTransportError(
            f"source_action device {source.device} != action device {target.device}"
        )
    step = _require_tensor(timestep, "timestep")
    if step.shape != (target.shape[0],):
        raise SourceTransportError(
            f"timestep must have shape {(target.shape[0],)}, got {tuple(step.shape)}"
        )
    if step.device != target.device:
        raise SourceTransportError("timestep and action must share a device")
    if not hasattr(scheduler, "add_noise") or not hasattr(
        scheduler, "training_target"
    ):
        raise TypeError(
            "scheduler must provide add_noise() and training_target()"
        )

    detached_source = source.detach()
    noisy = scheduler.add_noise(target, detached_source, step)
    velocity = scheduler.training_target(target, detached_source, step)
    if not isinstance(noisy, torch.Tensor) or not isinstance(velocity, torch.Tensor):
        raise TypeError("scheduler methods must return torch.Tensor values")
    if noisy.shape != target.shape or velocity.shape != target.shape:
        raise SourceTransportError(
            "scheduler outputs must preserve the [B,H,D] action shape"
        )
    return ActionFlowPair(
        noisy_action=noisy,
        target_velocity=velocity,
        source_action=detached_source,
    )


def measure_source_geometry(
    source_action: torch.Tensor,
    target_action: torch.Tensor,
    *,
    action_is_pad: torch.Tensor | None = None,
) -> SourceGeometry:
    """Measure paired source displacement with the training padding contract."""

    source = _require_action_tensor(source_action, "source_action")
    target = _require_action_tensor(target_action, "target_action")
    if source.shape != target.shape:
        raise SourceTransportError(
            f"source/target shapes must match, got {tuple(source.shape)} and {tuple(target.shape)}"
        )
    if source.dtype != target.dtype:
        raise TypeError("source_action and target_action must share a dtype")
    if source.device != target.device:
        raise SourceTransportError(
            "source_action and target_action must share a device"
        )

    squared = (source.float() - target.float()).square()
    if action_is_pad is None:
        valid = torch.ones(
            source.shape[:2], dtype=torch.bool, device=source.device
        )
    else:
        valid = _require_tensor(action_is_pad, "action_is_pad")
        if valid.dtype != torch.bool:
            raise TypeError(f"action_is_pad must have dtype bool, got {valid.dtype}")
        if valid.shape != source.shape[:2]:
            raise SourceTransportError(
                "action_is_pad must have shape "
                f"{tuple(source.shape[:2])}, got {tuple(valid.shape)}"
            )
        if valid.device != source.device:
            raise SourceTransportError(
                "action_is_pad and action tensors must share a device"
            )
        valid = ~valid

    valid_float = valid.to(dtype=squared.dtype).unsqueeze(-1)
    squared = squared * valid_float
    valid_steps = valid_float.sum(dim=(1, 2))
    rms_denominator = (valid_steps * source.shape[2]).clamp(min=1.0)
    rms = torch.sqrt(squared.sum(dim=(1, 2)) / rms_denominator)
    l2 = torch.sqrt(squared.sum(dim=(1, 2)))
    return SourceGeometry(rms=rms, l2=l2)


__all__ = [
    "NULL_COMPONENT",
    "ActionFlowPair",
    "ActionSourceContext",
    "ActionSourceOutput",
    "SourceGeometry",
    "SourcePhase",
    "SourcePolicy",
    "SourceTransportError",
    "build_action_flow_pair",
    "measure_source_geometry",
    "resolve_action_source",
    "select_source_components",
]
