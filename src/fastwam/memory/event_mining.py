"""Deterministic, annotation-free event candidate mining for full episodes.

The miner works on complete, time-major NumPy arrays.  Actions define the
canonical time axis: an episode with ``T`` actions has candidate indices in
``[0, T)``.  State-like inputs may contain either ``T`` samples or ``T + 1``
samples:

* with ``T + 1`` samples, change index ``t`` is ``x[t + 1] - x[t]``;
* with ``T`` samples, change index ``t`` is ``x[t] - x[t - 1]`` and the
  change at index zero is defined to be zero.

All output windows are half-open action ranges ``[start, stop)`` with exactly
the base-policy action horizon.  This module never pads or resamples actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


StartMode = Literal["uniform", "event", "hybrid"]


@dataclass(frozen=True)
class EventMiningConfig:
    """Configuration for deterministic episode event mining.

    Args:
        action_horizon: Number of original action steps in every output
            window.  Episodes shorter than this value are rejected.
        score_quantile: Within-trajectory score quantile required for an
            automatically detected local maximum.  Gripper transitions do
            not need to satisfy this threshold.
        local_max_radius: Radius used to identify local score maxima.
        nms_radius: Minimum temporal separation enforced between automatic
            candidates.  ``None`` uses half of ``action_horizon``.  Forced
            gripper candidates are never removed, even when adjacent.
        uniform_stride: Stride for uniform window starts.  ``None`` uses the
            action horizon, producing non-overlapping coverage plus a final
            tail-aligned window when required.
        gripper_change_threshold: Absolute gripper change above which a step
            is a forced candidate.
        mad_epsilon: Positive numerical floor in median/MAD normalization.
        robust_clip: Maximum value after one-sided robust normalization.
    """

    action_horizon: int
    score_quantile: float = 0.90
    local_max_radius: int = 2
    nms_radius: int | None = None
    uniform_stride: int | None = None
    gripper_change_threshold: float = 1e-6
    mad_epsilon: float = 1e-6
    robust_clip: float = 10.0

    def __post_init__(self) -> None:
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if not 0.0 <= self.score_quantile <= 1.0:
            raise ValueError("score_quantile must be in [0, 1]")
        if self.local_max_radius < 0:
            raise ValueError("local_max_radius must be non-negative")
        if self.nms_radius is not None and self.nms_radius < 0:
            raise ValueError("nms_radius must be non-negative or None")
        if self.uniform_stride is not None and self.uniform_stride <= 0:
            raise ValueError("uniform_stride must be positive or None")
        if self.gripper_change_threshold < 0.0:
            raise ValueError("gripper_change_threshold must be non-negative")
        if self.mad_epsilon <= 0.0:
            raise ValueError("mad_epsilon must be positive")
        if self.robust_clip <= 0.0:
            raise ValueError("robust_clip must be positive")

    @property
    def resolved_nms_radius(self) -> int:
        if self.nms_radius is not None:
            return self.nms_radius
        return self.action_horizon // 2

    @property
    def resolved_uniform_stride(self) -> int:
        if self.uniform_stride is not None:
            return self.uniform_stride
        return self.action_horizon


@dataclass(frozen=True)
class EventMiningResult:
    """Mining output on the original action time axis.

    ``windows[:, 0]`` is inclusive and ``windows[:, 1]`` is exclusive.
    ``component_scores`` contains the independently median/MAD-normalized
    change signals used in the composite ``scores``.
    """

    scores: NDArray[np.float64]
    component_scores: Mapping[str, NDArray[np.float64]]
    candidate_indices: NDArray[np.int64]
    forced_gripper_indices: NDArray[np.int64]
    window_starts: NDArray[np.int64]
    windows: NDArray[np.int64]


def robust_mad_normalize(
    values: ArrayLike,
    *,
    epsilon: float = 1e-6,
    clip: float = 10.0,
) -> NDArray[np.float64]:
    """One-sided robust normalization using trajectory median and MAD.

    Change magnitudes are non-negative and only unusually *large* changes are
    event evidence, so values below the trajectory median are clipped to zero.
    ``1.4826 * MAD`` is the normal-consistent robust scale.  Adding ``epsilon``
    also makes sparse signals with zero MAD finite and deterministic.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("values must not be empty")
    if not np.isfinite(array).all():
        raise ValueError("values must contain only finite numbers")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if clip <= 0.0:
        raise ValueError("clip must be positive")

    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = 1.4826 * mad + epsilon
    normalized = np.maximum((array - median) / scale, 0.0)
    return np.minimum(normalized, clip).astype(np.float64, copy=False)


def _as_time_major(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"{name} must have a leading time dimension")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numbers")
    return array.reshape(array.shape[0], -1)


def _validate_state_length(name: str, length: int, num_actions: int) -> None:
    if length not in (num_actions, num_actions + 1):
        raise ValueError(
            f"{name} must have T or T+1 samples for T actions; "
            f"got {length} samples and {num_actions} actions"
        )


def _rms_change(
    name: str,
    values: ArrayLike,
    num_actions: int,
) -> NDArray[np.float64]:
    """Align a state-like sequence's RMS temporal change to action indices."""

    array = _as_time_major(name, values)
    _validate_state_length(name, array.shape[0], num_actions)

    if array.shape[0] == num_actions + 1:
        delta = array[1:] - array[:-1]
        return np.sqrt(np.mean(np.square(delta), axis=1))

    result = np.zeros(num_actions, dtype=np.float64)
    if num_actions > 1:
        delta = array[1:] - array[:-1]
        result[1:] = np.sqrt(np.mean(np.square(delta), axis=1))
    return result


def _gripper_change_mask(
    gripper: ArrayLike,
    num_actions: int,
    threshold: float,
) -> NDArray[np.bool_]:
    array = np.asarray(gripper, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(
            "gripper must have shape [T], [T+1], [T, 1], or [T+1, 1]; "
            f"got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("gripper must contain only finite numbers")
    _validate_state_length("gripper", array.shape[0], num_actions)

    result = np.zeros(num_actions, dtype=np.bool_)
    changes = np.abs(array[1:] - array[:-1]) > threshold
    if array.shape[0] == num_actions + 1:
        result[:] = changes
    elif num_actions > 1:
        result[1:] = changes
    return result


def _local_maxima(
    scores: NDArray[np.float64],
    threshold: float,
    radius: int,
) -> NDArray[np.int64]:
    """Return local maxima, collapsing an exact contiguous plateau to its left edge."""

    maxima: list[int] = []
    length = scores.shape[0]
    for index, value in enumerate(scores):
        if value <= 0.0 or value < threshold:
            continue
        left = max(0, index - radius)
        right = min(length, index + radius + 1)
        if value == float(np.max(scores[left:right])):
            maxima.append(index)

    if not maxima:
        return np.empty(0, dtype=np.int64)

    collapsed = [maxima[0]]
    for index in maxima[1:]:
        previous = collapsed[-1]
        same_plateau = index == previous + 1 and scores[index] == scores[previous]
        if not same_plateau:
            collapsed.append(index)
    return np.asarray(collapsed, dtype=np.int64)


def _nms_with_forced_candidates(
    automatic: NDArray[np.int64],
    forced: NDArray[np.int64],
    scores: NDArray[np.float64],
    radius: int,
) -> NDArray[np.int64]:
    """Apply NMS to automatic peaks while preserving every forced candidate."""

    forced_list = sorted({int(index) for index in forced})
    forced_set = set(forced_list)
    automatic_order = sorted(
        (int(index) for index in automatic if int(index) not in forced_set),
        key=lambda index: (-float(scores[index]), index),
    )

    accepted = list(forced_list)
    for index in automatic_order:
        if all(abs(index - kept) > radius for kept in accepted):
            accepted.append(index)
    return np.asarray(sorted(accepted), dtype=np.int64)


def mine_event_candidates(
    actions: ArrayLike,
    proprio: ArrayLike,
    gripper: ArrayLike,
    *,
    config: EventMiningConfig,
    semantic_features: ArrayLike | None = None,
    vae_features: ArrayLike | None = None,
) -> tuple[
    NDArray[np.float64],
    Mapping[str, NDArray[np.float64]],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    """Compute event scores and deterministic candidate action indices.

    The composite score is the arithmetic mean of independently normalized
    action, proprioception, and optional visual change magnitudes.  Gripper
    transitions are kept separately and forcibly included after local-maximum
    detection and NMS.
    """

    action_array = _as_time_major("actions", actions)
    num_actions = action_array.shape[0]
    if num_actions < config.action_horizon:
        raise ValueError(
            f"episode has {num_actions} actions, shorter than action_horizon "
            f"{config.action_horizon}; padding and resampling are not allowed"
        )

    raw_components: dict[str, NDArray[np.float64]] = {
        "action_change": _rms_change("actions", action_array, num_actions),
        "proprio_change": _rms_change("proprio", proprio, num_actions),
    }
    if semantic_features is not None:
        raw_components["semantic_change"] = _rms_change(
            "semantic_features", semantic_features, num_actions
        )
    if vae_features is not None:
        raw_components["vae_change"] = _rms_change(
            "vae_features", vae_features, num_actions
        )

    normalized_components = {
        name: robust_mad_normalize(
            component,
            epsilon=config.mad_epsilon,
            clip=config.robust_clip,
        )
        for name, component in raw_components.items()
    }
    scores = np.mean(
        np.stack(tuple(normalized_components.values()), axis=0),
        axis=0,
        dtype=np.float64,
    )

    threshold = float(np.quantile(scores, config.score_quantile))
    automatic = _local_maxima(scores, threshold, config.local_max_radius)
    gripper_mask = _gripper_change_mask(
        gripper,
        num_actions,
        config.gripper_change_threshold,
    )
    forced = np.flatnonzero(gripper_mask).astype(np.int64, copy=False)
    candidates = _nms_with_forced_candidates(
        automatic,
        forced,
        scores,
        config.resolved_nms_radius,
    )
    return scores, normalized_components, candidates, forced


def select_window_starts(
    *,
    num_actions: int,
    candidate_indices: ArrayLike,
    config: EventMiningConfig,
    mode: StartMode,
) -> NDArray[np.int64]:
    """Select deterministic fixed-horizon action-window starts.

    Modes:

    * ``uniform``: regular, non-overlapping-by-default coverage, with an
      additional tail-aligned start when the episode length is not divisible
      by the stride;
    * ``event``: windows centered on candidates and clamped at boundaries;
    * ``hybrid``: the sorted union of uniform and event starts.

    Duplicate starts caused by boundary clamping or nearby candidates are
    removed.  No action values are read, padded, interpolated, or resampled.
    """

    if mode not in ("uniform", "event", "hybrid"):
        raise ValueError(f"unsupported start mode {mode!r}")
    if num_actions < config.action_horizon:
        raise ValueError(
            f"episode has {num_actions} actions, shorter than action_horizon "
            f"{config.action_horizon}; padding and resampling are not allowed"
        )

    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.ndim != 1:
        raise ValueError("candidate_indices must be one-dimensional")
    if candidates.size and (
        int(candidates.min()) < 0 or int(candidates.max()) >= num_actions
    ):
        raise ValueError("candidate_indices must lie in [0, num_actions)")

    max_start = num_actions - config.action_horizon
    stride = config.resolved_uniform_stride
    uniform = np.arange(0, max_start + 1, stride, dtype=np.int64)
    if uniform.size == 0 or int(uniform[-1]) != max_start:
        uniform = np.append(uniform, np.int64(max_start))

    event_starts = np.clip(
        candidates - config.action_horizon // 2,
        0,
        max_start,
    )
    event_starts = np.unique(event_starts).astype(np.int64, copy=False)

    if mode == "uniform":
        return uniform
    if mode == "event":
        return event_starts
    return np.union1d(uniform, event_starts).astype(np.int64, copy=False)


def mine_episode_events(
    actions: ArrayLike,
    proprio: ArrayLike,
    gripper: ArrayLike,
    *,
    config: EventMiningConfig,
    start_mode: StartMode = "hybrid",
    semantic_features: ArrayLike | None = None,
    vae_features: ArrayLike | None = None,
) -> EventMiningResult:
    """Mine candidates and fixed-horizon windows from one complete episode."""

    scores, components, candidates, forced = mine_event_candidates(
        actions,
        proprio,
        gripper,
        config=config,
        semantic_features=semantic_features,
        vae_features=vae_features,
    )
    starts = select_window_starts(
        num_actions=scores.shape[0],
        candidate_indices=candidates,
        config=config,
        mode=start_mode,
    )
    if starts.size:
        windows = np.column_stack(
            (starts, starts + config.action_horizon)
        ).astype(np.int64, copy=False)
    else:
        windows = np.empty((0, 2), dtype=np.int64)

    return EventMiningResult(
        scores=scores,
        component_scores=components,
        candidate_indices=candidates,
        forced_gripper_indices=forced,
        window_starts=starts,
        windows=windows,
    )


__all__ = [
    "EventMiningConfig",
    "EventMiningResult",
    "StartMode",
    "mine_episode_events",
    "mine_event_candidates",
    "robust_mad_normalize",
    "select_window_starts",
]
