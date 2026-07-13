"""Oracle retrieval diagnostics for the M1 event-bank gate.

The oracle is deliberately *not* an online policy component.  It asks whether
the exact context top-K contains a fixed-horizon model-space action which is
closer to the query ground truth than context top-1 or a recent-action prior.
Every lookup leaves the complete query episode out by identity and available
source/feature-content hashes through :class:`EventBank`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Literal, Sequence

import numpy as np

from .event_bank import EventBank
from .payload_names import MODEL_SPACE_ACTION
from .schema import EpisodeKey, EventId, coerce_episode_key


ArmLoss = Literal["mse", "huber"]


class OracleMetricError(ValueError):
    """Raised when an oracle query or action metric contract is invalid."""


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be a positive integer")
    if value <= 0:
        raise OracleMetricError(f"{field} must be positive")
    return value


def _finite_float(value: object, field: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise OracleMetricError(f"{field} must be finite")
    if non_negative and result < 0.0:
        raise OracleMetricError(f"{field} must be non-negative")
    return result


def _dimensions(value: tuple[int, ...], field: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field} must be a tuple of integer dimensions")
    result: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TypeError(f"{field} must contain only integers")
        if dimension < 0:
            raise OracleMetricError(f"{field} must contain non-negative dimensions")
        result.append(dimension)
    if len(set(result)) != len(result):
        raise OracleMetricError(f"{field} must not contain duplicate dimensions")
    return tuple(result)


def _readonly_float_array(value: np.ndarray, field: str, *, ndim: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{field} must be a NumPy array")
    if value.dtype.kind != "f":
        raise TypeError(f"{field} must have a floating dtype, got {value.dtype}")
    if value.ndim != ndim:
        raise OracleMetricError(f"{field} must have rank {ndim}, got shape {value.shape}")
    if any(dimension <= 0 for dimension in value.shape):
        raise OracleMetricError(f"{field} dimensions must be non-zero, got {value.shape}")
    if not np.isfinite(value).all():
        raise OracleMetricError(f"{field} must contain only finite values")
    result = np.array(value, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class ActionDistanceConfig:
    """Explicit model-action layout and component weighting.

    ``arm_dims`` and ``gripper_dims`` must be disjoint and together cover
    ``range(action_dim)``.  This prevents a dataset-specific action channel
    from being silently omitted.  Multiple gripper dimensions are supported
    for bimanual action spaces.
    """

    action_dim: int
    arm_dims: tuple[int, ...]
    gripper_dims: tuple[int, ...]
    arm_loss: ArmLoss = "mse"
    huber_delta: float = 1.0
    arm_weight: float = 1.0
    gripper_state_weight: float = 1.0
    gripper_timing_weight: float = 1.0
    gripper_threshold: float = 0.0

    def __post_init__(self) -> None:
        action_dim = _positive_int(self.action_dim, "action_dim")
        arm_dims = _dimensions(self.arm_dims, "arm_dims")
        gripper_dims = _dimensions(self.gripper_dims, "gripper_dims")
        if not arm_dims:
            raise OracleMetricError("arm_dims must contain at least one dimension")
        if set(arm_dims) & set(gripper_dims):
            raise OracleMetricError("arm_dims and gripper_dims must be disjoint")
        covered = set(arm_dims) | set(gripper_dims)
        expected = set(range(action_dim))
        if covered != expected:
            raise OracleMetricError(
                "arm_dims and gripper_dims must exactly cover range(action_dim); "
                f"missing={sorted(expected - covered)}, out_of_range={sorted(covered - expected)}"
            )
        if self.arm_loss not in ("mse", "huber"):
            raise OracleMetricError("arm_loss must be 'mse' or 'huber'")
        huber_delta = _finite_float(self.huber_delta, "huber_delta")
        if huber_delta <= 0.0:
            raise OracleMetricError("huber_delta must be positive")
        arm_weight = _finite_float(self.arm_weight, "arm_weight", non_negative=True)
        state_weight = _finite_float(
            self.gripper_state_weight, "gripper_state_weight", non_negative=True
        )
        timing_weight = _finite_float(
            self.gripper_timing_weight, "gripper_timing_weight", non_negative=True
        )
        if arm_weight + state_weight + timing_weight <= 0.0:
            raise OracleMetricError("at least one action-distance weight must be positive")
        threshold = _finite_float(self.gripper_threshold, "gripper_threshold")

        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "arm_dims", arm_dims)
        object.__setattr__(self, "gripper_dims", gripper_dims)
        object.__setattr__(self, "huber_delta", huber_delta)
        object.__setattr__(self, "arm_weight", arm_weight)
        object.__setattr__(self, "gripper_state_weight", state_weight)
        object.__setattr__(self, "gripper_timing_weight", timing_weight)
        object.__setattr__(self, "gripper_threshold", threshold)


@dataclass(frozen=True, slots=True)
class ActionDistance:
    total: float
    arm: float
    gripper_state: float
    gripper_timing: float


def _validate_action(
    value: np.ndarray, field: str, config: ActionDistanceConfig
) -> np.ndarray:
    action = _readonly_float_array(value, field, ndim=2)
    if action.shape[1] != config.action_dim:
        raise OracleMetricError(
            f"{field} must have action dimension {config.action_dim}, got {action.shape}"
        )
    return action


def action_distance(
    candidate: np.ndarray,
    target: np.ndarray,
    config: ActionDistanceConfig,
) -> ActionDistance:
    """Measure two fixed-horizon model-space action chunks.

    Arm loss is averaged over ``[horizon, arm_dims]``. ``gripper_state`` here
    means the thresholded *model-action gripper command*, not the observed
    robot gripper state stored elsewhere in the event payload. Its mismatch is
    the fraction of command states which differ. Timing mismatch is the
    fraction of adjacent command-transition indicators which differ; it is
    zero for a one-step horizon or when ``gripper_dims`` is empty.
    """

    if not isinstance(config, ActionDistanceConfig):
        raise TypeError("config must be ActionDistanceConfig")
    candidate_action = _validate_action(candidate, "candidate", config)
    target_action = _validate_action(target, "target", config)
    if candidate_action.shape != target_action.shape:
        raise OracleMetricError(
            f"candidate and target shapes must match, got "
            f"{candidate_action.shape} and {target_action.shape}"
        )

    difference = (
        candidate_action[:, config.arm_dims].astype(np.float64)
        - target_action[:, config.arm_dims].astype(np.float64)
    )
    if config.arm_loss == "mse":
        arm = float(np.mean(np.square(difference)))
    else:
        absolute = np.abs(difference)
        huber = np.where(
            absolute <= config.huber_delta,
            0.5 * np.square(absolute),
            config.huber_delta * (absolute - 0.5 * config.huber_delta),
        )
        arm = float(np.mean(huber))

    if config.gripper_dims:
        candidate_state = (
            candidate_action[:, config.gripper_dims] > config.gripper_threshold
        )
        target_state = target_action[:, config.gripper_dims] > config.gripper_threshold
        gripper_state = float(np.mean(candidate_state != target_state))
        if candidate_action.shape[0] > 1:
            candidate_transition = candidate_state[1:] != candidate_state[:-1]
            target_transition = target_state[1:] != target_state[:-1]
            gripper_timing = float(np.mean(candidate_transition != target_transition))
        else:
            gripper_timing = 0.0
    else:
        gripper_state = 0.0
        gripper_timing = 0.0

    total = (
        config.arm_weight * arm
        + config.gripper_state_weight * gripper_state
        + config.gripper_timing_weight * gripper_timing
    )
    return ActionDistance(
        total=float(total),
        arm=arm,
        gripper_state=gripper_state,
        gripper_timing=gripper_timing,
    )


@dataclass(frozen=True, slots=True)
class OracleQuery:
    """One leak-free oracle evaluation query."""

    query_key: np.ndarray
    gt_model_action: np.ndarray
    episode_key: EpisodeKey | EventId
    source_episode_sha256: str
    feature_episode_sha256: str
    recent_action: np.ndarray | None = None

    def __post_init__(self) -> None:
        query_key = _readonly_float_array(self.query_key, "query_key", ndim=1)
        if float(np.linalg.norm(query_key.astype(np.float64))) <= 0.0:
            raise OracleMetricError("query_key must have non-zero cosine norm")
        gt_action = _readonly_float_array(
            self.gt_model_action, "gt_model_action", ndim=2
        )
        episode_key = coerce_episode_key(self.episode_key)
        recent_action = self.recent_action
        if recent_action is not None:
            recent_action = _readonly_float_array(recent_action, "recent_action", ndim=2)
            if recent_action.shape != gt_action.shape:
                raise OracleMetricError(
                    "recent_action must match gt_model_action shape; "
                    f"got {recent_action.shape} and {gt_action.shape}"
                )
        source_hash = self.source_episode_sha256
        if not isinstance(source_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", source_hash
        ) is None:
            raise OracleMetricError(
                "source_episode_sha256 must be a lowercase SHA-256 digest"
            )
        feature_hash = self.feature_episode_sha256
        if not isinstance(feature_hash, str) or re.fullmatch(
            r"[0-9a-f]{64}", feature_hash
        ) is None:
            raise OracleMetricError(
                "feature_episode_sha256 must be a lowercase SHA-256 digest"
            )
        object.__setattr__(self, "query_key", query_key)
        object.__setattr__(self, "gt_model_action", gt_action)
        object.__setattr__(self, "episode_key", episode_key)
        object.__setattr__(self, "recent_action", recent_action)
        object.__setattr__(self, "source_episode_sha256", source_hash)
        object.__setattr__(self, "feature_episode_sha256", feature_hash)


@dataclass(frozen=True, slots=True)
class OracleQueryResult:
    query_index: int
    episode_key: EpisodeKey
    candidate_count: int
    context_top1_event_id: EventId | None
    context_top1_score: float | None
    context_top1_distance: ActionDistance | None
    oracle_event_id: EventId | None
    oracle_rank: int | None
    oracle_distance: ActionDistance | None
    recent_baseline_distance: ActionDistance | None
    relative_improvement_vs_context: float | None
    relative_improvement_vs_recent: float | None

    @property
    def covered(self) -> bool:
        return self.candidate_count > 0


@dataclass(frozen=True, slots=True)
class OracleMetricsReport:
    top_k: int
    query_count: int
    covered_query_count: int
    coverage: float
    context_top1_mean: ActionDistance | None
    oracle_topk_mean: ActionDistance | None
    recent_baseline_mean: ActionDistance | None
    recent_baseline_query_count: int
    recent_comparable_query_count: int
    relative_improvement_vs_context: float | None
    relative_improvement_vs_recent: float | None
    per_query: tuple[OracleQueryResult, ...]


def _relative_improvement(baseline: float | None, improved: float | None) -> float | None:
    if baseline is None or improved is None or baseline <= 0.0:
        return None
    return float((baseline - improved) / baseline)


def _mean_distance(values: Sequence[ActionDistance]) -> ActionDistance | None:
    if not values:
        return None
    return ActionDistance(
        total=float(np.mean([value.total for value in values], dtype=np.float64)),
        arm=float(np.mean([value.arm for value in values], dtype=np.float64)),
        gripper_state=float(
            np.mean([value.gripper_state for value in values], dtype=np.float64)
        ),
        gripper_timing=float(
            np.mean([value.gripper_timing for value in values], dtype=np.float64)
        ),
    )


def _json_int(value: object, field: str, *, non_negative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if non_negative and value < 0:
        raise OracleMetricError(f"{field} must be non-negative")
    return value


def _json_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise OracleMetricError(f"{field} must be finite for JSON serialization")
    return result


def _optional_json_float(value: object | None, field: str) -> float | None:
    return None if value is None else _json_float(value, field)


def _episode_key_to_dict(value: EpisodeKey | EventId, field: str) -> dict[str, str | int]:
    dataset_id, dataset_index, episode_index = coerce_episode_key(value)
    return {
        "dataset_id": dataset_id,
        "dataset_index": dataset_index,
        "episode_index": episode_index,
    }


def _event_id_to_dict(value: EventId | None, field: str) -> dict[str, str | int] | None:
    if value is None:
        return None
    if not isinstance(value, EventId):
        raise TypeError(f"{field} must be EventId or None")
    return value.to_dict()


def _action_distance_to_dict(
    value: ActionDistance | None, field: str
) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, ActionDistance):
        raise TypeError(f"{field} must be ActionDistance or None")
    return {
        "total": _json_float(value.total, f"{field}.total"),
        "arm": _json_float(value.arm, f"{field}.arm"),
        "gripper_state": _json_float(
            value.gripper_state, f"{field}.gripper_state"
        ),
        "gripper_timing": _json_float(
            value.gripper_timing, f"{field}.gripper_timing"
        ),
    }


def oracle_metrics_report_to_dict(report: OracleMetricsReport) -> dict[str, object]:
    """Convert an oracle report into finite, JSON-native primitives.

    Episode tuples and event ids become named JSON objects, action distances
    become finite float objects, per-query tuples become lists, and optional
    values remain JSON ``null``. Any NaN or infinity raises instead of relying
    on Python's non-standard JSON encoding.
    """

    if not isinstance(report, OracleMetricsReport):
        raise TypeError("report must be OracleMetricsReport")

    per_query: list[dict[str, object]] = []
    for position, result in enumerate(report.per_query):
        if not isinstance(result, OracleQueryResult):
            raise TypeError(f"report.per_query[{position}] must be OracleQueryResult")
        prefix = f"report.per_query[{position}]"
        per_query.append(
            {
                "query_index": _json_int(result.query_index, f"{prefix}.query_index"),
                "episode_key": _episode_key_to_dict(
                    result.episode_key, f"{prefix}.episode_key"
                ),
                "candidate_count": _json_int(
                    result.candidate_count,
                    f"{prefix}.candidate_count",
                    non_negative=True,
                ),
                "covered": bool(result.covered),
                "context_top1_event_id": _event_id_to_dict(
                    result.context_top1_event_id,
                    f"{prefix}.context_top1_event_id",
                ),
                "context_top1_score": _optional_json_float(
                    result.context_top1_score, f"{prefix}.context_top1_score"
                ),
                "context_top1_distance": _action_distance_to_dict(
                    result.context_top1_distance,
                    f"{prefix}.context_top1_distance",
                ),
                "oracle_event_id": _event_id_to_dict(
                    result.oracle_event_id, f"{prefix}.oracle_event_id"
                ),
                "oracle_rank": (
                    None
                    if result.oracle_rank is None
                    else _json_int(
                        result.oracle_rank,
                        f"{prefix}.oracle_rank",
                        non_negative=True,
                    )
                ),
                "oracle_distance": _action_distance_to_dict(
                    result.oracle_distance, f"{prefix}.oracle_distance"
                ),
                "recent_baseline_distance": _action_distance_to_dict(
                    result.recent_baseline_distance,
                    f"{prefix}.recent_baseline_distance",
                ),
                "relative_improvement_vs_context": _optional_json_float(
                    result.relative_improvement_vs_context,
                    f"{prefix}.relative_improvement_vs_context",
                ),
                "relative_improvement_vs_recent": _optional_json_float(
                    result.relative_improvement_vs_recent,
                    f"{prefix}.relative_improvement_vs_recent",
                ),
            }
        )

    payload: dict[str, object] = {
        "top_k": _json_int(report.top_k, "report.top_k", non_negative=True),
        "query_count": _json_int(
            report.query_count, "report.query_count", non_negative=True
        ),
        "covered_query_count": _json_int(
            report.covered_query_count,
            "report.covered_query_count",
            non_negative=True,
        ),
        "coverage": _json_float(report.coverage, "report.coverage"),
        "context_top1_mean": _action_distance_to_dict(
            report.context_top1_mean, "report.context_top1_mean"
        ),
        "oracle_topk_mean": _action_distance_to_dict(
            report.oracle_topk_mean, "report.oracle_topk_mean"
        ),
        "recent_baseline_mean": _action_distance_to_dict(
            report.recent_baseline_mean, "report.recent_baseline_mean"
        ),
        "recent_baseline_query_count": _json_int(
            report.recent_baseline_query_count,
            "report.recent_baseline_query_count",
            non_negative=True,
        ),
        "recent_comparable_query_count": _json_int(
            report.recent_comparable_query_count,
            "report.recent_comparable_query_count",
            non_negative=True,
        ),
        "relative_improvement_vs_context": _optional_json_float(
            report.relative_improvement_vs_context,
            "report.relative_improvement_vs_context",
        ),
        "relative_improvement_vs_recent": _optional_json_float(
            report.relative_improvement_vs_recent,
            "report.relative_improvement_vs_recent",
        ),
        "per_query": per_query,
    }
    # Defense in depth: this catches any future field accidentally added as a
    # NumPy scalar or other non-standard JSON value without changing the API.
    json.dumps(payload, allow_nan=False, sort_keys=True)
    return payload


def evaluate_oracle_retrieval(
    bank: EventBank,
    queries: Sequence[OracleQuery],
    config: ActionDistanceConfig,
    *,
    top_k: int = 32,
    action_payload: str = MODEL_SPACE_ACTION,
) -> OracleMetricsReport:
    """Evaluate context top-1 and GT-action oracle within context top-K.

    The candidate pool is always obtained through ``bank.search`` with the
    query episode identity plus its source- and feature-content hashes. There
    is no evaluator option to disable leave-entire-episode-out exclusion.
    """

    if not isinstance(bank, EventBank):
        raise TypeError("bank must be EventBank")
    if not isinstance(config, ActionDistanceConfig):
        raise TypeError("config must be ActionDistanceConfig")
    top_k = _positive_int(top_k, "top_k")
    if not isinstance(action_payload, str) or not action_payload:
        raise TypeError("action_payload must be a non-empty string")
    query_tuple = tuple(queries)
    if not query_tuple:
        raise OracleMetricError("queries must not be empty")
    if any(not isinstance(query, OracleQuery) for query in query_tuple):
        raise TypeError("every query must be OracleQuery")

    try:
        bank_actions = bank.payload(action_payload)
    except KeyError as exc:
        raise OracleMetricError(
            f"event bank is missing required action payload {action_payload!r}"
        ) from exc
    if bank_actions.dtype.kind != "f":
        raise TypeError(f"{action_payload} must have a floating dtype")
    if bank_actions.ndim != 3:
        raise OracleMetricError(
            f"{action_payload} must have shape [events, horizon, action_dim], "
            f"got {bank_actions.shape}"
        )
    if bank_actions.shape[1] <= 0 or bank_actions.shape[2] != config.action_dim:
        raise OracleMetricError(
            f"{action_payload} must have non-zero horizon and action_dim "
            f"{config.action_dim}, got {bank_actions.shape}"
        )
    if not np.isfinite(bank_actions).all():
        raise OracleMetricError(f"{action_payload} must contain only finite values")

    per_query: list[OracleQueryResult] = []
    context_distances: list[ActionDistance] = []
    oracle_distances: list[ActionDistance] = []
    recent_distances: list[ActionDistance] = []
    paired_recent_distances: list[ActionDistance] = []
    paired_oracle_distances: list[ActionDistance] = []

    expected_action_shape = (bank_actions.shape[1], config.action_dim)
    key_dim = bank.context_keys.shape[1]
    for query_index, query in enumerate(query_tuple):
        if query.query_key.shape != (key_dim,):
            raise OracleMetricError(
                f"query {query_index} key must have shape {(key_dim,)}, "
                f"got {query.query_key.shape}"
            )
        if query.gt_model_action.shape != expected_action_shape:
            raise OracleMetricError(
                f"query {query_index} gt_model_action must have shape "
                f"{expected_action_shape}, got {query.gt_model_action.shape}"
            )

        recent_distance = (
            None
            if query.recent_action is None
            else action_distance(query.recent_action, query.gt_model_action, config)
        )
        if recent_distance is not None:
            recent_distances.append(recent_distance)

        # Mandatory leakage barrier: every start frame in this episode is excluded.
        candidates = bank.search(
            query.query_key,
            top_k=top_k,
            exclude_episode=query.episode_key,
            exclude_source_episode_sha256=query.source_episode_sha256,
            exclude_feature_episode_sha256=query.feature_episode_sha256,
        )
        if not candidates:
            per_query.append(
                OracleQueryResult(
                    query_index=query_index,
                    episode_key=query.episode_key,
                    candidate_count=0,
                    context_top1_event_id=None,
                    context_top1_score=None,
                    context_top1_distance=None,
                    oracle_event_id=None,
                    oracle_rank=None,
                    oracle_distance=None,
                    recent_baseline_distance=recent_distance,
                    relative_improvement_vs_context=None,
                    relative_improvement_vs_recent=None,
                )
            )
            continue

        candidate_distances = [
            action_distance(
                bank_actions[candidate.index], query.gt_model_action, config
            )
            for candidate in candidates
        ]
        context_distance = candidate_distances[0]
        oracle_position = min(
            range(len(candidates)), key=lambda position: candidate_distances[position].total
        )
        oracle_distance = candidate_distances[oracle_position]
        context_distances.append(context_distance)
        oracle_distances.append(oracle_distance)
        if recent_distance is not None:
            paired_recent_distances.append(recent_distance)
            paired_oracle_distances.append(oracle_distance)

        per_query.append(
            OracleQueryResult(
                query_index=query_index,
                episode_key=query.episode_key,
                candidate_count=len(candidates),
                context_top1_event_id=candidates[0].event_id,
                context_top1_score=candidates[0].score,
                context_top1_distance=context_distance,
                oracle_event_id=candidates[oracle_position].event_id,
                oracle_rank=oracle_position + 1,
                oracle_distance=oracle_distance,
                recent_baseline_distance=recent_distance,
                relative_improvement_vs_context=_relative_improvement(
                    context_distance.total, oracle_distance.total
                ),
                relative_improvement_vs_recent=_relative_improvement(
                    None if recent_distance is None else recent_distance.total,
                    oracle_distance.total,
                ),
            )
        )

    context_mean = _mean_distance(context_distances)
    oracle_mean = _mean_distance(oracle_distances)
    recent_mean = _mean_distance(recent_distances)
    paired_recent_mean = _mean_distance(paired_recent_distances)
    paired_oracle_mean = _mean_distance(paired_oracle_distances)
    covered = len(context_distances)
    return OracleMetricsReport(
        top_k=top_k,
        query_count=len(query_tuple),
        covered_query_count=covered,
        coverage=covered / len(query_tuple),
        context_top1_mean=context_mean,
        oracle_topk_mean=oracle_mean,
        recent_baseline_mean=recent_mean,
        recent_baseline_query_count=len(recent_distances),
        recent_comparable_query_count=len(paired_recent_distances),
        relative_improvement_vs_context=_relative_improvement(
            None if context_mean is None else context_mean.total,
            None if oracle_mean is None else oracle_mean.total,
        ),
        relative_improvement_vs_recent=_relative_improvement(
            None if paired_recent_mean is None else paired_recent_mean.total,
            None if paired_oracle_mean is None else paired_oracle_mean.total,
        ),
        per_query=tuple(per_query),
    )


def evaluate_oracle_metrics(
    bank: EventBank,
    queries: Sequence[OracleQuery],
    config: ActionDistanceConfig,
    *,
    top_k: int = 32,
    action_payload: str = MODEL_SPACE_ACTION,
) -> OracleMetricsReport:
    """Alias with a report-oriented name for experiment scripts."""

    return evaluate_oracle_retrieval(
        bank,
        queries,
        config,
        top_k=top_k,
        action_payload=action_payload,
    )
