"""Closed configuration for the complete consequence-aligned WARM model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping


class WarmRetrospectionConfigError(ValueError):
    """Raised before allocating an ambiguous full-WARM architecture."""


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise WarmRetrospectionConfigError(f"{field} must be a positive integer")
    return int(value)


def _finite(
    value: object,
    field: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WarmRetrospectionConfigError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise WarmRetrospectionConfigError(f"{field} must be finite")
    if positive and result <= 0:
        raise WarmRetrospectionConfigError(f"{field} must be positive")
    if non_negative and result < 0:
        raise WarmRetrospectionConfigError(f"{field} must be non-negative")
    return result


@dataclass(frozen=True, slots=True)
class WarmRetrospectionConfig:
    """Dimensions and loss weights for one reproducible WARM architecture."""

    context_dim: int
    semantic_dim: int
    video_dim: int
    text_dim: int
    proprio_dim: int
    action_dim: int
    action_horizon: int
    timing_dim: int = 4
    gist_dim: int = 1024
    event_model_dim: int = 512
    bridge_heads: int = 8
    gist_heads: int = 8
    event_heads: int = 8
    reranker_hidden_dim: int = 128
    gate_hidden_dim: int = 16
    episode_action_chunk_size: int = 1
    # ``none`` preserves the original LIBERO model-space action payload.
    # ``start_proprio_delta`` is the closed RoboTwin/RMBench contract: stored
    # absolute joint commands are translated from the event's factual start
    # state to the current factual start state.  Gripper command channels are
    # deliberately excluded from the translation.
    canonical_action_mode: str = "none"
    canonical_gripper_dims: tuple[int, ...] = ()
    video_adapter_rank: int = 1
    video_adapter_layers: tuple[int, ...] = ()
    video_adapter_scale: float = 1.0
    source_sigma_min: float = 0.2
    residual_rho: float = 0.25
    consequence_weight: float = 1.0
    support_weight: float = 0.0
    magnitude_weight: float = 0.25
    # Candidate-only reranker logits are shift invariant.  Source acceptance
    # therefore uses calibrated distribution statistics rather than an
    # absolute score: selected probability, top-1/top-2 probability margin,
    # and normalized entropy.
    selection_temperature: float = 1.0
    minimum_candidate_probability: float = 0.06
    maximum_candidate_entropy: float = 0.94
    inference_source_gate_threshold: float = 0.15
    # Repeated closed-loop action summaries are factual evidence that the
    # current memory thread is not making progress.  They attenuate the source
    # continuously and trigger a hard Gaussian fallback at the configured
    # limit.
    stagnation_decay: float = 3.0
    stagnation_hard_threshold: float = 0.75
    # Online-only temporal coherence prior.  It favors monotonic continuation
    # inside one retrieved demonstration without forcing a candidate when the
    # calibrated source gate rejects memory.
    thread_score_weight: float = 0.75
    thread_switch_penalty: float = 0.25
    # Once one retrieved event phase has supplied a complete action horizon,
    # reusing that exact/older phase is no longer temporal continuation.  Any
    # strictly later factual event remains eligible; dense H32 banks normally
    # advance by only four frames. ``thread_backtrack_tolerance`` is retained
    # in the serialized config for checkpoint-contract compatibility but is
    # not allowed to suppress positive successor deltas.
    thread_reuse_penalty: float = 1.0
    thread_backtrack_tolerance: int = 16
    thread_forward_window: int = 256
    thread_max_null_steps: int = 3
    utility_effect_weight: float = 1.0
    utility_temperature: float = 1.0
    gate_effect_weight: float = 1.0
    gate_temperature: float = 1.0
    loss_retrieval: float = 0.10
    loss_bridge: float = 0.05
    loss_gist: float = 0.05
    loss_effect: float = 0.05
    loss_gate: float = 0.05
    loss_adaptation: float = 0.05
    corruption_normal_weight: float = 0.50
    corruption_drop_weight: float = 0.125
    corruption_null_weight: float = 0.125
    corruption_hard_negative_weight: float = 0.25
    corruption_seed: int = 3407

    @property
    def episode_action_summary_dim(self) -> int:
        """Mean displacement, final displacement, terminal command, 4 scalars."""

        return 3 * self.action_dim + 4

    def __post_init__(self) -> None:
        integer_fields = (
            "context_dim",
            "semantic_dim",
            "video_dim",
            "text_dim",
            "proprio_dim",
            "action_dim",
            "action_horizon",
            "timing_dim",
            "gist_dim",
            "event_model_dim",
            "bridge_heads",
            "gist_heads",
            "event_heads",
            "reranker_hidden_dim",
            "gate_hidden_dim",
            "episode_action_chunk_size",
            "video_adapter_rank",
            "thread_backtrack_tolerance",
            "thread_forward_window",
            "thread_max_null_steps",
        )
        for field in integer_fields:
            object.__setattr__(
                self, field, _positive_int(getattr(self, field), field)
            )
        for width, heads, field in (
            (self.semantic_dim, self.bridge_heads, "semantic_dim"),
            (self.gist_dim, self.gist_heads, "gist_dim"),
            (self.event_model_dim, self.event_heads, "event_model_dim"),
        ):
            if width % heads:
                raise WarmRetrospectionConfigError(
                    f"{field}={width} must be divisible by heads={heads}"
                )
        if self.video_adapter_rank >= self.video_dim:
            raise WarmRetrospectionConfigError(
                "video_adapter_rank must be smaller than video_dim"
            )
        raw_layers = self.video_adapter_layers
        if not isinstance(raw_layers, (list, tuple)):
            raise WarmRetrospectionConfigError(
                "video_adapter_layers must be a list or tuple"
            )
        layers: list[int] = []
        for value in raw_layers:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise WarmRetrospectionConfigError(
                    "video_adapter_layers must contain non-negative integers"
                )
            layers.append(int(value))
        if len(set(layers)) != len(layers):
            raise WarmRetrospectionConfigError(
                "video_adapter_layers must not contain duplicates"
            )
        object.__setattr__(self, "video_adapter_layers", tuple(layers))
        if self.episode_action_chunk_size > self.action_horizon:
            raise WarmRetrospectionConfigError(
                "episode_action_chunk_size cannot exceed action_horizon"
            )
        if (
            not isinstance(self.canonical_action_mode, str)
            or self.canonical_action_mode
            not in {"none", "start_proprio_delta"}
        ):
            raise WarmRetrospectionConfigError(
                "canonical_action_mode must be 'none' or 'start_proprio_delta'"
            )
        raw_gripper_dims = self.canonical_gripper_dims
        if not isinstance(raw_gripper_dims, (list, tuple)):
            raise WarmRetrospectionConfigError(
                "canonical_gripper_dims must be a list or tuple"
            )
        gripper_dims: list[int] = []
        for value in raw_gripper_dims:
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < self.action_dim
            ):
                raise WarmRetrospectionConfigError(
                    "canonical_gripper_dims must contain valid action indices"
                )
            gripper_dims.append(int(value))
        if len(set(gripper_dims)) != len(gripper_dims):
            raise WarmRetrospectionConfigError(
                "canonical_gripper_dims must not contain duplicates"
            )
        object.__setattr__(
            self, "canonical_gripper_dims", tuple(sorted(gripper_dims))
        )
        if self.canonical_action_mode == "none" and gripper_dims:
            raise WarmRetrospectionConfigError(
                "canonical_gripper_dims must be empty when canonical_action_mode='none'"
            )
        if gripper_dims and self.timing_dim != 4 * len(gripper_dims):
            raise WarmRetrospectionConfigError(
                "timing_dim must provide four phase/validity facts per "
                "canonical gripper dimension"
            )
        if (
            self.canonical_action_mode == "start_proprio_delta"
            and self.proprio_dim != self.action_dim
        ):
            raise WarmRetrospectionConfigError(
                "start_proprio_delta requires proprio_dim == action_dim"
            )
        for field in (
            "source_sigma_min",
            "residual_rho",
            "utility_temperature",
            "gate_temperature",
            "video_adapter_scale",
            "selection_temperature",
        ):
            object.__setattr__(
                self, field, _finite(getattr(self, field), field, positive=True)
            )
        for field in (
            "consequence_weight",
            "support_weight",
            "magnitude_weight",
            "utility_effect_weight",
            "gate_effect_weight",
            "loss_retrieval",
            "loss_bridge",
            "loss_gist",
            "loss_effect",
            "loss_gate",
            "loss_adaptation",
            "minimum_candidate_probability",
            "maximum_candidate_entropy",
            "inference_source_gate_threshold",
            "stagnation_decay",
            "stagnation_hard_threshold",
            "thread_score_weight",
            "thread_switch_penalty",
            "thread_reuse_penalty",
            "corruption_normal_weight",
            "corruption_drop_weight",
            "corruption_null_weight",
            "corruption_hard_negative_weight",
        ):
            object.__setattr__(
                self,
                field,
                _finite(getattr(self, field), field, non_negative=True),
            )
        for field in (
            "minimum_candidate_probability",
            "maximum_candidate_entropy",
            "inference_source_gate_threshold",
            "stagnation_hard_threshold",
        ):
            if getattr(self, field) > 1.0:
                raise WarmRetrospectionConfigError(
                    f"{field} must lie in [0,1]"
                )
        if self.maximum_candidate_entropy <= 0.0:
            raise WarmRetrospectionConfigError(
                "maximum_candidate_entropy must lie in (0,1]"
            )
        corruption_total = sum(
            getattr(self, field)
            for field in (
                "corruption_normal_weight",
                "corruption_drop_weight",
                "corruption_null_weight",
                "corruption_hard_negative_weight",
            )
        )
        if not math.isclose(corruption_total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise WarmRetrospectionConfigError(
                "corruption weights must sum to one within 1e-12"
            )
        if (
            isinstance(self.corruption_seed, bool)
            or not isinstance(self.corruption_seed, int)
            or not 0 <= self.corruption_seed < (1 << 64)
        ):
            raise WarmRetrospectionConfigError(
                "corruption_seed must be an unsigned 64-bit integer"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-domain representation of this config.

        Architecture fields are normalized to immutable tuples in memory,
        while JSON necessarily serializes them as arrays.  Checkpoint and
        trainer-state contracts must compare parsed semantics rather than raw
        tuple/list container types.
        """

        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        value = json.loads(encoded)
        if not isinstance(value, dict):  # pragma: no cover - asdict is a dict
            raise WarmRetrospectionConfigError(
                "retrospection config did not encode a JSON object"
            )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WarmRetrospectionConfig":
        if not isinstance(value, Mapping):
            raise TypeError("retrospection config must be a mapping")
        expected = set(cls.__dataclass_fields__)
        extra = set(value) - expected
        if extra:
            raise WarmRetrospectionConfigError(
                f"unknown retrospection config fields: {sorted(extra)}"
            )
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise WarmRetrospectionConfigError(
                "retrospection config is missing required fields"
            ) from exc


__all__ = [
    "WarmRetrospectionConfig",
    "WarmRetrospectionConfigError",
]
