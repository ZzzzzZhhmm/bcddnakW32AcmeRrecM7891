"""Closed configuration for the complete consequence-aligned WARM model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    video_adapter_rank: int = 1
    video_adapter_layers: tuple[int, ...] = ()
    video_adapter_scale: float = 1.0
    source_sigma_min: float = 0.2
    residual_rho: float = 0.25
    consequence_weight: float = 1.0
    support_weight: float = 0.0
    magnitude_weight: float = 0.25
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
        for field in (
            "source_sigma_min",
            "residual_rho",
            "utility_temperature",
            "gate_temperature",
            "video_adapter_scale",
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
        ):
            object.__setattr__(
                self,
                field,
                _finite(getattr(self, field), field, non_negative=True),
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
