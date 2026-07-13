"""Pure-Python configuration contracts for WARM's lightweight adapters.

The learned modules live in separate Torch files so this module remains
importable on the local, CPU-only development machine.  The contracts reject
ambiguous dimensions before a model is allocated; server code can therefore
serialize these dataclasses as ordinary dictionaries without importing Torch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


class AdapterConfigError(ValueError):
    """Raised when a learned-adapter architecture is not well defined."""


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdapterConfigError(f"{field} must be a positive integer")
    return int(value)


def _finite_positive(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterConfigError(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise AdapterConfigError(f"{field} must be a finite positive number")
    return result


def _dropout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterConfigError("dropout must be a finite number in [0, 1)")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise AdapterConfigError("dropout must be a finite number in [0, 1)")
    return result


def _attention_width(width: int, heads: int, *, field: str) -> None:
    if width % heads:
        raise AdapterConfigError(
            f"{field}={width} must be divisible by num_heads={heads}"
        )


@dataclass(frozen=True, slots=True)
class SemanticBridgeConfig:
    """Dimensions for the two-layer world tap and four-token bridge."""

    early_dim: int
    late_dim: int
    semantic_dim: int = 256
    query_count: int = 4
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    cosine_eps: float = 1.0e-6

    def __post_init__(self) -> None:
        for field in (
            "early_dim",
            "late_dim",
            "semantic_dim",
            "query_count",
            "num_heads",
        ):
            object.__setattr__(
                self, field, _positive_int(getattr(self, field), field)
            )
        # The paper architecture deliberately exposes exactly four compact
        # spatial semantic tokens; changing this is a different method.
        if self.query_count != 4:
            raise AdapterConfigError("SemanticQueryBridge requires query_count=4")
        _attention_width(
            self.semantic_dim, self.num_heads, field="semantic_dim"
        )
        object.__setattr__(
            self, "mlp_ratio", _finite_positive(self.mlp_ratio, "mlp_ratio")
        )
        object.__setattr__(self, "dropout", _dropout(self.dropout))
        object.__setattr__(
            self,
            "cosine_eps",
            _finite_positive(self.cosine_eps, "cosine_eps"),
        )


@dataclass(frozen=True, slots=True)
class RetrospectiveGistConfig:
    """Dimensions for the exact two-block, eight-query gist adapter."""

    world_dim: int
    semantic_dim: int
    episode_dim: int
    event_dim: int
    text_dim: int
    model_dim: int = 1024
    future_target_dim: int = 256
    query_count: int = 8
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    cosine_eps: float = 1.0e-6

    def __post_init__(self) -> None:
        for field in (
            "world_dim",
            "semantic_dim",
            "episode_dim",
            "event_dim",
            "text_dim",
            "model_dim",
            "future_target_dim",
            "query_count",
            "num_heads",
        ):
            object.__setattr__(
                self, field, _positive_int(getattr(self, field), field)
            )
        if self.query_count != 8:
            raise AdapterConfigError(
                "RetrospectiveGistAdapter requires query_count=8"
            )
        _attention_width(self.model_dim, self.num_heads, field="model_dim")
        object.__setattr__(
            self, "mlp_ratio", _finite_positive(self.mlp_ratio, "mlp_ratio")
        )
        object.__setattr__(self, "dropout", _dropout(self.dropout))
        object.__setattr__(
            self,
            "cosine_eps",
            _finite_positive(self.cosine_eps, "cosine_eps"),
        )


@dataclass(frozen=True, slots=True)
class RetrospectiveEventAdapterConfig:
    """Dimensions and bounded-deformation contract for one event adapter."""

    action_dim: int
    action_horizon: int
    timing_dim: int
    world_dim: int
    gist_dim: int
    proprio_dim: int
    text_dim: int
    event_dim: int
    model_dim: int = 1024
    effect_dim: int = 256
    context_query_count: int = 4
    num_heads: int = 8
    rho: float = 0.25
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    cosine_eps: float = 1.0e-6

    def __post_init__(self) -> None:
        for field in (
            "action_dim",
            "action_horizon",
            "timing_dim",
            "world_dim",
            "gist_dim",
            "proprio_dim",
            "text_dim",
            "event_dim",
            "model_dim",
            "effect_dim",
            "context_query_count",
            "num_heads",
        ):
            object.__setattr__(
                self, field, _positive_int(getattr(self, field), field)
            )
        if self.context_query_count != 4:
            raise AdapterConfigError(
                "RetrospectiveEventAdapter requires context_query_count=4"
            )
        _attention_width(self.model_dim, self.num_heads, field="model_dim")
        object.__setattr__(self, "rho", _finite_positive(self.rho, "rho"))
        object.__setattr__(
            self, "mlp_ratio", _finite_positive(self.mlp_ratio, "mlp_ratio")
        )
        object.__setattr__(self, "dropout", _dropout(self.dropout))
        object.__setattr__(
            self,
            "cosine_eps",
            _finite_positive(self.cosine_eps, "cosine_eps"),
        )


__all__ = [
    "AdapterConfigError",
    "RetrospectiveEventAdapterConfig",
    "RetrospectiveGistConfig",
    "SemanticBridgeConfig",
]
