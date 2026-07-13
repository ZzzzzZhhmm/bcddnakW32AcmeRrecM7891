"""Deterministic, framework-free memory-corruption curriculum helpers.

The M3 training curriculum is keyed by a stable sample identity instead of a
process-global pseudo-random generator.  Consequently, changing dataloader
worker count or batch order cannot silently change which samples receive
normal memory, a dropped bank, an explicit null source, or a hard negative.

``drop`` and ``null`` deliberately have different semantics.  ``drop`` makes
the long-term bank unavailable before reranking, while ``null`` leaves the
candidate evidence visible but forces the source component to be Gaussian.
Together their default probability is the 25 percent no-memory portion of the
WARM curriculum; hard negatives occupy the remaining corrupted 25 percent.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Literal


CorruptionMode = Literal["normal", "drop", "null", "hard_negative"]
_MODES: tuple[CorruptionMode, ...] = (
    "normal",
    "drop",
    "null",
    "hard_negative",
)
_UINT64_DENOMINATOR = float(1 << 64)


class CorruptionCurriculumError(ValueError):
    """Raised when deterministic curriculum inputs violate their contract."""


@dataclass(frozen=True, slots=True)
class CorruptionWeights:
    """Closed four-way probability simplex for the M3 curriculum."""

    normal: float = 0.50
    drop: float = 0.125
    null: float = 0.125
    hard_negative: float = 0.25

    def __post_init__(self) -> None:
        values = tuple(getattr(self, mode) for mode in _MODES)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in values
        ):
            raise CorruptionCurriculumError(
                "corruption weights must be finite non-negative numbers"
            )
        if not math.isclose(
            sum(float(value) for value in values),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise CorruptionCurriculumError(
                "corruption weights must sum to exactly one within 1e-12"
            )


@dataclass(frozen=True, slots=True)
class CorruptionDecision:
    """One reproducible curriculum decision and its hard-negative entropy."""

    mode: CorruptionMode
    unit_draw: float
    hard_negative_token: int

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise CorruptionCurriculumError(
                f"unsupported corruption mode {self.mode!r}"
            )
        if (
            isinstance(self.unit_draw, bool)
            or not isinstance(self.unit_draw, (int, float))
            or not math.isfinite(float(self.unit_draw))
            or not 0.0 <= float(self.unit_draw) < 1.0
        ):
            raise CorruptionCurriculumError("unit_draw must lie in [0, 1)")
        if (
            isinstance(self.hard_negative_token, bool)
            or not isinstance(self.hard_negative_token, int)
            or not 0 <= self.hard_negative_token < (1 << 64)
        ):
            raise CorruptionCurriculumError(
                "hard_negative_token must be an unsigned 64-bit integer"
            )


def _require_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < (1 << 64):
        raise CorruptionCurriculumError("seed must lie in [0, 2**64)")
    return seed


def _require_sample_identity(sample_identity: object) -> str:
    if not isinstance(sample_identity, str):
        raise TypeError("sample_identity must be a string")
    if not sample_identity or sample_identity.strip() != sample_identity:
        raise CorruptionCurriculumError(
            "sample_identity must be non-empty with no surrounding whitespace"
        )
    return sample_identity


def derive_corruption_decision(
    seed: int,
    sample_identity: str,
    *,
    weights: CorruptionWeights = CorruptionWeights(),
) -> CorruptionDecision:
    """Derive one order-independent decision from SHA-256.

    ``sample_identity`` should be the immutable QueryId (or another identity
    containing split, episode, frame, and view).  No Python ``hash()`` or
    process-global RNG state is involved.
    """

    checked_seed = _require_seed(seed)
    identity = _require_sample_identity(sample_identity)
    if not isinstance(weights, CorruptionWeights):
        raise TypeError("weights must be CorruptionWeights")
    payload = (
        b"warm-m3-corruption-v1\x00"
        + checked_seed.to_bytes(8, byteorder="big", signed=False)
        + b"\x00"
        + identity.encode("utf-8")
    )
    digest = hashlib.sha256(payload).digest()
    unit_draw = int.from_bytes(digest[:8], "big") / _UINT64_DENOMINATOR
    hard_negative_token = int.from_bytes(digest[8:16], "big")

    cumulative = 0.0
    mode: CorruptionMode = "hard_negative"
    for candidate_mode in _MODES[:-1]:
        cumulative += float(getattr(weights, candidate_mode))
        if unit_draw < cumulative:
            mode = candidate_mode
            break
    return CorruptionDecision(
        mode=mode,
        unit_draw=unit_draw,
        hard_negative_token=hard_negative_token,
    )


def derive_corruption_batch(
    seed: int,
    sample_identities: Iterable[str],
    *,
    weights: CorruptionWeights = CorruptionWeights(),
) -> tuple[CorruptionDecision, ...]:
    """Derive a batch without making the result depend on iteration history."""

    checked_seed = _require_seed(seed)
    identities = tuple(
        _require_sample_identity(identity) for identity in sample_identities
    )
    return tuple(
        derive_corruption_decision(
            checked_seed,
            identity,
            weights=weights,
        )
        for identity in identities
    )


def select_hard_negative_index(
    decision: CorruptionDecision,
    eligible_indices: Iterable[int],
) -> int:
    """Choose one eligible hard-negative slot without additional RNG state."""

    if not isinstance(decision, CorruptionDecision):
        raise TypeError("decision must be CorruptionDecision")
    if decision.mode != "hard_negative":
        raise CorruptionCurriculumError(
            "hard-negative selection requires a hard_negative decision"
        )
    indices = tuple(eligible_indices)
    if not indices:
        raise CorruptionCurriculumError(
            "hard-negative selection requires at least one eligible index"
        )
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise CorruptionCurriculumError(
            "eligible hard-negative indices must be non-negative integers"
        )
    if len(set(indices)) != len(indices):
        raise CorruptionCurriculumError(
            "eligible hard-negative indices must be unique"
        )
    return indices[decision.hard_negative_token % len(indices)]


__all__ = [
    "CorruptionCurriculumError",
    "CorruptionDecision",
    "CorruptionMode",
    "CorruptionWeights",
    "derive_corruption_batch",
    "derive_corruption_decision",
    "select_hard_negative_index",
]
