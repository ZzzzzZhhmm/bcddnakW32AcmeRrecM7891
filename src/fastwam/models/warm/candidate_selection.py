"""NumPy reference semantics for WARM candidate selection and source sampling.

The required effect is deliberately supplied as one candidate-independent vector.
Memory candidates only provide *observed* effects.  Component ``0`` is always the
explicit null component; memory candidate ``j`` is represented by component
``j + 1``.

This module is intentionally independent of Torch and of the FastWAM runtime.  It
serves as an executable mathematical contract for later model implementations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
NULL_COMPONENT = 0


@dataclass(frozen=True)
class EffectScoreBreakdown:
    """Per-candidate observed-effect compatibility terms."""

    cosine: FloatArray
    log_magnitude_gap: FloatArray
    combined: FloatArray


@dataclass(frozen=True)
class CandidateMixture:
    """Logits and probabilities with the null component in position zero."""

    memory_logits: FloatArray
    logits: FloatArray
    probabilities: FloatArray
    effect_scores: EffectScoreBreakdown

    @property
    def num_memory_candidates(self) -> int:
        return int(self.memory_logits.shape[0])


@dataclass(frozen=True)
class MemorySource:
    """Parameters for a memory component source ``mu + sigma * epsilon``."""

    mu: ArrayLike
    sigma: ArrayLike


MemorySourceGetter = Callable[[int], MemorySource]


def _float_array(name: str, value: ArrayLike, *, ndim: int) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be convertible to a float array") from error
    if array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_scalar(name: str, value: Any) -> float:
    try:
        scalar = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if scalar.ndim != 0 or not np.isfinite(scalar.item()):
        raise ValueError(f"{name} must be a finite scalar")
    return float(scalar.item())


def _nonnegative_scalar(name: str, value: Any) -> float:
    scalar = _finite_scalar(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _positive_scalar(name: str, value: Any) -> float:
    scalar = _finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _action_shape(action_shape: Sequence[int]) -> tuple[int, ...]:
    if isinstance(action_shape, (str, bytes)):
        raise ValueError("action_shape must be a non-empty sequence of integers")
    try:
        shape = tuple(action_shape)
    except TypeError as error:
        raise ValueError("action_shape must be a non-empty sequence of integers") from error
    if not shape:
        raise ValueError("action_shape must be non-empty")
    for dimension in shape:
        if isinstance(dimension, (bool, np.bool_)) or not isinstance(
            dimension, (int, np.integer)
        ):
            raise ValueError("action_shape dimensions must be integers")
        if int(dimension) <= 0:
            raise ValueError("action_shape dimensions must be positive")
    return tuple(int(dimension) for dimension in shape)


def _component_index(component: Any) -> int:
    if isinstance(component, (bool, np.bool_)) or not isinstance(
        component, (int, np.integer)
    ):
        raise ValueError("component must be a non-negative integer")
    component = int(component)
    if component < 0:
        raise ValueError("component must be a non-negative integer")
    return component


def _rng_method(rng: Any, name: str) -> Callable[..., Any]:
    method = getattr(rng, name, None)
    if not callable(method):
        raise TypeError(f"rng must provide a callable {name} method")
    return method


def _standard_normal(rng: Any, shape: tuple[int, ...]) -> FloatArray:
    draw = _rng_method(rng, "standard_normal")(size=shape)
    noise = _float_array("rng.standard_normal output", draw, ndim=len(shape))
    if noise.shape != shape:
        raise ValueError(
            f"rng.standard_normal returned shape {noise.shape}, expected {shape}"
        )
    return noise


def score_observed_effects(
    required_effect: ArrayLike,
    observed_effects: ArrayLike,
    *,
    log_magnitude_weight: float = 1.0,
    eps: float = 1e-8,
) -> EffectScoreBreakdown:
    """Score factual candidate effects against one independent required effect.

    The score is cosine compatibility minus the absolute log-magnitude gap.  A
    zero-norm effect has cosine score zero; its magnitude remains well-defined via
    ``eps``.  Candidate effects are never pooled when constructing the required
    effect, so corrupting one candidate cannot alter another candidate's score.
    """

    required = _float_array("required_effect", required_effect, ndim=1)
    observed = _float_array("observed_effects", observed_effects, ndim=2)
    if required.shape[0] == 0:
        raise ValueError("required_effect must have at least one feature")
    if observed.shape[1] != required.shape[0]:
        raise ValueError(
            "observed_effects feature dimension must match required_effect: "
            f"{observed.shape[1]} != {required.shape[0]}"
        )

    magnitude_weight = _nonnegative_scalar(
        "log_magnitude_weight", log_magnitude_weight
    )
    eps = _positive_scalar("eps", eps)

    required_norm = float(np.linalg.norm(required))
    observed_norms = np.linalg.norm(observed, axis=1)
    denominators = required_norm * observed_norms
    dots = observed @ required
    cosine = np.divide(
        dots,
        denominators,
        out=np.zeros_like(dots, dtype=np.float64),
        where=denominators > eps,
    )
    cosine = np.clip(cosine, -1.0, 1.0)

    required_log_magnitude = np.log(required_norm + eps)
    observed_log_magnitudes = np.log(observed_norms + eps)
    log_magnitude_gap = np.abs(
        observed_log_magnitudes - required_log_magnitude
    )
    combined = cosine - magnitude_weight * log_magnitude_gap

    if not (
        np.all(np.isfinite(cosine))
        and np.all(np.isfinite(log_magnitude_gap))
        and np.all(np.isfinite(combined))
    ):
        raise ValueError("effect score computation produced non-finite values")

    return EffectScoreBreakdown(
        cosine=cosine,
        log_magnitude_gap=log_magnitude_gap,
        combined=combined,
    )


def softmax_probabilities(logits: ArrayLike) -> FloatArray:
    """Compute a stable softmax over a finite, non-empty logit vector."""

    logits_array = _float_array("logits", logits, ndim=1)
    if logits_array.size == 0:
        raise ValueError("logits must contain at least one component")
    shifted = logits_array - np.max(logits_array)
    exponentials = np.exp(shifted)
    normalizer = float(np.sum(exponentials))
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise ValueError("softmax normalization must be finite and positive")
    probabilities = exponentials / normalizer
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("softmax produced non-finite probabilities")
    return probabilities


def build_candidate_mixture(
    required_effect: ArrayLike,
    observed_effects: ArrayLike,
    context_scores: ArrayLike,
    *,
    null_logit: float,
    context_weight: float = 1.0,
    consequence_weight: float = 1.0,
    log_magnitude_weight: float = 1.0,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> CandidateMixture:
    """Build an explicit ``[null, memory_0, ..., memory_K]`` mixture.

    Context and observed-effect compatibility are the only memory-candidate
    terms.  ``null_logit`` is supplied independently and remains available even
    when there are no memory candidates.
    """

    effect_scores = score_observed_effects(
        required_effect,
        observed_effects,
        log_magnitude_weight=log_magnitude_weight,
        eps=eps,
    )
    context = _float_array("context_scores", context_scores, ndim=1)
    if context.shape != effect_scores.combined.shape:
        raise ValueError(
            "context_scores must have one value per observed effect: "
            f"{context.shape} != {effect_scores.combined.shape}"
        )

    null_logit = _finite_scalar("null_logit", null_logit)
    context_weight = _nonnegative_scalar("context_weight", context_weight)
    consequence_weight = _nonnegative_scalar(
        "consequence_weight", consequence_weight
    )
    temperature = _positive_scalar("temperature", temperature)

    with np.errstate(over="raise", invalid="raise"):
        try:
            memory_logits = (
                context_weight * context
                + consequence_weight * effect_scores.combined
            )
        except FloatingPointError as error:
            raise ValueError("candidate logit computation overflowed") from error
    if not np.all(np.isfinite(memory_logits)):
        raise ValueError("candidate logit computation produced non-finite values")

    logits = np.concatenate(
        [np.asarray([null_logit], dtype=np.float64), memory_logits]
    )
    probabilities = softmax_probabilities(logits / temperature)
    return CandidateMixture(
        memory_logits=memory_logits,
        logits=logits,
        probabilities=probabilities,
        effect_scores=effect_scores,
    )


def _validated_probabilities(probabilities: ArrayLike) -> FloatArray:
    array = _float_array("probabilities", probabilities, ndim=1)
    if array.size == 0:
        raise ValueError("probabilities must contain at least the null component")
    if np.any(array < 0.0):
        raise ValueError("probabilities must be non-negative")
    total = float(np.sum(array))
    if not np.isclose(total, 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError(f"probabilities must sum to one, got {total}")
    return array / total


def sample_component(probabilities: ArrayLike, rng: Any) -> int:
    """Sample a categorical component using the caller-provided RNG."""

    probabilities_array = _validated_probabilities(probabilities)
    choice = _rng_method(rng, "choice")
    component = choice(probabilities_array.size, p=probabilities_array)
    component = _component_index(component)
    if component >= probabilities_array.size:
        raise ValueError("rng.choice returned an out-of-range component")
    return component


def sample_null_gaussian_source(
    action_shape: Sequence[int],
    rng: Any,
    *,
    sigma: float = 1.0,
) -> FloatArray:
    """Sample the explicit null source ``sigma * epsilon``."""

    shape = _action_shape(action_shape)
    sigma = _positive_scalar("null sigma", sigma)
    return sigma * _standard_normal(rng, shape)


def sample_memory_source(
    memory_source: MemorySource,
    rng: Any,
    *,
    expected_shape: Sequence[int] | None = None,
) -> FloatArray:
    """Sample a memory source exactly as ``mu + sigma * epsilon``."""

    if not isinstance(memory_source, MemorySource):
        raise TypeError("memory_source getter must return MemorySource")
    mu = np.asarray(memory_source.mu, dtype=np.float64)
    if mu.ndim == 0 or mu.size == 0:
        raise ValueError("memory source mu must be a non-empty action tensor")
    if not np.all(np.isfinite(mu)):
        raise ValueError("memory source mu must contain only finite values")

    if expected_shape is not None:
        shape = _action_shape(expected_shape)
        if mu.shape != shape:
            raise ValueError(
                f"memory source mu has shape {mu.shape}, expected {shape}"
            )
    else:
        shape = mu.shape

    try:
        sigma = np.asarray(memory_source.sigma, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("memory source sigma must be numeric") from error
    if sigma.ndim not in (0, mu.ndim):
        raise ValueError("memory source sigma must be scalar or match mu shape")
    if sigma.ndim > 0 and sigma.shape != mu.shape:
        raise ValueError(
            f"memory source sigma has shape {sigma.shape}, expected {mu.shape}"
        )
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("memory source sigma must be finite and strictly positive")

    epsilon = _standard_normal(rng, shape)
    source = mu + sigma * epsilon
    if not np.all(np.isfinite(source)):
        raise ValueError("memory source sampling produced non-finite values")
    return source


def sample_source_for_component(
    component: int,
    action_shape: Sequence[int],
    rng: Any,
    memory_source_getter: MemorySourceGetter | None,
    *,
    null_sigma: float = 1.0,
) -> FloatArray:
    """Sample a source for an already-selected component.

    The null branch returns before inspecting or calling
    ``memory_source_getter``.  This is the contract that prevents unnecessary or
    unsafe memory payload reads during fallback.
    """

    component = _component_index(component)
    shape = _action_shape(action_shape)
    if component == NULL_COMPONENT:
        return sample_null_gaussian_source(shape, rng, sigma=null_sigma)

    if not callable(memory_source_getter):
        raise TypeError("memory_source_getter must be callable for memory components")
    candidate_index = component - 1
    memory_source = memory_source_getter(candidate_index)
    return sample_memory_source(memory_source, rng, expected_shape=shape)


def sample_source_from_mixture(
    probabilities: ArrayLike,
    action_shape: Sequence[int],
    rng: Any,
    memory_source_getter: MemorySourceGetter | None,
    *,
    null_sigma: float = 1.0,
) -> tuple[int, FloatArray]:
    """Categorically select a component and lazily sample its action source."""

    probabilities_array = _validated_probabilities(probabilities)
    component = sample_component(probabilities_array, rng)
    source = sample_source_for_component(
        component,
        action_shape,
        rng,
        memory_source_getter,
        null_sigma=null_sigma,
    )
    return component, source


__all__ = [
    "NULL_COMPONENT",
    "CandidateMixture",
    "EffectScoreBreakdown",
    "MemorySource",
    "build_candidate_mixture",
    "sample_component",
    "sample_memory_source",
    "sample_null_gaussian_source",
    "sample_source_for_component",
    "sample_source_from_mixture",
    "score_observed_effects",
    "softmax_probabilities",
]
