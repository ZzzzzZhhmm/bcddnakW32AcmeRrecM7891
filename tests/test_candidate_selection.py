from __future__ import annotations

import numpy as np
import pytest

from fastwam.models.warm.candidate_selection import (
    NULL_COMPONENT,
    MemorySource,
    build_candidate_mixture,
    sample_component,
    sample_memory_source,
    sample_source_for_component,
    sample_source_from_mixture,
    score_observed_effects,
)


def test_corrupting_one_observed_effect_does_not_change_other_scores() -> None:
    required = np.array([1.0, 0.0])
    clean = np.array([[1.0, 0.0], [1.0, 0.0]])
    corrupted = clean.copy()
    corrupted[1] = [-1.0, 0.0]

    clean_scores = score_observed_effects(required, clean)
    corrupted_scores = score_observed_effects(required, corrupted)

    np.testing.assert_allclose(
        corrupted_scores.combined[0], clean_scores.combined[0]
    )
    assert corrupted_scores.combined[0] > corrupted_scores.combined[1]
    assert corrupted_scores.cosine[1] == pytest.approx(-1.0)


def test_context_effect_and_explicit_null_form_one_mixture() -> None:
    mixture = build_candidate_mixture(
        required_effect=[1.0, 0.0],
        observed_effects=[[1.0, 0.0], [-1.0, 0.0]],
        context_scores=[0.0, 0.0],
        null_logit=0.25,
    )

    assert mixture.logits.shape == (3,)
    assert mixture.probabilities.shape == (3,)
    assert mixture.logits[NULL_COMPONENT] == pytest.approx(0.25)
    assert np.sum(mixture.probabilities) == pytest.approx(1.0)
    assert mixture.probabilities[1] > mixture.probabilities[0]
    assert mixture.probabilities[1] > mixture.probabilities[2]


def test_empty_memory_candidates_reduce_to_certain_null() -> None:
    mixture = build_candidate_mixture(
        required_effect=[1.0, 0.0],
        observed_effects=np.empty((0, 2)),
        context_scores=np.empty((0,)),
        null_logit=-7.0,
    )

    np.testing.assert_array_equal(mixture.logits, [-7.0])
    np.testing.assert_array_equal(mixture.probabilities, [1.0])


def test_multimodal_memory_candidates_remain_distinct_components() -> None:
    mixture = build_candidate_mixture(
        required_effect=[1.0, 0.0],
        observed_effects=[[1.0, 0.0], [1.0, 0.0]],
        context_scores=[0.0, 0.0],
        null_logit=-10.0,
    )

    assert mixture.probabilities[1] == pytest.approx(mixture.probabilities[2])
    assert mixture.probabilities[1] > 0.49
    assert mixture.probabilities[2] > 0.49

    rng = np.random.default_rng(7)
    draws = np.array(
        [sample_component(mixture.probabilities, rng) for _ in range(400)]
    )
    assert np.count_nonzero(draws == 1) > 100
    assert np.count_nonzero(draws == 2) > 100


def test_null_selection_does_not_read_memory_payload() -> None:
    calls: list[int] = []

    def forbidden_getter(index: int) -> MemorySource:
        calls.append(index)
        raise AssertionError("null source must not read memory")

    component, source = sample_source_from_mixture(
        probabilities=[1.0, 0.0, 0.0],
        action_shape=(4, 3),
        rng=np.random.default_rng(11),
        memory_source_getter=forbidden_getter,
    )

    assert component == NULL_COMPONENT
    assert source.shape == (4, 3)
    assert calls == []


def test_memory_source_is_exactly_mu_plus_sigma_times_epsilon() -> None:
    mu = np.arange(6, dtype=np.float64).reshape(2, 3)
    sigma = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    memory_source = MemorySource(mu=mu, sigma=sigma)

    rng = np.random.default_rng(23)
    expected_rng = np.random.default_rng(23)
    expected = mu + sigma * expected_rng.standard_normal(size=mu.shape)

    actual = sample_memory_source(memory_source, rng, expected_shape=mu.shape)
    np.testing.assert_allclose(actual, expected)


def test_injected_rng_makes_component_and_source_sequence_deterministic() -> None:
    probabilities = np.array([0.2, 0.4, 0.4])
    payloads = [
        MemorySource(mu=np.full((2, 2), -1.0), sigma=0.25),
        MemorySource(mu=np.full((2, 2), 2.0), sigma=0.5),
    ]

    def getter(index: int) -> MemorySource:
        return payloads[index]

    def rollout(seed: int) -> tuple[list[int], list[np.ndarray]]:
        rng = np.random.default_rng(seed)
        components: list[int] = []
        sources: list[np.ndarray] = []
        for _ in range(20):
            component, source = sample_source_from_mixture(
                probabilities,
                (2, 2),
                rng,
                getter,
            )
            components.append(component)
            sources.append(source)
        return components, sources

    components_a, sources_a = rollout(101)
    components_b, sources_b = rollout(101)
    assert components_a == components_b
    for source_a, source_b in zip(sources_a, sources_b, strict=True):
        np.testing.assert_array_equal(source_a, source_b)


def test_memory_component_uses_candidate_index_offset_by_one() -> None:
    requested: list[int] = []

    def getter(index: int) -> MemorySource:
        requested.append(index)
        return MemorySource(mu=np.full((1, 2), index), sigma=0.1)

    source = sample_source_for_component(
        component=2,
        action_shape=(1, 2),
        rng=np.random.default_rng(3),
        memory_source_getter=getter,
    )

    assert source.shape == (1, 2)
    assert requested == [1]


@pytest.mark.parametrize(
    ("required", "observed", "context"),
    [
        ([np.nan, 0.0], [[1.0, 0.0]], [0.0]),
        ([1.0, 0.0], [1.0, 0.0], [0.0]),
        ([1.0, 0.0], [[1.0, 0.0]], [0.0, 1.0]),
        ([1.0, 0.0], [[1.0, np.inf]], [0.0]),
    ],
)
def test_candidate_mixture_rejects_invalid_shapes_and_values(
    required: object,
    observed: object,
    context: object,
) -> None:
    with pytest.raises(ValueError):
        build_candidate_mixture(
            required,
            observed,
            context,
            null_logit=0.0,
        )


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.2, 0.2],
        [1.1, -0.1],
        [np.nan, 0.0],
        [],
    ],
)
def test_component_sampling_rejects_invalid_probabilities(
    probabilities: object,
) -> None:
    with pytest.raises(ValueError):
        sample_component(probabilities, np.random.default_rng(0))


def test_source_sampling_rejects_nonfinite_or_wrong_shaped_payloads() -> None:
    with pytest.raises(ValueError):
        sample_memory_source(
            MemorySource(mu=[[0.0, np.nan]], sigma=0.2),
            np.random.default_rng(0),
        )
    with pytest.raises(ValueError):
        sample_memory_source(
            MemorySource(mu=[[0.0, 1.0]], sigma=-0.2),
            np.random.default_rng(0),
        )
    with pytest.raises(ValueError):
        sample_memory_source(
            MemorySource(mu=[[0.0, 1.0]], sigma=0.2),
            np.random.default_rng(0),
            expected_shape=(2, 1),
        )
