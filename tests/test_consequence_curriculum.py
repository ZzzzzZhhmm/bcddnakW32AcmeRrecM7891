from __future__ import annotations

import pytest

from fastwam.models.warm.consequence_curriculum import (
    CorruptionCurriculumError,
    CorruptionDecision,
    CorruptionWeights,
    derive_corruption_batch,
    derive_corruption_decision,
    select_hard_negative_index,
)


def test_sha256_decision_is_stable_and_covers_all_curriculum_modes() -> None:
    expected = {
        "q-0": ("normal", 8727477138659511867),
        "q-5": ("hard_negative", 16181739625832782448),
        "q-14": ("null", 7927567706016202418),
        "q-16": ("drop", 13712263866649058987),
    }
    for identity, (mode, token) in expected.items():
        first = derive_corruption_decision(17, identity)
        second = derive_corruption_decision(17, identity)
        assert first == second
        assert first.mode == mode
        assert first.hard_negative_token == token
        assert 0.0 <= first.unit_draw < 1.0


def test_batch_order_does_not_change_identity_keyed_decisions() -> None:
    identities = ("suite/task/episode-0/frame-3", "suite/task/episode-1/frame-9")
    forward = dict(zip(identities, derive_corruption_batch(123, identities)))
    reverse_identities = tuple(reversed(identities))
    reverse = dict(
        zip(reverse_identities, derive_corruption_batch(123, reverse_identities))
    )
    assert forward == reverse
    assert derive_corruption_batch(124, identities) != tuple(forward.values())
    repeated = derive_corruption_batch(123, (identities[0], identities[0]))
    assert repeated[0] == repeated[1]


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        (CorruptionWeights(1.0, 0.0, 0.0, 0.0), "normal"),
        (CorruptionWeights(0.0, 1.0, 0.0, 0.0), "drop"),
        (CorruptionWeights(0.0, 0.0, 1.0, 0.0), "null"),
        (CorruptionWeights(0.0, 0.0, 0.0, 1.0), "hard_negative"),
    ],
)
def test_each_mode_can_be_selected_without_runtime_rng(
    weights: CorruptionWeights, expected: str
) -> None:
    assert derive_corruption_decision(0, "query", weights=weights).mode == expected


def test_hard_negative_slot_uses_prederived_entropy() -> None:
    decision = CorruptionDecision(
        mode="hard_negative", unit_draw=0.9, hard_negative_token=11
    )
    assert select_hard_negative_index(decision, (2, 5, 8)) == 8
    assert select_hard_negative_index(decision, (8, 5, 2)) == 2

    with pytest.raises(CorruptionCurriculumError, match="at least one"):
        select_hard_negative_index(decision, ())
    with pytest.raises(CorruptionCurriculumError, match="unique"):
        select_hard_negative_index(decision, (1, 1))
    with pytest.raises(CorruptionCurriculumError, match="requires"):
        select_hard_negative_index(
            CorruptionDecision("normal", 0.1, 0), (1,)
        )


@pytest.mark.parametrize(
    "weights",
    [
        (0.5, 0.2, 0.2, 0.2),
        (-0.1, 0.1, 0.5, 0.5),
        (float("nan"), 0.0, 0.0, 1.0),
    ],
)
def test_probability_simplex_is_closed_and_finite(
    weights: tuple[float, float, float, float]
) -> None:
    with pytest.raises(CorruptionCurriculumError):
        CorruptionWeights(*weights)


def test_seed_identity_and_batch_contracts_reject_ambiguous_inputs() -> None:
    with pytest.raises(TypeError, match="seed"):
        derive_corruption_decision(True, "query")
    with pytest.raises(CorruptionCurriculumError, match="seed"):
        derive_corruption_decision(-1, "query")
    with pytest.raises(CorruptionCurriculumError, match="sample_identity"):
        derive_corruption_decision(1, " query ")
    with pytest.raises(TypeError, match="sample_identity"):
        derive_corruption_batch(1, ("query", 3))  # type: ignore[arg-type]
