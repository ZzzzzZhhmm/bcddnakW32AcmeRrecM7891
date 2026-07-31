from __future__ import annotations

import os

import pytest

if os.environ.get("WARM_REQUIRE_TORCH_TESTS") == "1":
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - server contract
        raise RuntimeError(
            "WARM_REQUIRE_TORCH_TESTS=1 but PyTorch is unavailable"
        ) from error
else:
    torch = pytest.importorskip("torch")

from fastwam.models.warm.consequence import (
    AUTO_SELECT,
    FORCE_NULL,
    ActionUtilityReranker,
    ConsequenceAlignmentError,
    SourceConfidenceGate,
    build_action_effect_utility_targets,
    build_corruption_controls,
    calibrate_source_acceptance,
    consequence_consistency,
    select_consequence_candidate,
    utility_kl_divergence,
    utility_supervised_gate_bce,
    utility_supervised_gate_target,
)


def test_reranker_is_compact_finite_and_masks_padding_without_infinity() -> None:
    torch.manual_seed(7)
    reranker = ActionUtilityReranker(
        query_dim=5,
        candidate_context_dim=4,
        action_summary_dim=3,
        effect_dim=2,
        timing_dim=2,
        hidden_dim=16,
    )
    valid = torch.tensor([[True, False, True], [False, False, False]])
    scores = reranker(
        torch.randn(2, 5),
        torch.randn(2, 3, 4),
        torch.randn(2, 3, 3),
        torch.randn(2, 3, 2),
        torch.randn(2, 3, 2),
        valid,
    )
    assert scores.shape == (2, 3)
    assert torch.isfinite(scores).all()
    assert scores.masked_select(~valid).tolist() == [0.0, 0.0, 0.0, 0.0]
    assert sum(parameter.numel() for parameter in reranker.parameters()) < 5_000_000


def test_action_effect_soft_targets_and_masked_kl_include_null_rows_safely() -> None:
    candidate_actions = torch.tensor(
        [
            [[[0.0], [0.0]], [[1.0], [1.0]], [[4.0], [4.0]]],
            [[[9.0], [9.0]], [[8.0], [8.0]], [[7.0], [7.0]]],
        ]
    )
    target_action = torch.zeros(2, 2, 1)
    candidate_effects = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        ]
    )
    target_effect = torch.zeros(2, 2)
    valid = torch.tensor([[True, True, False], [False, False, False]])
    targets = build_action_effect_utility_targets(
        candidate_actions,
        target_action,
        candidate_effects,
        target_effect,
        valid,
        effect_weight=2.0,
        temperature=0.5,
    )

    torch.testing.assert_close(targets.probabilities[0].sum(), torch.tensor(1.0))
    torch.testing.assert_close(targets.probabilities[1], torch.zeros(3))
    assert targets.probabilities[0, 2].item() == 0.0
    assert targets.valid_rows.tolist() == [True, False]
    predicted_scores = targets.utilities / 0.5
    loss = utility_kl_divergence(
        predicted_scores, targets.probabilities, valid
    )
    torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0.0)

    null_loss = utility_kl_divergence(
        torch.zeros(2, 0),
        torch.zeros(2, 0),
        torch.zeros(2, 0, dtype=torch.bool),
    )
    assert null_loss.item() == 0.0


def test_consequence_consistency_checks_direction_and_log_magnitude() -> None:
    predicted = torch.tensor([[[3.0, 4.0], [6.0, 8.0], [-3.0, -4.0]]])
    gist = torch.tensor([[3.0, 4.0]])
    score = consequence_consistency(
        predicted, gist, magnitude_weight=0.5, epsilon=1e-8
    )
    expected = torch.tensor([[1.0, 1.0 - 0.5 * torch.log(torch.tensor(2.0)), -1.0]])
    torch.testing.assert_close(score, expected, atol=1e-6, rtol=1e-6)


def test_selector_compares_candidates_with_explicit_null_and_reports_margin() -> None:
    retrieval = torch.tensor([[0.2, 0.3], [-0.2, -0.1], [4.0, 3.0]])
    consistency = torch.tensor([[0.9, 0.2], [0.0, 0.0], [1.0, 1.0]])
    support = torch.tensor([[3, 9], [1, 1], [1, 1]])
    valid = torch.tensor([[True, True], [True, True], [False, False]])
    selected = select_consequence_candidate(
        retrieval,
        consistency,
        support,
        valid,
        consequence_weight=1.0,
        support_weight=0.0,
        null_score=0.0,
    )

    assert selected.candidate_indices.tolist() == [0, -1, -1]
    assert selected.component_indices.tolist() == [1, 0, 0]
    assert selected.memory_mask.tolist() == [True, False, False]
    torch.testing.assert_close(selected.selection_margin, torch.tensor([0.6, 0.1, 0.0]))
    torch.testing.assert_close(selected.selected_consistency, torch.tensor([0.9, 0.0, 0.0]))


def test_corruption_controls_distinguish_drop_null_and_forced_hard_negative() -> None:
    valid = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, False, False],
            [True, True, False],
        ]
    )
    controls = build_corruption_controls(
        valid,
        ("normal", "drop", "null", "hard_negative"),
        hard_negative_indices=torch.tensor([-1, -1, -1, 1]),
    )
    assert controls.forced_candidate_indices.tolist() == [
        AUTO_SELECT,
        AUTO_SELECT,
        FORCE_NULL,
        1,
    ]
    assert controls.candidate_valid_mask.tolist() == [
        [True, True, False],
        [False, False, False],
        [True, False, False],
        [True, True, False],
    ]

    selected = select_consequence_candidate(
        torch.tensor([[2.0, 1.0, 0.0]] * 4),
        torch.zeros(4, 3),
        torch.ones(4, 3),
        controls.candidate_valid_mask,
        forced_candidate_indices=controls.forced_candidate_indices,
    )
    assert selected.candidate_indices.tolist() == [0, -1, -1, 1]


def test_gate_starts_at_bias_minus_two_and_uses_utility_supervision() -> None:
    selection = select_consequence_candidate(
        torch.tensor([[2.0], [-1.0]]),
        torch.tensor([[0.8], [0.0]]),
        torch.tensor([[4], [0]]),
        torch.tensor([[True], [False]]),
    )
    gate = SourceConfidenceGate(hidden_dim=8)
    output = gate(selection, torch.tensor([0.1, 0.0]))
    torch.testing.assert_close(output.logits, torch.tensor([-2.0, -2.0]))
    torch.testing.assert_close(
        output.probability,
        torch.tensor([torch.sigmoid(torch.tensor(-2.0)), 0.0]),
    )
    assert output.features.shape == (2, 7)

    target = utility_supervised_gate_target(
        torch.zeros(2, 2, 1),
        torch.zeros(2, 2, 1),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        selection.memory_mask,
    )
    assert target.tolist() == [1.0, 0.0]
    loss = utility_supervised_gate_bce(output, target)
    assert torch.isfinite(loss)
    loss.backward()
    final = gate.network[-1]
    assert final.bias.grad is not None
    assert final.bias.grad.abs().item() > 0.0


def test_masked_utility_kl_never_builds_nonfinite_padded_gradients() -> None:
    scores = torch.tensor(
        [[2.0, -3.0, 99.0], [4.0, -2.0, 1.0]],
        requires_grad=True,
    )
    targets = torch.tensor(
        [[0.75, 0.25, 0.0], [0.0, 0.0, 0.0]],
    )
    valid = torch.tensor(
        [[True, True, False], [False, False, False]],
    )

    loss = utility_kl_divergence(scores, targets, valid)
    loss.backward()

    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad[0, 2:]).item() == 0
    assert torch.count_nonzero(scores.grad[1]).item() == 0


def test_all_null_gate_loss_is_differentiable_zero() -> None:
    selection = select_consequence_candidate(
        torch.empty(2, 0),
        torch.empty(2, 0),
        torch.empty(2, 0, dtype=torch.long),
        torch.empty(2, 0, dtype=torch.bool),
    )
    gate = SourceConfidenceGate()
    output = gate(selection, torch.zeros(2))
    loss = utility_supervised_gate_bce(output, torch.zeros(2))
    assert loss.item() == 0.0
    loss.backward()


def test_candidate_confidence_is_shift_invariant_and_ambiguous_rows_fall_back() -> None:
    scores = torch.tensor([[0.001, 0.0, -0.001], [2.0, -1.0, -2.0]])
    kwargs = {
        "consistency_scores": torch.zeros_like(scores),
        "support_count": torch.ones_like(scores, dtype=torch.long),
        "candidate_valid_mask": torch.ones_like(scores, dtype=torch.bool),
        "automatic_null": False,
        "selection_temperature": 0.25,
    }
    original = select_consequence_candidate(scores, **kwargs)
    shifted = select_consequence_candidate(scores + 137.0, **kwargs)
    torch.testing.assert_close(
        original.selected_probability, shifted.selected_probability
    )
    torch.testing.assert_close(original.probability_margin, shifted.probability_margin)
    torch.testing.assert_close(original.normalized_entropy, shifted.normalized_entropy)

    gate = SourceConfidenceGate(hidden_dim=8)
    gate_output = gate(original, torch.zeros(2), torch.zeros(2))
    acceptance = calibrate_source_acceptance(
        gate_output,
        original,
        torch.zeros(2),
        minimum_candidate_probability=0.06,
        maximum_candidate_entropy=0.94,
        stagnation_decay=3.0,
        stagnation_hard_threshold=0.75,
        inference_gate_threshold=0.01,
        hard_reject=True,
    )
    assert acceptance.accepted_mask.tolist() == [False, True]
    assert acceptance.effective_probability[0].item() == 0.0


def test_factual_stagnation_forces_explicit_gaussian_null() -> None:
    selection = select_consequence_candidate(
        torch.tensor([[3.0, -2.0]]),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
        automatic_null=False,
        selection_temperature=0.25,
    )
    gate = SourceConfidenceGate(hidden_dim=8)
    output = gate(selection, torch.zeros(1), torch.tensor([0.9]))
    acceptance = calibrate_source_acceptance(
        output,
        selection,
        torch.tensor([0.9]),
        minimum_candidate_probability=0.06,
        maximum_candidate_entropy=0.94,
        stagnation_decay=3.0,
        stagnation_hard_threshold=0.75,
        inference_gate_threshold=0.0,
        hard_reject=True,
    )
    assert acceptance.accepted_mask.tolist() == [False]
    assert acceptance.effective_probability.tolist() == [0.0]


def test_contracts_reject_nonfinite_broadcast_and_invalid_forcing() -> None:
    reranker = ActionUtilityReranker(
        query_dim=2,
        candidate_context_dim=2,
        action_summary_dim=2,
        effect_dim=2,
        timing_dim=1,
        hidden_dim=4,
    )
    bad_query = torch.tensor([[float("nan"), 0.0]])
    with pytest.raises(ConsequenceAlignmentError, match="finite"):
        reranker(
            bad_query,
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, dtype=torch.bool),
        )

    with pytest.raises(ConsequenceAlignmentError, match="invalid slot"):
        select_consequence_candidate(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.ones(1, 2),
            torch.tensor([[True, False]]),
            forced_candidate_indices=torch.tensor([1]),
        )

    with pytest.raises(ConsequenceAlignmentError, match="target_action"):
        build_action_effect_utility_targets(
            torch.zeros(1, 2, 3, 1),
            torch.zeros(1, 2, 1),
            torch.zeros(1, 2, 2),
            torch.zeros(1, 2),
            torch.ones(1, 2, dtype=torch.bool),
        )
