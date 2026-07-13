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

from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from fastwam.models.warm.source_transport import (
    ActionSourceContext,
    SourceTransportError,
    build_action_flow_pair,
    measure_source_geometry,
    resolve_action_source,
    select_source_components,
)


def test_fixed_top1_uses_rank_zero_exactly_and_empty_rows_fall_back() -> None:
    valid = torch.tensor(
        [
            [True, True, False],
            [False, True, True],
            [False, False, False],
        ],
        dtype=torch.bool,
    )

    components = select_source_components(
        valid,
        policy="fixed_context_top1",
        phase="train",
    )

    assert components.tolist() == [1, 0, 0]


def test_memory_enabled_mask_forces_the_same_explicit_null() -> None:
    valid = torch.ones((3, 1), dtype=torch.bool)
    enabled = torch.tensor([True, False, True], dtype=torch.bool)
    components = select_source_components(
        valid,
        policy="fixed_context_top1",
        phase="train",
        memory_enabled_mask=enabled,
    )
    assert components.tolist() == [1, 0, 1]


def test_gaussian_policy_never_requires_candidates() -> None:
    valid = torch.empty((2, 0), dtype=torch.bool)
    components = select_source_components(
        valid,
        policy="gaussian_null",
        phase="rollout",
    )
    assert components.tolist() == [0, 0]


def test_oracle_is_an_explicit_non_deployable_upper_bound() -> None:
    valid = torch.tensor([[True, True], [True, False]], dtype=torch.bool)
    oracle = torch.tensor([1, -1], dtype=torch.long)

    components = select_source_components(
        valid,
        policy="oracle_action_top1",
        phase="offline",
        oracle_candidate_indices=oracle,
    )
    assert components.tolist() == [2, 0]

    for phase in ("infer", "rollout"):
        with pytest.raises(SourceTransportError, match="forbidden"):
            select_source_components(
                valid,
                policy="oracle_action_top1",
                phase=phase,
                oracle_candidate_indices=oracle,
            )


def test_oracle_rejects_invalid_or_missing_slots() -> None:
    valid = torch.tensor([[True, False]], dtype=torch.bool)
    with pytest.raises(SourceTransportError, match="requires precomputed"):
        select_source_components(
            valid,
            policy="oracle_action_top1",
            phase="train",
        )
    with pytest.raises(SourceTransportError, match="invalid candidate"):
        select_source_components(
            valid,
            policy="oracle_action_top1",
            phase="train",
            oracle_candidate_indices=torch.tensor([1], dtype=torch.long),
        )


def test_all_null_returns_original_gaussian_without_reading_payload() -> None:
    gaussian = torch.randn(2, 4, 3)
    context = ActionSourceContext(
        component_indices=torch.zeros(2, dtype=torch.long),
        candidate_means=None,
        candidate_valid_mask=None,
    )

    output = resolve_action_source(gaussian, context, memory_sigma=0.2)

    assert output.source is gaussian
    assert output.base_gaussian is gaussian
    assert output.selected_means is None
    assert not output.memory_mask.any()
    assert output.selected_candidate_indices.tolist() == [-1, -1]


def test_mixed_batch_uses_mu_plus_sigma_epsilon_and_preserves_null_rows() -> None:
    gaussian = torch.tensor(
        [
            [[1.0, -1.0], [2.0, -2.0]],
            [[3.0, -3.0], [4.0, -4.0]],
            [[5.0, -5.0], [6.0, -6.0]],
        ]
    )
    means = torch.zeros((3, 2, 2, 2), dtype=gaussian.dtype)
    means[0, 0] = 10.0
    means[2, 1] = -10.0
    valid = torch.tensor(
        [[True, True], [False, False], [True, True]], dtype=torch.bool
    )
    context = ActionSourceContext(
        component_indices=torch.tensor([1, 0, 2], dtype=torch.long),
        candidate_means=means,
        candidate_valid_mask=valid,
    )

    output = resolve_action_source(gaussian, context, memory_sigma=0.25)

    torch.testing.assert_close(output.source[0], 10.0 + 0.25 * gaussian[0])
    torch.testing.assert_close(output.source[1], gaussian[1])
    torch.testing.assert_close(output.source[2], -10.0 + 0.25 * gaussian[2])
    assert output.component_indices.tolist() == [1, 0, 2]
    assert output.memory_mask.tolist() == [True, False, True]


def test_selected_memory_payload_is_stop_gradient() -> None:
    gaussian = torch.randn(1, 3, 2)
    means = torch.randn(1, 1, 3, 2, requires_grad=True)
    context = ActionSourceContext(
        component_indices=torch.ones(1, dtype=torch.long),
        candidate_means=means,
        candidate_valid_mask=torch.ones((1, 1), dtype=torch.bool),
    )

    output = resolve_action_source(gaussian, context, memory_sigma=0.2)

    assert not output.source.requires_grad
    assert output.selected_means is not None
    assert not output.selected_means.requires_grad


@pytest.mark.parametrize("bad_sigma", [0.0, -0.1, float("inf"), float("nan")])
def test_memory_sigma_must_be_positive_and_finite(bad_sigma: float) -> None:
    gaussian = torch.randn(1, 2, 2)
    context = ActionSourceContext(
        component_indices=torch.zeros(1, dtype=torch.long),
        candidate_means=None,
        candidate_valid_mask=None,
    )
    with pytest.raises(SourceTransportError, match="memory_sigma"):
        resolve_action_source(gaussian, context, memory_sigma=bad_sigma)


def test_memory_source_rejects_broadcast_dtype_and_invalid_component() -> None:
    gaussian = torch.randn(1, 2, 3, dtype=torch.float32)
    valid = torch.ones((1, 1), dtype=torch.bool)

    with pytest.raises(SourceTransportError, match="candidate_means must have shape"):
        resolve_action_source(
            gaussian,
            ActionSourceContext(
                component_indices=torch.ones(1, dtype=torch.long),
                candidate_means=torch.zeros(1, 1, 1, 3),
                candidate_valid_mask=valid,
            ),
            memory_sigma=0.2,
        )

    with pytest.raises(TypeError, match="same dtype"):
        resolve_action_source(
            gaussian,
            ActionSourceContext(
                component_indices=torch.ones(1, dtype=torch.long),
                candidate_means=torch.zeros(1, 1, 2, 3, dtype=torch.float64),
                candidate_valid_mask=valid,
            ),
            memory_sigma=0.2,
        )

    with pytest.raises(SourceTransportError, match="outside"):
        resolve_action_source(
            gaussian,
            ActionSourceContext(
                component_indices=torch.tensor([2], dtype=torch.long),
                candidate_means=torch.zeros(1, 1, 2, 3),
                candidate_valid_mask=valid,
            ),
            memory_sigma=0.2,
        )


def test_memory_source_rejects_nonfinite_selected_payload() -> None:
    gaussian = torch.randn(1, 2, 2)
    means = torch.zeros(1, 1, 2, 2)
    means[0, 0, 0, 0] = float("nan")
    with pytest.raises(SourceTransportError, match="must be finite"):
        resolve_action_source(
            gaussian,
            ActionSourceContext(
                component_indices=torch.ones(1, dtype=torch.long),
                candidate_means=means,
                candidate_valid_mask=torch.ones((1, 1), dtype=torch.bool),
            ),
            memory_sigma=0.2,
        )


def test_fastwam_scheduler_pair_has_source_to_data_sign() -> None:
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=1.0)
    action = torch.tensor([[[2.0, 4.0]]])
    source = torch.tensor([[[10.0, -2.0]]])
    timestep = torch.tensor([250.0])

    pair = build_action_flow_pair(action, source, timestep, scheduler)

    expected = 0.75 * action + 0.25 * source
    torch.testing.assert_close(pair.noisy_action, expected)
    torch.testing.assert_close(pair.target_velocity, source - action)
    assert pair.source_action.data_ptr() == source.data_ptr()


def test_flow_pair_detaches_source_and_rejects_implicit_broadcast() -> None:
    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=1.0)
    action = torch.zeros(1, 2, 3)
    source = torch.ones(1, 2, 3, requires_grad=True)
    pair = build_action_flow_pair(action, source, torch.tensor([500.0]), scheduler)
    assert not pair.source_action.requires_grad
    assert not pair.target_velocity.requires_grad

    with pytest.raises(SourceTransportError, match="shape"):
        build_action_flow_pair(
            action,
            torch.ones(1, 1, 3),
            torch.tensor([500.0]),
            scheduler,
        )


def test_source_geometry_honors_padding_and_all_pad_is_safe() -> None:
    target = torch.zeros(2, 3, 2)
    source = torch.tensor(
        [
            [[3.0, 4.0], [100.0, 100.0], [100.0, 100.0]],
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        ]
    )
    is_pad = torch.tensor(
        [[False, True, True], [True, True, True]], dtype=torch.bool
    )

    geometry = measure_source_geometry(source, target, action_is_pad=is_pad)

    torch.testing.assert_close(geometry.l2, torch.tensor([5.0, 0.0]))
    torch.testing.assert_close(
        geometry.rms,
        torch.tensor([(25.0 / 2.0) ** 0.5, 0.0]),
    )
