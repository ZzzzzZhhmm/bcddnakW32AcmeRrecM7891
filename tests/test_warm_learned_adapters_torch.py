from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from fastwam.models.warm.adapter_config import (  # noqa: E402
    RetrospectiveEventAdapterConfig,
    RetrospectiveGistConfig,
    SemanticBridgeConfig,
)
from fastwam.models.warm.retrospective_event_adapter import (  # noqa: E402
    RetrospectiveEventAdapter,
    RetrospectiveEventAdapterError,
)
from fastwam.models.warm.retrospective_gist import (  # noqa: E402
    RetrospectiveGistAdapter,
    RetrospectiveGistError,
)
from fastwam.models.warm.semantic_bridge import (  # noqa: E402
    SemanticBridgeError,
    WorldFeatureSemanticBridge,
)


def test_semantic_bridge_shapes_mask_weights_and_stop_gradient() -> None:
    torch.manual_seed(1)
    config = SemanticBridgeConfig(
        early_dim=6,
        late_dim=8,
        semantic_dim=4,
        num_heads=2,
    )
    module = WorldFeatureSemanticBridge(config)
    early = torch.randn(2, 5, 6, requires_grad=True)
    late = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    output = module(early, late, mask)
    assert output.world_tokens.shape == (2, 5, 4)
    assert output.semantic_tokens.shape == (2, 4, 4)
    assert torch.equal(output.world_tokens[0, 3:], torch.zeros(2, 4))
    assert torch.allclose(output.layer_weights, torch.tensor([0.5, 0.5]))
    assert torch.allclose(output.layer_weights.sum(), torch.tensor(1.0))

    teacher = torch.randn(2, 4, 4, requires_grad=True)
    loss = module.alignment_loss(output, teacher)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert teacher.grad is None
    assert early.grad is not None
    assert late.grad is not None


def test_semantic_bridge_rejects_misaligned_streams_and_empty_sample() -> None:
    config = SemanticBridgeConfig(
        early_dim=4, late_dim=4, semantic_dim=4, num_heads=2
    )
    module = WorldFeatureSemanticBridge(config)
    with pytest.raises(SemanticBridgeError, match="aligned"):
        module(
            torch.randn(1, 3, 4),
            torch.randn(1, 4, 4),
            torch.ones(1, 3, dtype=torch.bool),
        )
    with pytest.raises(SemanticBridgeError, match="at least one"):
        module(
            torch.randn(1, 3, 4),
            torch.randn(1, 3, 4),
            torch.zeros(1, 3, dtype=torch.bool),
        )


def _gist_inputs():
    return {
        "world_tokens": torch.randn(2, 5, 6),
        "world_mask": torch.tensor(
            [[True, True, True, False, False], [True] * 5]
        ),
        "semantic_tokens": torch.randn(2, 4, 4),
        "semantic_mask": torch.ones(2, 4, dtype=torch.bool),
        "episode_tokens": torch.randn(2, 3, 5),
        "episode_mask": torch.tensor(
            [[True, False, False], [True, True, False]]
        ),
        "event_pre_tokens": torch.randn(2, 3, 2, 7),
        "event_delta_tokens": torch.randn(2, 3, 2, 7),
        "event_token_mask": torch.tensor(
            [
                [[True, True], [True, False], [False, False]],
                [[True, True], [False, False], [False, False]],
            ]
        ),
        "text_tokens": torch.randn(2, 3, 9),
        "text_mask": torch.tensor(
            [[True, True, False], [True, True, True]]
        ),
    }


def test_gist_adapter_exact_queries_blocks_and_future_stop_gradient() -> None:
    torch.manual_seed(2)
    config = RetrospectiveGistConfig(
        world_dim=6,
        semantic_dim=4,
        episode_dim=5,
        event_dim=7,
        text_dim=9,
        model_dim=8,
        future_target_dim=6,
        num_heads=2,
    )
    module = RetrospectiveGistAdapter(config)
    output = module(**_gist_inputs())
    assert len(module.blocks) == 2
    assert output.gist_tokens.shape == (2, 8, 8)
    assert output.pooled_gist.shape == (2, 8)
    assert output.source_mask.ndim == 2

    target = torch.randn(2, 6, requires_grad=True)
    loss = module.future_alignment_loss(output, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert target.grad is None
    assert module.gist_queries.grad is not None


def test_gist_adapter_rejects_event_shape_and_mask_mismatch() -> None:
    config = RetrospectiveGistConfig(
        world_dim=6,
        semantic_dim=4,
        episode_dim=5,
        event_dim=7,
        text_dim=9,
        model_dim=8,
        future_target_dim=6,
        num_heads=2,
    )
    module = RetrospectiveGistAdapter(config)
    inputs = _gist_inputs()
    inputs["event_delta_tokens"] = torch.randn(2, 2, 2, 7)
    with pytest.raises(RetrospectiveGistError, match="identical shapes"):
        module(**inputs)


def _event_inputs():
    valid = torch.tensor([[True, True, False], [True, False, False]])
    actions = torch.randn(2, 3, 5, 3)
    timing = torch.randn(2, 3, 4)
    actions[~valid] = 0.0
    timing[~valid] = 0.0
    delta_mask = valid.unsqueeze(-1).expand(-1, -1, 2).clone()
    return {
        "warped_actions": actions,
        "gripper_timing": timing,
        "candidate_valid_mask": valid,
        "world_tokens": torch.randn(2, 4, 6),
        "world_mask": torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        ),
        "gist_tokens": torch.randn(2, 8, 8),
        "proprio": torch.randn(2, 3),
        "text_tokens": torch.randn(2, 3, 5),
        "text_mask": torch.tensor(
            [[True, True, False], [True, True, True]]
        ),
        "event_delta_tokens": torch.randn(2, 3, 2, 4),
        "event_delta_mask": delta_mask,
    }


def _event_module() -> RetrospectiveEventAdapter:
    config = RetrospectiveEventAdapterConfig(
        action_dim=3,
        action_horizon=5,
        timing_dim=4,
        world_dim=6,
        gist_dim=8,
        proprio_dim=3,
        text_dim=5,
        event_dim=4,
        model_dim=8,
        effect_dim=6,
        num_heads=2,
        rho=0.25,
    )
    return RetrospectiveEventAdapter(
        config, action_std=torch.tensor([1.0, 2.0, 0.5])
    )


def test_event_adapter_zero_init_shapes_bounds_and_effect_stop_gradient() -> None:
    torch.manual_seed(3)
    module = _event_module()
    inputs = _event_inputs()
    output = module(**inputs)
    valid = inputs["candidate_valid_mask"]
    assert output.adapted_action_mean.shape == (2, 3, 5, 3)
    assert output.predicted_effect.shape == (2, 3, 6)
    assert output.action_context_tokens.shape == (2, 3, 4, 8)
    assert torch.equal(output.action_residual, torch.zeros_like(output.action_residual))
    assert torch.allclose(output.adapted_action_mean, inputs["warped_actions"])
    assert torch.equal(
        output.action_context_tokens,
        torch.zeros_like(output.action_context_tokens),
    )
    assert torch.equal(
        output.predicted_effect[~valid],
        torch.zeros_like(output.predicted_effect[~valid]),
    )

    target = torch.randn(2, 6, requires_grad=True)
    loss = module.effect_alignment_loss(output, target)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert target.grad is None

    with torch.no_grad():
        module.residual_head.bias.fill_(100.0)
    bounded = module(**_event_inputs()).action_residual
    limits = 0.25 * torch.tensor([1.0, 2.0, 0.5])
    assert bool((bounded.abs() <= limits.view(1, 1, 1, 3) + 1.0e-6).all())


def test_event_adapter_rejects_nonprefix_and_nonzero_padding() -> None:
    module = _event_module()
    inputs = _event_inputs()
    inputs["candidate_valid_mask"] = torch.tensor(
        [[True, False, True], [True, False, False]]
    )
    with pytest.raises(RetrospectiveEventAdapterError, match="true prefix"):
        module(**inputs)

    inputs = _event_inputs()
    inputs["warped_actions"][0, 2, 0, 0] = 1.0
    with pytest.raises(RetrospectiveEventAdapterError, match="must be zero"):
        module(**inputs)
