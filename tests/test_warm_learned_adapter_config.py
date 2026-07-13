from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fastwam.models.warm.adapter_config import (
    AdapterConfigError,
    RetrospectiveEventAdapterConfig,
    RetrospectiveGistConfig,
    SemanticBridgeConfig,
)


def test_adapter_configs_are_cpu_only_strict_and_frozen() -> None:
    bridge = SemanticBridgeConfig(
        early_dim=12, late_dim=16, semantic_dim=8, num_heads=2
    )
    gist = RetrospectiveGistConfig(
        world_dim=8,
        semantic_dim=8,
        episode_dim=6,
        event_dim=5,
        text_dim=7,
        model_dim=16,
        future_target_dim=10,
        num_heads=4,
    )
    event = RetrospectiveEventAdapterConfig(
        action_dim=7,
        action_horizon=16,
        timing_dim=4,
        world_dim=8,
        gist_dim=16,
        proprio_dim=9,
        text_dim=7,
        event_dim=5,
        model_dim=16,
        effect_dim=10,
        num_heads=4,
    )
    assert bridge.query_count == 4
    assert gist.query_count == 8
    assert event.context_query_count == 4
    assert event.rho == pytest.approx(0.25)
    with pytest.raises(FrozenInstanceError):
        event.rho = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory,match",
    [
        (
            lambda: SemanticBridgeConfig(
                early_dim=4,
                late_dim=4,
                semantic_dim=7,
                num_heads=2,
            ),
            "divisible",
        ),
        (
            lambda: SemanticBridgeConfig(
                early_dim=4, late_dim=4, query_count=5
            ),
            "query_count=4",
        ),
        (
            lambda: RetrospectiveGistConfig(
                world_dim=4,
                semantic_dim=4,
                episode_dim=4,
                event_dim=4,
                text_dim=4,
                query_count=7,
            ),
            "query_count=8",
        ),
        (
            lambda: RetrospectiveEventAdapterConfig(
                action_dim=2,
                action_horizon=4,
                timing_dim=4,
                world_dim=4,
                gist_dim=4,
                proprio_dim=3,
                text_dim=4,
                event_dim=4,
                context_query_count=3,
            ),
            "context_query_count=4",
        ),
        (
            lambda: RetrospectiveEventAdapterConfig(
                action_dim=2,
                action_horizon=4,
                timing_dim=4,
                world_dim=4,
                gist_dim=4,
                proprio_dim=3,
                text_dim=4,
                event_dim=4,
                rho=0.0,
            ),
            "rho",
        ),
    ],
)
def test_adapter_configs_reject_method_drift(factory, match: str) -> None:
    with pytest.raises(AdapterConfigError, match=match):
        factory()


def test_adapter_configs_reject_bool_dimensions_and_invalid_dropout() -> None:
    with pytest.raises(AdapterConfigError, match="early_dim"):
        SemanticBridgeConfig(early_dim=True, late_dim=8)
    with pytest.raises(AdapterConfigError, match="dropout"):
        RetrospectiveGistConfig(
            world_dim=4,
            semantic_dim=4,
            episode_dim=4,
            event_dim=4,
            text_dim=4,
            dropout=1.0,
        )
