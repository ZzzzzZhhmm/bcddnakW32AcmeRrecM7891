from __future__ import annotations

import numpy as np
import pytest

from fastwam.memory.bank_builder import EpisodeFeatures, build_event_bank
from fastwam.memory.event_mining import EventMiningConfig


def _episode(dataset_index: int, episode_index: int, *, action_dim: int = 2) -> EpisodeFeatures:
    steps = 8
    actions = np.arange(steps * action_dim, dtype=np.float32).reshape(steps, action_dim)
    proprio = np.arange((steps + 1) * 3, dtype=np.float32).reshape(steps + 1, 3)
    gripper = np.zeros(steps + 1, dtype=np.float32)
    gripper[4:] = 1.0
    context = np.zeros((steps + 1, 3), dtype=np.float32)
    context[:, 0] = 1.0
    context[:, 1] = dataset_index
    context[:, 2] = episode_index
    semantics = np.arange((steps + 1) * 4, dtype=np.float32).reshape(steps + 1, 4)
    return EpisodeFeatures(
        dataset_id=f"dataset-{dataset_index}",
        dataset_index=dataset_index,
        episode_index=episode_index,
        task_index=dataset_index,
        model_actions=actions,
        proprio=proprio,
        gripper=gripper,
        context_keys=context,
        semantic_features=semantics,
    )


def test_builder_keeps_exact_fixed_horizon_actions_and_factual_effects() -> None:
    episode = _episode(0, 3)
    bank = build_event_bank(
        [episode],
        mining_config=EventMiningConfig(action_horizon=4),
        start_mode="uniform",
    )

    assert len(bank) == 2
    np.testing.assert_array_equal(bank.payload("model_action")[0], episode.model_actions[0:4])
    np.testing.assert_array_equal(bank.payload("model_action")[1], episode.model_actions[4:8])
    np.testing.assert_array_equal(bank.payload("effect_pre")[0], episode.semantic_features[0])
    np.testing.assert_array_equal(bank.payload("effect_post")[0], episode.semantic_features[4])
    assert bank.payload("contains_forced_gripper").tolist() == [True, False]


def test_builder_produces_globally_distinct_ids_and_episode_exclusion() -> None:
    first = _episode(0, 0)
    second = _episode(1, 0)
    bank = build_event_bank(
        [first, second],
        mining_config=EventMiningConfig(action_horizon=4),
        start_mode="uniform",
    )

    assert len(set(bank.event_ids)) == 4
    results = bank.search(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        top_k=10,
        exclude_episode=bank.event_ids[0],
    )
    assert results
    assert all(result.event_id.dataset_index == 1 for result in results)


def test_episode_features_require_terminal_factual_semantic_state() -> None:
    episode = _episode(0, 0)
    with pytest.raises(ValueError, match=r"T\+1 factual states"):
        EpisodeFeatures(
            dataset_id=episode.dataset_id,
            dataset_index=episode.dataset_index,
            episode_index=episode.episode_index,
            task_index=episode.task_index,
            model_actions=episode.model_actions,
            proprio=episode.proprio,
            gripper=episode.gripper,
            context_keys=episode.context_keys,
            semantic_features=episode.semantic_features[:-1],
        )


def test_builder_rejects_action_layout_mismatch() -> None:
    with pytest.raises(ValueError, match="model_actions trailing shape"):
        build_event_bank(
            [_episode(0, 0, action_dim=2), _episode(0, 1, action_dim=3)],
            mining_config=EventMiningConfig(action_horizon=4),
        )
