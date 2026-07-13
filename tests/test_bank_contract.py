from __future__ import annotations

import numpy as np
import pytest

from fastwam.memory.bank_builder import EpisodeFeatures, build_event_bank
from fastwam.memory.bank_contract import WarmBankContractError, validate_warm_v1_bank
from fastwam.memory.event_bank import EventBank
from fastwam.memory.event_mining import EventMiningConfig
from fastwam.memory.schema import EventId


def _episode(
    index: int,
    *,
    source_hash: str | None = None,
    feature_hash: str | None = None,
) -> EpisodeFeatures:
    steps = 4
    keys = np.ones((steps + 1, 3), dtype=np.float32)
    keys[:, 1] = index + 1
    return EpisodeFeatures(
        dataset_id="libero",
        dataset_index=0,
        episode_index=index,
        task_index=0,
        source_episode_sha256=source_hash or f"{index + 1:064x}",
        feature_episode_sha256=feature_hash or f"{index + 101:064x}",
        model_actions=np.zeros((steps, 7), dtype=np.float32),
        proprio=np.zeros((steps + 1, 8), dtype=np.float32),
        gripper=np.zeros((steps + 1,), dtype=np.float32),
        context_keys=keys,
        semantic_features=np.zeros((steps + 1, 2, 5), dtype=np.float32),
    )


def test_canonical_builder_passes_warm_contract() -> None:
    bank = build_event_bank(
        [_episode(0), _episode(1)],
        mining_config=EventMiningConfig(action_horizon=4),
        start_mode="uniform",
    )

    summary = validate_warm_v1_bank(
        bank, expected_action_horizon=4, expected_action_dim=7
    )
    assert summary.num_events == 2
    assert summary.num_source_episodes == 2
    assert summary.effect_shape == (2, 5)


def test_contract_rejects_alias_payload_and_duplicate_source_content() -> None:
    wrong_name = EventBank.from_arrays(
        [EventId("d", 0, 0, 0)],
        np.ones((1, 2), dtype=np.float32),
        model_action=np.zeros((1, 4, 7), dtype=np.float32),
    )
    with pytest.raises(WarmBankContractError, match="model_space_action"):
        validate_warm_v1_bank(wrong_name)

    digest = "a" * 64
    duplicate = build_event_bank(
        [_episode(0, source_hash=digest), _episode(1, source_hash=digest)],
        mining_config=EventMiningConfig(action_horizon=4),
        start_mode="uniform",
    )
    with pytest.raises(WarmBankContractError, match="duplicate source episode content"):
        validate_warm_v1_bank(duplicate)

    duplicate_feature = build_event_bank(
        [
            _episode(0, feature_hash=digest),
            _episode(1, feature_hash=digest),
        ],
        mining_config=EventMiningConfig(action_horizon=4),
        start_mode="uniform",
    )
    with pytest.raises(WarmBankContractError, match="duplicate feature episode content"):
        validate_warm_v1_bank(duplicate_feature)
