from __future__ import annotations

import numpy as np
import pytest

from fastwam.memory.robotwin_artifacts import (
    ROBOTWIN_ACTION_DIM,
    ROBOTWIN_ARM_DIMS,
    ROBOTWIN_CONTROL_MODE,
    ROBOTWIN_GRIPPER_DIMS,
    ROBOTWIN_NORMALIZATION_MODE,
    RobotwinArtifactContractError,
    RobotwinQposZScore,
    robotwin_qpos_action_contract,
)


def _stats() -> dict[str, object]:
    return {
        "action": {
            "default": {
                "global_mean": [float(index) for index in range(14)],
                "global_std": [float(index + 1) / 10.0 for index in range(14)],
                "stepwise_mean": [[0.0] * 14],
            }
        }
    }


def test_robotwin_action_contract_binds_qpos_partition_and_global_zscore() -> None:
    contract = robotwin_qpos_action_contract(
        normalization_stats_sha256="a" * 64,
        gripper_threshold=0.25,
    )

    assert contract.action_dim == ROBOTWIN_ACTION_DIM == 14
    assert contract.gripper_dims == ROBOTWIN_GRIPPER_DIMS == (6, 13)
    assert contract.arm_dims == ROBOTWIN_ARM_DIMS
    assert set(contract.arm_dims) | set(contract.gripper_dims) == set(range(14))
    assert contract.control_mode == ROBOTWIN_CONTROL_MODE
    assert contract.normalization_mode == ROBOTWIN_NORMALIZATION_MODE
    assert contract.gripper_threshold == pytest.approx(0.25)


def test_robotwin_zscore_matches_fastwam_formula_and_round_trips() -> None:
    normalizer = RobotwinQposZScore.from_dataset_stats(_stats())
    raw = np.stack(
        [normalizer.mean, normalizer.mean + normalizer.std], axis=0
    ).astype(np.float32)

    model, restored = normalizer.checked_round_trip(raw)

    expected = (raw - normalizer.mean) / (
        normalizer.std + np.float32(1e-8)
    )
    np.testing.assert_allclose(model, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(restored, raw, rtol=1e-5, atol=1e-6)
    for value in (normalizer.mean, normalizer.std, model, restored):
        assert value.dtype == np.float32
        assert value.flags.c_contiguous
        assert not value.flags.writeable


def test_robotwin_zscore_clipping_is_explicit_and_not_claimed_as_reversible() -> None:
    normalizer = RobotwinQposZScore.from_dataset_stats(_stats())
    extreme = normalizer.mean + normalizer.std * np.float32(10.0)

    clipped = normalizer.normalize(extreme)
    np.testing.assert_array_equal(clipped, np.full((14,), 5.0, dtype=np.float32))
    with pytest.raises(RobotwinArtifactContractError, match="reversible"):
        normalizer.normalize(extreme, fail_on_clip=True)
    with pytest.raises(RobotwinArtifactContractError, match="reversible"):
        normalizer.checked_round_trip(extreme)


@pytest.mark.parametrize(
    "stats",
    [
        {},
        {"action": {}},
        {"action": {"default": {"global_mean": [0.0] * 14}}},
        {
            "action": {
                "default": {
                    "global_mean": [0.0] * 13,
                    "global_std": [1.0] * 14,
                }
            }
        },
    ],
)
def test_robotwin_zscore_rejects_incomplete_or_wrong_dimensional_stats(stats) -> None:
    with pytest.raises(RobotwinArtifactContractError):
        RobotwinQposZScore.from_dataset_stats(stats)
