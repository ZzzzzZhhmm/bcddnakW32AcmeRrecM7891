from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from fastwam.memory.action_contract import (
    ACTION_SPACE_SCHEMA,
    ACTION_SPACE_SCHEMA_VERSION,
    ActionSpaceContract,
    ActionSpaceContractError,
    validate_action_space_contract,
)


def _valid_contract() -> dict[str, object]:
    return {
        "schema": "warm.action-space",
        "version": 1,
        "action_dim": 7,
        "arm_dims": [0, 1, 2, 3, 4, 5],
        "gripper_dims": [6],
        "gripper_threshold": 0.0,
        "normalization_mode": "fastwam.dataset-stats-v1",
        "normalization_stats_sha256": "a" * 64,
        "control_mode": "ee-delta-pose",
        "embodiment": "libero-panda",
    }


def test_action_space_contract_round_trips_to_canonical_json() -> None:
    raw = _valid_contract()
    contract = validate_action_space_contract(raw)

    assert ACTION_SPACE_SCHEMA == "warm.action-space"
    assert ACTION_SPACE_SCHEMA_VERSION == 1
    assert contract.arm_dims == (0, 1, 2, 3, 4, 5)
    assert contract.gripper_dims == (6,)
    assert contract.to_dict() == raw
    assert ActionSpaceContract.from_dict(contract.to_dict()) == contract


def test_action_space_contract_is_immutable() -> None:
    contract = ActionSpaceContract.from_dict(_valid_contract())

    with pytest.raises(FrozenInstanceError):
        contract.action_dim = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "warm.action-space-v2"),
        ("version", 2),
        ("version", True),
        ("action_dim", 0),
        ("action_dim", True),
        ("action_dim", 7.0),
    ],
)
def test_action_space_contract_rejects_wrong_identity_or_dimension(field, value) -> None:
    raw = _valid_contract()
    raw[field] = value

    with pytest.raises(ActionSpaceContractError):
        ActionSpaceContract.from_dict(raw)


def test_action_space_contract_rejects_missing_and_unknown_fields() -> None:
    missing = _valid_contract()
    del missing["control_mode"]
    with pytest.raises(ActionSpaceContractError, match="missing=.*control_mode"):
        ActionSpaceContract.from_dict(missing)

    extra = _valid_contract()
    extra["units"] = "meters"
    with pytest.raises(ActionSpaceContractError, match="extra=.*units"):
        ActionSpaceContract.from_dict(extra)


@pytest.mark.parametrize(
    "arm_dims,gripper_dims,error",
    [
        ([0, 1, 1, 2, 3, 4, 5], [6], "duplicate"),
        ([0, 1, 2, 3, 4, 5, 6], [6], "overlap"),
        ([0, 1, 2, 3, 4], [6], "cover every"),
        ([0, 1, 2, 3, 4, 7], [6], "outside"),
        ([0, 1, 2, 3, 4, -1], [6], "outside"),
        ([0, 2, 1, 3, 4, 5], [6], "ascending"),
        ([0, 1, 2, 3, 4, 5, True], [6], "integers"),
        ([], [0, 1, 2, 3, 4, 5, 6], "at least one"),
        ([0, 1, 2, 3, 4, 5, 6], [], "at least one"),
    ],
)
def test_action_channels_must_be_a_complete_unique_partition(
    arm_dims, gripper_dims, error
) -> None:
    raw = _valid_contract()
    raw["arm_dims"] = arm_dims
    raw["gripper_dims"] = gripper_dims

    with pytest.raises(ActionSpaceContractError, match=error):
        ActionSpaceContract.from_dict(raw)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0"])
def test_gripper_threshold_must_be_a_finite_real(value) -> None:
    raw = _valid_contract()
    raw["gripper_threshold"] = value

    with pytest.raises(ActionSpaceContractError, match="finite real"):
        ActionSpaceContract.from_dict(raw)


@pytest.mark.parametrize("value", [None, "", "A" * 64, "0" * 63, "z" * 64])
def test_normalization_statistics_must_be_content_addressed(value) -> None:
    raw = _valid_contract()
    raw["normalization_stats_sha256"] = value

    with pytest.raises(ActionSpaceContractError, match="lowercase SHA-256"):
        ActionSpaceContract.from_dict(raw)


@pytest.mark.parametrize("field", ["normalization_mode", "control_mode", "embodiment"])
@pytest.mark.parametrize("value", ["", "   ", " padded", "padded ", "bad\x00value", 3])
def test_semantic_identifiers_must_be_canonical_nonempty_strings(field, value) -> None:
    raw = _valid_contract()
    raw[field] = value

    with pytest.raises(ActionSpaceContractError):
        ActionSpaceContract.from_dict(raw)


def test_json_dimension_fields_must_be_lists() -> None:
    raw = _valid_contract()
    raw["arm_dims"] = (0, 1, 2, 3, 4, 5)

    with pytest.raises(ActionSpaceContractError, match="JSON list"):
        ActionSpaceContract.from_dict(raw)
