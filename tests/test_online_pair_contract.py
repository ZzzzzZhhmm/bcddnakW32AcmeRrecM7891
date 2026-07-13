from __future__ import annotations

from copy import deepcopy

import pytest

from fastwam.models.warm.online_pair_contract import (
    ALLOWED_CONFIG_DIFFERENCE_PATHS,
    ONLINE_PAIR_KIND,
    OnlinePairContractError,
    WarmOnlinePairContract,
)


def _value() -> dict[str, object]:
    return {
        "schema": "warm.online-policy-checkpoint-pair-contract",
        "version": 1,
        "comparison_kind": ONLINE_PAIR_KIND,
        "fixed_online_run_contract_sha256": "1" * 64,
        "gaussian_null_online_run_contract_sha256": "2" * 64,
        "fixed_resolved_eval_config_sha256": "3" * 64,
        "gaussian_null_resolved_eval_config_sha256": "4" * 64,
        "fixed_warm_checkpoint_sha256": "5" * 64,
        "gaussian_null_warm_checkpoint_sha256": "6" * 64,
        "fixed_training_attestation_sha256": "8" * 64,
        "gaussian_null_training_attestation_sha256": "9" * 64,
        "shared_training_recipe_sha256": "b" * 64,
        "shared_training_runtime_sha256": "c" * 64,
        "parity_report_sha256": "0" * 64,
        "shared_science_identity_sha256": "7" * 64,
        "allowed_config_difference_paths": list(
            ALLOWED_CONFIG_DIFFERENCE_PATHS
        ),
        "observed_config_difference_paths": list(
            ALLOWED_CONFIG_DIFFERENCE_PATHS
        ),
        "git_commit": "a" * 40,
        "git_dirty": False,
    }


def test_online_pair_contract_round_trip_and_digest_are_deterministic() -> None:
    value = _value()
    contract = WarmOnlinePairContract.from_dict(value)
    assert contract.to_dict() == value
    assert WarmOnlinePairContract.from_dict(contract.to_dict()).sha256 == contract.sha256
    assert len(contract.sha256) == 64


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("fixed_online_run_contract_sha256", "A" * 64),
        ("git_commit", "b" * 39),
        ("git_dirty", True),
        ("comparison_kind", "same_weights_intervention"),
        (
            "allowed_config_difference_paths",
            ["/ckpt", "/model/source_policy"],
        ),
        (
            "observed_config_difference_paths",
            [*ALLOWED_CONFIG_DIFFERENCE_PATHS, "/EVALUATION/num_trials"],
        ),
        (
            "observed_config_difference_paths",
            [
                path
                for path in ALLOWED_CONFIG_DIFFERENCE_PATHS
                if path != "/EVALUATION/output_dir"
            ],
        ),
    ],
)
def test_online_pair_contract_rejects_non_closed_world_values(
    field: str,
    bad: object,
) -> None:
    value = deepcopy(_value())
    value[field] = bad
    with pytest.raises((OnlinePairContractError, TypeError)):
        WarmOnlinePairContract.from_dict(value)


def test_online_pair_contract_rejects_equal_checkpoint_identities() -> None:
    value = _value()
    value["gaussian_null_warm_checkpoint_sha256"] = value[
        "fixed_warm_checkpoint_sha256"
    ]
    with pytest.raises(OnlinePairContractError, match="distinct checkpoint"):
        WarmOnlinePairContract.from_dict(value)


def test_online_pair_contract_rejects_equal_training_attestations() -> None:
    value = _value()
    value["gaussian_null_training_attestation_sha256"] = value[
        "fixed_training_attestation_sha256"
    ]
    with pytest.raises(OnlinePairContractError, match="training attestations"):
        WarmOnlinePairContract.from_dict(value)


@pytest.mark.parametrize(
    ("fixed_field", "null_field", "message"),
    [
        (
            "fixed_online_run_contract_sha256",
            "gaussian_null_online_run_contract_sha256",
            "online contract identities",
        ),
        (
            "fixed_resolved_eval_config_sha256",
            "gaussian_null_resolved_eval_config_sha256",
            "resolved config identities",
        ),
    ],
)
def test_online_pair_contract_rejects_equal_policy_specific_identities(
    fixed_field: str,
    null_field: str,
    message: str,
) -> None:
    value = _value()
    value[null_field] = value[fixed_field]
    with pytest.raises(OnlinePairContractError, match=message):
        WarmOnlinePairContract.from_dict(value)


def test_online_pair_contract_rejects_unknown_fields() -> None:
    value = _value()
    value["sampler"] = "changed"
    with pytest.raises(OnlinePairContractError, match="extra"):
        WarmOnlinePairContract.from_dict(value)
