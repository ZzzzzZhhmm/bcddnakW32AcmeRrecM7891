from __future__ import annotations

import math

import pytest

from fastwam.models.warm.online_contract import (
    ONLINE_RETRIEVAL_IMPLEMENTATION,
    ONLINE_RUN_SCHEMA,
    ONLINE_RUN_SCHEMA_VERSION,
    OnlineRunContractError,
    WarmOnlineRunContract,
)


def _contract_dict(**overrides):
    value = {
        "schema": ONLINE_RUN_SCHEMA,
        "version": ONLINE_RUN_SCHEMA_VERSION,
        "training_run_contract_sha256": "0" * 64,
        "validation_run_contract_sha256": "a" * 64,
        "warm_checkpoint_sha256": "1" * 64,
        "training_attestation_sha256": "4" * 64,
        "shared_training_recipe_sha256": "5" * 64,
        "training_runtime_sha256": "6" * 64,
        "bank_manifest_sha256": "2" * 64,
        "bank_content_sha256": "3" * 64,
        "encoder_contract_sha256": "4" * 64,
        "encoder_runtime_sha256": "0" * 64,
        "camera_contract_sha256": "5" * 64,
        "m1_data_config_sha256": "f" * 64,
        "dino_checkpoint_tree_sha256": "6" * 64,
        "dino_checkpoint_file_count": 7,
        "normalization_stats_sha256": "8" * 64,
        "action_space_contract_sha256": "9" * 64,
        "catalog_sha256": "a" * 64,
        "audit_sha256": "b" * 64,
        "resolved_eval_config_sha256": "c" * 64,
        "vae_checkpoint_sha256": "d" * 64,
        "text_encoder_tree_sha256": "e" * 64,
        "tokenizer_tree_sha256": "0" * 64,
        "evaluation_namespace_sha256": "f" * 64,
        "task_suite": "libero_10",
        "task_id": 3,
        "task_description": "put the bowl on the plate",
        "root_seed": 17,
        "initial_states_sha256": "1" * 64,
        "bddl_sha256": "2" * 64,
        "retrieval_implementation": ONLINE_RETRIEVAL_IMPLEMENTATION,
        "top_k": 4,
        "source_policy": "fixed_context_top1",
        "memory_sigma": 0.2,
        "action_horizon": 32,
        "action_dim": 7,
        "git_commit": "3" * 40,
        "git_dirty": False,
    }
    value.update(overrides)
    return value


def test_online_contract_round_trips_and_hashes_canonically() -> None:
    first = WarmOnlineRunContract.from_dict(_contract_dict())
    second = WarmOnlineRunContract.from_dict(
        dict(reversed(list(_contract_dict().items())))
    )

    assert first.to_dict() == _contract_dict()
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_online_contract_is_closed_world() -> None:
    missing = _contract_dict()
    missing.pop("encoder_contract_sha256")
    with pytest.raises(OnlineRunContractError, match="missing"):
        WarmOnlineRunContract.from_dict(missing)

    with pytest.raises(OnlineRunContractError, match="extra"):
        WarmOnlineRunContract.from_dict(_contract_dict(unsafe_path="x"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_run_contract_sha256", "not-a-digest"),
        ("bank_manifest_sha256", "A" * 64),
        ("dino_checkpoint_file_count", 0),
        ("top_k", 0),
        ("action_horizon", 0),
        ("action_dim", -1),
        ("task_id", -1),
        ("root_seed", -1),
        ("git_commit", "4" * 39),
    ],
)
def test_online_contract_rejects_invalid_identity_or_shape(
    field: str, value: object
) -> None:
    with pytest.raises((OnlineRunContractError, TypeError), match=field):
        WarmOnlineRunContract.from_dict(_contract_dict(**{field: value}))


@pytest.mark.parametrize("value", [0.0, -0.1, math.inf, math.nan, True])
def test_online_contract_requires_finite_positive_memory_sigma(value: object) -> None:
    with pytest.raises((OnlineRunContractError, TypeError), match="memory_sigma"):
        WarmOnlineRunContract.from_dict(_contract_dict(memory_sigma=value))


@pytest.mark.parametrize("policy", ["oracle_action_top1", "learned", ""])
def test_online_contract_rejects_non_deployment_source_policy(policy: str) -> None:
    with pytest.raises(OnlineRunContractError, match="source_policy"):
        WarmOnlineRunContract.from_dict(_contract_dict(source_policy=policy))


def test_online_contract_requires_exact_retrieval_implementation() -> None:
    with pytest.raises(OnlineRunContractError, match="retrieval implementation"):
        WarmOnlineRunContract.from_dict(
            _contract_dict(retrieval_implementation="approximate_ann_v0")
        )


@pytest.mark.parametrize("value", ["", " libero_10", "libero_10 ", "a\x00b"])
def test_online_contract_requires_normalized_strings(value: str) -> None:
    with pytest.raises(OnlineRunContractError, match="task_suite"):
        WarmOnlineRunContract.from_dict(_contract_dict(task_suite=value))


def test_online_contract_rejects_dirty_formal_rollout() -> None:
    with pytest.raises(OnlineRunContractError, match="clean Git tree"):
        WarmOnlineRunContract.from_dict(_contract_dict(git_dirty=True))
