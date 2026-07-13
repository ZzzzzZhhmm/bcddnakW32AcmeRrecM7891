from __future__ import annotations

import pytest

from fastwam.models.warm.source_contract import (
    SOURCE_RUN_SCHEMA,
    SOURCE_RUN_SCHEMA_VERSION,
    SourceRunContractError,
    WarmSourceRunContract,
)


def _contract_dict(**overrides):
    value = {
        "schema": SOURCE_RUN_SCHEMA,
        "version": SOURCE_RUN_SCHEMA_VERSION,
        "bank_manifest_sha256": "1" * 64,
        "bank_content_sha256": "2" * 64,
        "candidate_manifest_sha256": "3" * 64,
        "query_corpus_sha256": "4" * 64,
        "catalog_sha256": "5" * 64,
        "audit_sha256": "6" * 64,
        "normalization_stats_sha256": "7" * 64,
        "action_space_contract_sha256": "8" * 64,
        "base_checkpoint_sha256": "9" * 64,
        "query_split": "train",
        "global_sample_stride": 1,
        "action_horizon": 32,
        "action_dim": 7,
    }
    value.update(overrides)
    return value


def test_source_run_contract_round_trips_and_hashes_canonically() -> None:
    first = WarmSourceRunContract.from_dict(_contract_dict())
    second = WarmSourceRunContract.from_dict(dict(reversed(list(_contract_dict().items()))))
    assert first.to_dict() == _contract_dict()
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_source_run_contract_is_closed_world() -> None:
    missing = _contract_dict()
    missing.pop("audit_sha256")
    with pytest.raises(SourceRunContractError, match="missing"):
        WarmSourceRunContract.from_dict(missing)

    extra = _contract_dict(extra_field="unsafe")
    with pytest.raises(SourceRunContractError, match="extra"):
        WarmSourceRunContract.from_dict(extra)


@pytest.mark.parametrize("split", ["test", "training", ""])
def test_source_run_contract_rejects_unknown_query_split(split: str) -> None:
    with pytest.raises(SourceRunContractError, match="query_split"):
        WarmSourceRunContract.from_dict(_contract_dict(query_split=split))


@pytest.mark.parametrize("stride", [0, 2, 4])
def test_source_run_contract_requires_raw_frame_stride_one(stride: int) -> None:
    with pytest.raises((SourceRunContractError, TypeError), match="global_sample_stride"):
        WarmSourceRunContract.from_dict(
            _contract_dict(global_sample_stride=stride)
        )


def test_source_run_contract_rejects_non_digest_and_shape_values() -> None:
    with pytest.raises(SourceRunContractError, match="base_checkpoint_sha256"):
        WarmSourceRunContract.from_dict(
            _contract_dict(base_checkpoint_sha256="not-a-digest")
        )
    with pytest.raises(SourceRunContractError, match="action_horizon"):
        WarmSourceRunContract.from_dict(_contract_dict(action_horizon=0))
