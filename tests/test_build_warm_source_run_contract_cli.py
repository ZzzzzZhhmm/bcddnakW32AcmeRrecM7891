from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastwam.memory.candidate_cache import (
    CandidateCache,
    canonical_event_bank_content_hash,
)
from fastwam.memory.event_bank import EventBank, MANIFEST_FILENAME
from fastwam.memory.manifest import (
    sha256_canonical_json,
    sha256_file,
)
from fastwam.models.warm.source_contract import WarmSourceRunContract
from fastwam.utils.artifact_claim import (
    ArtifactAlreadyClaimedError,
    artifact_claim,
)
import scripts.build_warm_source_run_contract as contract_cli
from scripts.build_warm_source_run_contract import (
    SourceRunContractBuildError,
    main,
)
from tests.test_runtime_candidates import (
    AUDIT_HASH,
    CATALOG_HASH,
    QUERY_CORPUS_HASH,
    STATS_HASH,
    _action_contract,
    _write_artifacts,
)


def _base_argv(
    bank: Path,
    candidates: Path,
    checkpoint: Path,
    output: Path,
) -> list[str]:
    return [
        "--bank",
        str(bank),
        "--candidate-cache",
        str(candidates),
        "--base-checkpoint",
        str(checkpoint),
        "--output",
        str(output),
    ]


def test_cli_builds_closed_contract_from_verified_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bank, candidates, _ = _write_artifacts(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "base.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"synthetic FastWAM checkpoint\x00v1")
    output = tmp_path / "contracts" / "m2-source-run.json"

    assert (
        main(
            [
                *_base_argv(bank, candidates, checkpoint, output),
                "--expected-query-corpus-sha256",
                QUERY_CORPUS_HASH,
                "--expected-action-horizon",
                "4",
                "--expected-action-dim",
                "3",
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    contract = WarmSourceRunContract.from_dict(written)
    assert summary["contract"] == written
    assert summary["source_run_contract_sha256"] == contract.sha256
    assert contract.bank_manifest_sha256 == sha256_file(bank / MANIFEST_FILENAME)
    assert contract.base_checkpoint_sha256 == sha256_file(checkpoint)
    assert contract.query_corpus_sha256 == QUERY_CORPUS_HASH
    assert contract.catalog_sha256 == CATALOG_HASH
    assert contract.audit_sha256 == AUDIT_HASH
    assert contract.normalization_stats_sha256 == STATS_HASH
    assert contract.action_space_contract_sha256 == sha256_canonical_json(
        _action_contract().to_dict()
    )
    assert contract.query_split == "train"
    assert contract.global_sample_stride == 1
    assert contract.action_horizon == 4
    assert contract.action_dim == 3
    assert not list(output.parent.glob(".*.tmp"))
    assert not contract_cli._lock_path(output.resolve()).exists()


def test_cli_forbids_overwrite_by_default_and_can_replace_explicitly(
    tmp_path: Path,
) -> None:
    bank, candidates, _ = _write_artifacts(tmp_path)
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base-v1")
    output = tmp_path / "source-run.json"
    argv = _base_argv(bank, candidates, checkpoint, output)

    assert main(argv) == 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        main(argv)
    assert output.read_bytes() == original

    checkpoint.write_bytes(b"base-v2")
    assert main([*argv, "--overwrite"]) == 0
    replacement = WarmSourceRunContract.from_dict(
        json.loads(output.read_text(encoding="utf-8"))
    )
    assert replacement.base_checkpoint_sha256 == sha256_file(checkpoint)
    assert output.read_bytes() != original


@pytest.mark.parametrize(
    ("flag", "value", "match"),
    [
        ("--expected-action-horizon", "5", "action horizon"),
        ("--expected-action-dim", "7", "action dimension"),
    ],
)
def test_cli_rejects_expected_model_shape_mismatch(
    tmp_path: Path,
    flag: str,
    value: str,
    match: str,
) -> None:
    bank, candidates, _ = _write_artifacts(tmp_path)
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base")
    output = tmp_path / "source-run.json"

    with pytest.raises(SourceRunContractBuildError, match=match):
        main(
            [
                *_base_argv(bank, candidates, checkpoint, output),
                flag,
                value,
            ]
        )
    assert not output.exists()


def _write_dev_stride_two_cache(
    tmp_path: Path,
    bank_path: Path,
    train_candidates: Path,
) -> Path:
    bank = EventBank.load(bank_path)
    assert bank.manifest is not None
    source = CandidateCache.load(train_candidates)
    cache = CandidateCache(source.query_ids, source.candidates)
    output = tmp_path / "dev-candidates"
    cache.save(
        output,
        event_bank_manifest_hash=sha256_file(bank_path / MANIFEST_FILENAME),
        event_bank_content_hash=canonical_event_bank_content_hash(
            bank.manifest.content_hashes
        ),
        query_corpus_hash="9" * 64,
        query_key_encoder=bank.manifest.encoder,
        build_recipe={
            "implementation": "exact_cosine_v1",
            "action_horizon": 4,
            "query_stride": 2,
            "top_k": 3,
            "query_split": "dev",
            "query_data_binding": {
                "catalog_sha256": CATALOG_HASH,
                "audit_report_sha256": AUDIT_HASH,
                "split": "dev",
            },
            "episode_exclusion": [
                "global_episode_identity",
                "source_episode_sha256",
                "feature_episode_sha256",
            ],
        },
    )
    return output


def test_cli_rejects_non_unit_query_stride_for_any_m2_contract(
    tmp_path: Path,
) -> None:
    bank, train_candidates, _ = _write_artifacts(tmp_path)
    candidates = _write_dev_stride_two_cache(tmp_path, bank, train_candidates)
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base")
    output = tmp_path / "source-run.json"

    with pytest.raises(SourceRunContractBuildError, match="query_stride=1"):
        main(
            [
                *_base_argv(bank, candidates, checkpoint, output),
                "--query-split",
                "dev",
            ]
        )
    assert not output.exists()


@pytest.mark.parametrize("changed_input", ["candidate", "checkpoint"])
def test_cli_refuses_publication_if_an_input_changes_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_input: str,
) -> None:
    bank, candidates, _ = _write_artifacts(tmp_path)
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base")
    output = tmp_path / "source-run.json"
    original_builder = contract_cli._build_contract

    def build_then_mutate(*args, **kwargs):
        contract = original_builder(*args, **kwargs)
        if changed_input == "candidate":
            manifest = candidates / contract_cli.CANDIDATE_MANIFEST_FILENAME
            manifest.write_bytes(manifest.read_bytes() + b"\n")
        else:
            checkpoint.write_bytes(b"changed checkpoint")
        return contract

    monkeypatch.setattr(contract_cli, "_build_contract", build_then_mutate)
    with pytest.raises(SourceRunContractBuildError, match="changed while"):
        main(_base_argv(bank, candidates, checkpoint, output))
    assert not output.exists()
    assert not list(output.parent.glob(".*.tmp"))
    assert not contract_cli._lock_path(output.resolve()).exists()


@pytest.mark.parametrize("unsafe_root", ["bank", "candidate"])
def test_cli_rejects_output_under_immutable_artifacts(
    tmp_path: Path,
    unsafe_root: str,
) -> None:
    bank, candidates, _ = _write_artifacts(tmp_path)
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base")
    root = bank if unsafe_root == "bank" else candidates
    output = root / "source-run.json"

    with pytest.raises(SourceRunContractBuildError, match="outside the immutable"):
        main(_base_argv(bank, candidates, checkpoint, output))
    assert not output.exists()


def test_cli_honors_cross_process_publication_claim(tmp_path: Path) -> None:
    bank, candidates, _ = _write_artifacts(tmp_path)
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base")
    output = tmp_path / "contracts" / "source-run.json"
    output.parent.mkdir()
    lock_path = contract_cli._lock_path(output.resolve())

    with artifact_claim(lock_path, purpose="competing source contract writer"):
        with pytest.raises(ArtifactAlreadyClaimedError, match="competing"):
            main(_base_argv(bank, candidates, checkpoint, output))
        assert lock_path.exists()
        assert not output.exists()
    assert not lock_path.exists()


def test_cli_requires_a_regular_base_checkpoint(tmp_path: Path) -> None:
    bank, candidates, _ = _write_artifacts(tmp_path)
    checkpoint = tmp_path / "checkpoint-directory"
    checkpoint.mkdir()
    output = tmp_path / "source-run.json"

    with pytest.raises(SourceRunContractBuildError, match="regular file"):
        main(_base_argv(bank, candidates, checkpoint, output))
    assert not output.exists()
