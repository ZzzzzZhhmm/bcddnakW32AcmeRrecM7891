from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fastwam.memory.bank_builder import EpisodeFeatures
from fastwam.memory.candidate_cache import (
    CandidateCache,
    MANIFEST_FILENAME,
    canonical_event_bank_content_hash,
)
from fastwam.memory.event_bank import (
    MANIFEST_FILENAME as EVENT_BANK_MANIFEST_FILENAME,
    EventBank,
)
from fastwam.memory.event_mining import EventMiningConfig
from fastwam.memory.feature_cache import (
    FeatureCacheMetadata,
    save_episode_feature_cache,
)
from fastwam.memory.offline_pipeline import (
    FeatureDataBinding,
    build_event_bank_from_collection,
    load_feature_cache_collection,
)
from fastwam.memory.manifest import sha256_file
from fastwam.utils.artifact_claim import (
    ArtifactAlreadyClaimedError,
    artifact_claim,
)
import scripts.build_warm_candidate_cache as candidate_cli
from scripts.build_warm_candidate_cache import CandidateCacheBuildError, main
from tests.warm_test_data import write_test_catalog_and_audit


HASHES = {
    "catalog_hash": "a" * 64,
    "normalizer_hash": "b" * 64,
    "encoder_hash": "c" * 64,
    "camera_hash": "d" * 64,
}


def _episode(index: int) -> EpisodeFeatures:
    steps = 8
    actions = np.zeros((steps, 3), dtype=np.float32)
    actions[:, :2] = float(index + 1)
    actions[4:, 2] = 1.0
    context = np.zeros((steps + 1, 3), dtype=np.float32)
    context[:, 0] = 1.0
    context[:, 1] = index * 0.05
    context[:, 2] = np.linspace(0.01, 0.02, steps + 1, dtype=np.float32)
    semantic = np.full((steps + 1, 2, 4), float(index), dtype=np.float32)
    semantic[:, :, 0] += np.arange(steps + 1, dtype=np.float32)[:, None]
    return EpisodeFeatures(
        dataset_id="libero",
        dataset_index=0,
        episode_index=index,
        task_index=0,
        source_episode_sha256=f"{index + 1:064x}",
        model_actions=actions,
        proprio=np.full((steps + 1, 8), float(index), dtype=np.float32),
        gripper=np.concatenate(
            [np.zeros(4, dtype=np.float32), np.ones(steps + 1 - 4, dtype=np.float32)]
        ),
        context_keys=context,
        semantic_features=semantic,
    )


def _save_feature(
    root: Path,
    episode: EpisodeFeatures,
    *,
    split: str,
    catalog_hash: str,
) -> Path:
    path = root / f"episode-{episode.episode_index}.npz"
    metadata = FeatureCacheMetadata.for_episode(
        episode,
        split=split,
        **{**HASHES, "catalog_hash": catalog_hash},
    )
    save_episode_feature_cache(path, episode, metadata=metadata)
    return path


def _bank_and_queries(tmp_path: Path) -> tuple[Path, Path, Path]:
    episodes = [_episode(index) for index in range(4)]
    _, _, catalog, audit_report = write_test_catalog_and_audit(
        tmp_path / "data-contract",
        [
            (episodes[0], "train"),
            (episodes[1], "train"),
            (episodes[2], "dev"),
            (episodes[3], "dev"),
        ],
    )
    train_paths = [
        _save_feature(
            tmp_path / "train", episodes[0], split="train", catalog_hash=catalog.content_sha256
        ),
        _save_feature(
            tmp_path / "train", episodes[1], split="train", catalog_hash=catalog.content_sha256
        ),
    ]
    first_query = _save_feature(
        tmp_path / "queries", episodes[2], split="dev", catalog_hash=catalog.content_sha256
    )
    second_query = _save_feature(
        tmp_path / "queries", episodes[3], split="dev", catalog_hash=catalog.content_sha256
    )
    train = load_feature_cache_collection(train_paths)
    bank, _, provenance = build_event_bank_from_collection(
        train,
        mining_config=EventMiningConfig(action_horizon=4),
        start_mode="uniform",
        data_binding=FeatureDataBinding(
            catalog_sha256=catalog.content_sha256,
            audit_report_sha256=str(audit_report["report_sha256"]),
            split="train",
        ),
    )
    bank_path = tmp_path / "bank"
    bank.save(
        bank_path,
        action_normalizer={"file_sha256": HASHES["normalizer_hash"]},
        encoder={"file_sha256": HASHES["encoder_hash"]},
        camera_layout={"file_sha256": HASHES["camera_hash"]},
        provenance=provenance,
    )
    return bank_path, first_query, second_query


def _data_args(bank_path: Path) -> list[str]:
    root = bank_path.parent / "data-contract"
    return [
        "--catalog",
        str(root / "catalog.json"),
        "--audit-report",
        str(root / "audit.json"),
    ]


def test_cli_builds_self_validates_and_writes_atomic_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bank_path, query_path, second_query = _bank_and_queries(tmp_path)
    output = tmp_path / "candidates"
    summary_path = tmp_path / "reports" / "candidate-summary.json"

    result = main(
        [
            "--bank",
            str(bank_path),
            *_data_args(bank_path),
            "--feature-cache",
            str(query_path),
            "--feature-cache",
            str(second_query),
            "--output",
            str(output),
            "--query-stride",
            "4",
            "--top-k",
            "3",
            "--summary",
            str(summary_path),
        ]
    )

    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert printed == written
    assert printed["action_horizon"] == 4
    assert printed["query_count"] == 4
    assert printed["candidate_count"] == 12
    assert printed["feature_cache_count"] == 2
    assert (output / MANIFEST_FILENAME).is_file()
    assert not list(output.glob(".*.tmp"))
    assert not list(summary_path.parent.glob(".*.tmp"))
    assert not candidate_cli._artifact_lock_path(output.resolve()).exists()
    assert not candidate_cli._artifact_lock_path(summary_path.resolve()).exists()

    bank = EventBank.load(bank_path)
    restored = CandidateCache.load(output)
    restored.validate_against_event_bank(bank)
    assert restored.manifest is not None
    assert restored.manifest.query_corpus_hash == printed["query_corpus_hash"]
    assert dict(restored.manifest.query_key_encoder) == dict(bank.manifest.encoder)
    assert restored.manifest.event_bank_manifest_hash == sha256_file(
        bank_path / EVENT_BANK_MANIFEST_FILENAME
    )
    assert restored.manifest.event_bank_content_hash == canonical_event_bank_content_hash(
        bank.manifest.content_hashes
    )


def test_cli_combines_repeated_inputs_and_resolves_relative_feature_lists(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bank_path, first_query, second_query = _bank_and_queries(tmp_path)
    lists = tmp_path / "lists"
    lists.mkdir()
    feature_list = lists / "queries.txt"
    relative_second = Path("..") / "queries" / second_query.name
    feature_list.write_text(
        f"# generated query set\n\n{relative_second.as_posix()}\n",
        encoding="utf-8-sig",
    )
    output = tmp_path / "combined-candidates"

    assert (
        main(
            [
                "--bank",
                str(bank_path),
                *_data_args(bank_path),
                "--feature-cache",
                str(first_query),
                "--feature-cache",
                str(first_query),
                "--feature-list",
                str(feature_list),
                "--output",
                str(output),
                "--action-horizon",
                "4",
                "--query-stride",
                "4",
                "--top-k",
                "2",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["feature_cache_count"] == 2
    assert summary["query_count"] == 4
    assert summary["candidate_count"] == 8

    with pytest.raises(FileExistsError, match="already exists"):
        main(
            [
                "--bank",
                str(bank_path),
                *_data_args(bank_path),
                "--feature-list",
                str(feature_list),
                "--feature-cache",
                str(first_query),
                "--output",
                str(output),
                "--query-stride",
                "4",
                "--top-k",
                "2",
            ]
        )


def test_cli_rejects_an_empty_feature_list(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bank_path, _, _ = _bank_and_queries(tmp_path)
    feature_list = tmp_path / "empty.txt"
    feature_list.write_text("# no query caches\n\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--bank",
                str(bank_path),
                *_data_args(bank_path),
                "--feature-list",
                str(feature_list),
                "--output",
                str(tmp_path / "output"),
                "--query-stride",
                "4",
                "--top-k",
                "2",
            ]
        )
    assert exc_info.value.code == 2
    assert "at least one --feature-cache" in capsys.readouterr().err


@pytest.mark.parametrize("summary_root", ["bank", "output"])
def test_cli_rejects_summary_under_artifact_trees_even_with_overwrite(
    tmp_path: Path,
    summary_root: str,
) -> None:
    bank_path, query_path, second_query = _bank_and_queries(tmp_path)
    output = tmp_path / "candidates"
    summary_parent = bank_path if summary_root == "bank" else output
    summary_path = summary_parent / "unsafe-summary.json"
    original_bank_hash = sha256_file(bank_path / EVENT_BANK_MANIFEST_FILENAME)

    with pytest.raises(CandidateCacheBuildError, match="outside both"):
        main(
            [
                "--bank",
                str(bank_path),
                *_data_args(bank_path),
                "--feature-cache",
                str(query_path),
                "--feature-cache",
                str(second_query),
                "--output",
                str(output),
                "--query-stride",
                "4",
                "--top-k",
                "2",
                "--summary",
                str(summary_path),
                "--overwrite",
            ]
        )

    assert sha256_file(bank_path / EVENT_BANK_MANIFEST_FILENAME) == original_bank_hash
    assert not (output / MANIFEST_FILENAME).exists()
    assert not summary_path.exists()


def test_cli_rejects_candidate_output_inside_immutable_bank(
    tmp_path: Path,
) -> None:
    bank_path, query_path, second_query = _bank_and_queries(tmp_path)
    output = bank_path / "candidate-cache"
    original_bank_hash = sha256_file(bank_path / EVENT_BANK_MANIFEST_FILENAME)

    with pytest.raises(CandidateCacheBuildError, match="outside the immutable"):
        main(
            [
                "--bank",
                str(bank_path),
                *_data_args(bank_path),
                "--feature-cache",
                str(query_path),
                "--feature-cache",
                str(second_query),
                "--output",
                str(output),
                "--query-stride",
                "4",
                "--top-k",
                "2",
                "--overwrite",
            ]
        )

    assert sha256_file(bank_path / EVENT_BANK_MANIFEST_FILENAME) == original_bank_hash
    assert not output.exists()


def test_cli_overwrite_never_replaces_feature_input(tmp_path: Path) -> None:
    bank_path, query_path, second_query = _bank_and_queries(tmp_path)
    before = query_path.read_bytes()

    with pytest.raises(CandidateCacheBuildError, match="feature-cache"):
        main(
            [
                "--bank",
                str(bank_path),
                *_data_args(bank_path),
                "--feature-cache",
                str(query_path),
                "--feature-cache",
                str(second_query),
                "--output",
                str(tmp_path / "candidates"),
                "--query-stride",
                "4",
                "--top-k",
                "2",
                "--summary",
                str(query_path),
                "--overwrite",
            ]
        )

    assert query_path.read_bytes() == before
    assert not (tmp_path / "candidates").exists()


def test_cli_rejects_event_bank_manifest_change_during_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank_path, query_path, second_query = _bank_and_queries(tmp_path)
    output = tmp_path / "candidates"
    manifest_path = bank_path / EVENT_BANK_MANIFEST_FILENAME
    original_builder = candidate_cli.build_candidate_cache_from_collection

    def mutate_manifest_after_generation(*args: object, **kwargs: object) -> CandidateCache:
        cache = original_builder(*args, **kwargs)
        manifest_path.write_text(
            manifest_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        return cache

    monkeypatch.setattr(
        candidate_cli,
        "build_candidate_cache_from_collection",
        mutate_manifest_after_generation,
    )
    with pytest.raises(CandidateCacheBuildError, match="being generated"):
        main(
            [
                "--bank",
                str(bank_path),
                *_data_args(bank_path),
                "--feature-cache",
                str(query_path),
                "--feature-cache",
                str(second_query),
                "--output",
                str(output),
                "--query-stride",
                "4",
                "--top-k",
                "2",
            ]
        )

    assert not (output / MANIFEST_FILENAME).exists()


@pytest.mark.parametrize("locked_target", ["output", "summary"])
def test_cli_rejects_a_preclaimed_publication_target_without_writing(
    tmp_path: Path,
    locked_target: str,
) -> None:
    bank_path, query_path, second_query = _bank_and_queries(tmp_path)
    output = tmp_path / "candidates"
    summary_path = tmp_path / "reports" / "candidate-summary.json"
    target = output if locked_target == "output" else summary_path
    lock_path = candidate_cli._artifact_lock_path(target.resolve())
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with artifact_claim(lock_path, purpose=f"competing {locked_target} writer"):
        with pytest.raises(ArtifactAlreadyClaimedError, match="competing"):
            main(
                [
                    "--bank",
                    str(bank_path),
                    *_data_args(bank_path),
                    "--feature-cache",
                    str(query_path),
                    "--feature-cache",
                    str(second_query),
                    "--output",
                    str(output),
                    "--query-stride",
                    "4",
                    "--top-k",
                    "2",
                    "--summary",
                    str(summary_path),
                    "--overwrite",
                ]
            )

        assert lock_path.exists()
        assert not (output / MANIFEST_FILENAME).exists()
        assert not summary_path.exists()

        other_target = summary_path if locked_target == "output" else output
        other_lock = candidate_cli._artifact_lock_path(other_target.resolve())
        assert not other_lock.exists()

    assert not lock_path.exists()
