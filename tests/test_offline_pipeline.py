from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastwam.datasets.lerobot.audit import load_audit_report
from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog
from fastwam.memory.bank_builder import EpisodeFeatures
from fastwam.memory.candidate_cache import QueryId
from fastwam.memory.event_mining import EventMiningConfig
from fastwam.memory.feature_cache import FeatureCacheMetadata, save_episode_feature_cache
from fastwam.memory.offline_pipeline import (
    FeatureDataBinding,
    OfflinePipelineError,
    assert_feature_collections_disjoint,
    build_candidate_cache_from_collection,
    build_event_bank_from_collection,
    build_oracle_queries_from_collection,
    factual_state_query_starts,
    fixed_horizon_query_starts,
    load_feature_cache_collection,
    validate_feature_collection_against_catalog,
    validate_query_collection_against_bank,
)
from fastwam.memory.oracle_metrics import ActionDistanceConfig, evaluate_oracle_retrieval
from tests.warm_test_data import write_test_catalog_and_audit


HASHES = {
    "catalog_hash": "a" * 64,
    "normalizer_hash": "b" * 64,
    "encoder_hash": "c" * 64,
    "camera_hash": "d" * 64,
}


def _episode(index: int, *, source_hash: str | None = None) -> EpisodeFeatures:
    steps = 8
    actions = np.zeros((steps, 3), dtype=np.float32)
    actions[:, :2] = float(index)
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
        source_episode_sha256=source_hash or f"{index + 1:064x}",
        model_actions=actions,
        proprio=np.full((steps + 1, 8), float(index), dtype=np.float32),
        gripper=np.concatenate(
            [np.zeros(4, dtype=np.float32), np.ones(steps + 1 - 4, dtype=np.float32)]
        ),
        context_keys=context,
        semantic_features=semantic,
    )


def _save(
    root: Path,
    episode: EpisodeFeatures,
    *,
    encoder_hash: str = "c" * 64,
    split: str = "train",
    catalog_hash: str = "a" * 64,
) -> Path:
    path = root / f"episode-{episode.episode_index}.npz"
    metadata = FeatureCacheMetadata.for_episode(
        episode,
        split=split,
        **{
            **HASHES,
            "catalog_hash": catalog_hash,
            "encoder_hash": encoder_hash,
        },
    )
    save_episode_feature_cache(path, episode, metadata=metadata)
    return path


def _binding(*, split: str = "train", catalog_hash: str = "a" * 64) -> FeatureDataBinding:
    return FeatureDataBinding(
        catalog_sha256=catalog_hash,
        audit_report_sha256="f" * 64,
        split=split,
    )


def test_end_to_end_cache_bank_candidates_and_oracle(tmp_path: Path) -> None:
    train = load_feature_cache_collection(
        [_save(tmp_path / "train", _episode(0)), _save(tmp_path / "train", _episode(1))]
    )
    dev = load_feature_cache_collection(
        [_save(tmp_path / "dev", _episode(2), split="dev")]
    )
    assert_feature_collections_disjoint(train, dev)
    assert train.split == "train"
    assert dev.split == "dev"

    bank, summary, provenance = build_event_bank_from_collection(
        train,
        mining_config=EventMiningConfig(action_horizon=4),
        start_mode="uniform",
        data_binding=_binding(),
    )
    assert summary.num_events == 4
    assert provenance["split"] == "train"
    assert provenance["feature_collection_sha256"] == train.content_hash

    bank.save(
        tmp_path / "bank",
        action_normalizer={"file_sha256": HASHES["normalizer_hash"]},
        encoder={"file_sha256": HASHES["encoder_hash"]},
        camera_layout={"file_sha256": HASHES["camera_hash"]},
        provenance=provenance,
    )
    loaded_bank = type(bank).load(tmp_path / "bank")
    validate_query_collection_against_bank(loaded_bank, dev)

    candidates = build_candidate_cache_from_collection(
        loaded_bank,
        dev,
        action_horizon=4,
        query_stride=4,
        top_k=3,
    )
    assert len(candidates) == 2
    assert all(row for row in candidates.candidates)
    assert all(
        candidate.event_id.episode_key != query.episode_key
        for query, row in zip(candidates.query_ids, candidates.candidates, strict=True)
        for candidate in row
    )

    queries = build_oracle_queries_from_collection(
        dev, action_horizon=4, query_stride=4
    )
    report = evaluate_oracle_retrieval(
        loaded_bank,
        queries,
        ActionDistanceConfig(
            action_dim=3,
            arm_dims=(0, 1),
            gripper_dims=(2,),
        ),
        top_k=3,
    )
    assert report.query_count == 2
    assert report.coverage == 1.0


def test_train_candidate_cache_requires_stride_one_and_excludes_query_episode(
    tmp_path: Path,
) -> None:
    train = load_feature_cache_collection(
        [
            _save(tmp_path / "train", _episode(0)),
            _save(tmp_path / "train", _episode(1)),
        ]
    )
    bank, _, provenance = build_event_bank_from_collection(
        train,
        mining_config=EventMiningConfig(action_horizon=4),
        start_mode="uniform",
        data_binding=_binding(),
    )
    bank.save(
        tmp_path / "bank",
        action_normalizer={"file_sha256": HASHES["normalizer_hash"]},
        encoder={"file_sha256": HASHES["encoder_hash"]},
        camera_layout={"file_sha256": HASHES["camera_hash"]},
        provenance=provenance,
    )
    loaded_bank = type(bank).load(tmp_path / "bank")

    cache = build_candidate_cache_from_collection(
        loaded_bank,
        train,
        action_horizon=4,
        query_stride=1,
        top_k=4,
        query_split="train",
    )
    assert len(cache) == 10
    assert all(
        candidate.event_id.episode_key != query.episode_key
        for query, row in zip(cache.query_ids, cache.candidates, strict=True)
        for candidate in row
    )

    with pytest.raises(OfflinePipelineError, match="query_stride=1"):
        build_candidate_cache_from_collection(
            loaded_bank,
            train,
            action_horizon=4,
            query_stride=2,
            top_k=4,
            query_split="train",
        )


def test_collection_rejects_contract_and_content_leakage(tmp_path: Path) -> None:
    first_path = _save(tmp_path / "first", _episode(0))
    mismatched = _save(
        tmp_path / "mismatch", _episode(1), encoder_hash="e" * 64
    )
    with pytest.raises(OfflinePipelineError, match="different model/data contract"):
        load_feature_cache_collection([first_path, mismatched])

    duplicate_source = _save(
        tmp_path / "duplicate",
        _episode(2, source_hash=_episode(0).source_episode_sha256),
        split="dev",
    )
    first = load_feature_cache_collection([first_path])
    duplicate = load_feature_cache_collection([duplicate_source])
    with pytest.raises(OfflinePipelineError, match="source episode content"):
        assert_feature_collections_disjoint(first, duplicate)

    mixed_split = _save(tmp_path / "mixed", _episode(3), split="dev")
    with pytest.raises(OfflinePipelineError, match="mixes splits"):
        load_feature_cache_collection([first_path, mixed_split])


def test_event_bank_requires_train_collection(tmp_path: Path) -> None:
    dev = load_feature_cache_collection(
        [_save(tmp_path / "dev", _episode(2), split="dev")]
    )

    with pytest.raises(OfflinePipelineError, match="must use split 'train'"):
        build_event_bank_from_collection(
            dev,
            mining_config=EventMiningConfig(action_horizon=4),
            start_mode="uniform",
            data_binding=_binding(),
        )


def test_bank_query_validation_rejects_encoded_content_overlap(
    tmp_path: Path,
) -> None:
    source = _episode(0)
    duplicate = EpisodeFeatures(
        dataset_id=source.dataset_id,
        dataset_index=source.dataset_index,
        episode_index=99,
        task_index=source.task_index,
        source_episode_sha256="f" * 64,
        model_actions=source.model_actions,
        proprio=source.proprio,
        gripper=source.gripper,
        context_keys=source.context_keys,
        semantic_features=source.semantic_features,
    )
    train = load_feature_cache_collection(
        [_save(tmp_path / "train", source, split="train")]
    )
    dev = load_feature_cache_collection(
        [_save(tmp_path / "dev", duplicate, split="dev")]
    )
    assert train.episode_content_hashes == dev.episode_content_hashes
    assert train.source_episode_hashes.isdisjoint(dev.source_episode_hashes)

    bank, _, provenance = build_event_bank_from_collection(
        train,
        mining_config=EventMiningConfig(action_horizon=4),
        start_mode="uniform",
        data_binding=_binding(),
    )
    bank.save(
        tmp_path / "bank",
        action_normalizer={"file_sha256": HASHES["normalizer_hash"]},
        encoder={"file_sha256": HASHES["encoder_hash"]},
        camera_layout={"file_sha256": HASHES["camera_hash"]},
        provenance=provenance,
    )

    with pytest.raises(OfflinePipelineError, match="encoded episode content"):
        validate_query_collection_against_bank(type(bank).load(tmp_path / "bank"), dev)


def test_fixed_horizon_query_starts_are_tail_complete() -> None:
    assert fixed_horizon_query_starts(3, action_horizon=4, stride=2) == ()
    assert fixed_horizon_query_starts(10, action_horizon=4, stride=4) == (0, 4, 6)
    assert factual_state_query_starts(0, stride=2) == ()
    assert factual_state_query_starts(9, stride=4) == (0, 4, 8)


def test_candidate_cache_can_cover_every_factual_state_including_terminal(
    tmp_path: Path,
) -> None:
    train = load_feature_cache_collection(
        [
            _save(tmp_path / "train", _episode(0)),
            _save(tmp_path / "train", _episode(1)),
        ]
    )
    bank, _, provenance = build_event_bank_from_collection(
        train,
        mining_config=EventMiningConfig(action_horizon=4),
        start_mode="uniform",
        data_binding=_binding(),
    )
    bank.save(
        tmp_path / "bank",
        action_normalizer={"file_sha256": HASHES["normalizer_hash"]},
        encoder={"file_sha256": HASHES["encoder_hash"]},
        camera_layout={"file_sha256": HASHES["camera_hash"]},
        provenance=provenance,
    )
    cache = build_candidate_cache_from_collection(
        type(bank).load(tmp_path / "bank"),
        train,
        action_horizon=4,
        query_stride=1,
        top_k=4,
        query_split="train",
        include_partial_action_queries=True,
    )
    assert len(cache) == 18
    assert QueryId("libero", 0, 0, 8) in cache.query_ids
    assert QueryId("libero", 0, 1, 8) in cache.query_ids
    assert all(row for row in cache.candidates)


def test_catalog_audit_is_the_split_authority(tmp_path: Path) -> None:
    train_episode = _episode(0)
    dev_episode = _episode(1)
    catalog_path, audit_path, catalog, _ = write_test_catalog_and_audit(
        tmp_path / "binding",
        [(train_episode, "train"), (dev_episode, "dev")],
    )
    assert EpisodeCatalog.load(catalog_path) == catalog
    audit = load_audit_report(audit_path)

    train = load_feature_cache_collection(
        [
            _save(
                tmp_path / "verified-train",
                train_episode,
                split="train",
                catalog_hash=catalog.content_sha256,
            )
        ]
    )
    binding = validate_feature_collection_against_catalog(
        train,
        catalog,
        audit,
        expected_split="train",
    )
    assert binding.catalog_sha256 == catalog.content_sha256

    falsely_labeled_dev = load_feature_cache_collection(
        [
            _save(
                tmp_path / "false-train",
                dev_episode,
                split="train",
                catalog_hash=catalog.content_sha256,
            )
        ]
    )
    with pytest.raises(OfflinePipelineError, match="not catalog-derived"):
        validate_feature_collection_against_catalog(
            falsely_labeled_dev,
            catalog,
            audit,
            expected_split="train",
        )
