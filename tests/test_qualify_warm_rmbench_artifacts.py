from __future__ import annotations

import numpy as np
import pytest

from scripts.qualify_warm_rmbench_artifacts import (
    PhaseRecallThreshold,
    QualificationError,
    evaluate_phase_recall,
    validate_teacher_forced_parity,
    validate_temporal_bank,
)
from fastwam.memory.bank_builder import EpisodeFeatures
from fastwam.memory.candidate_cache import CachedCandidate, CandidateCache, QueryId
from fastwam.memory.event_bank import EventBank
from fastwam.memory.feature_cache import FeatureCacheMetadata, save_episode_feature_cache
from fastwam.memory.offline_pipeline import (
    fixed_horizon_query_starts,
    load_feature_cache_collection,
)
from fastwam.memory.payload_names import (
    ACTION_VALID_MASK,
    CONTAINS_FORCED_GRIPPER,
    EFFECT_POST,
    EFFECT_PRE,
    EVENT_ORDINAL,
    EVENT_SCORE,
    FEATURE_EPISODE_SHA256,
    MODEL_SPACE_ACTION,
    NORMALIZED_PHASE,
    OBSERVED_GRIPPER_STATE,
    SOURCE_EPISODE_SHA256,
    START_PROPRIO,
    SUCCESSOR_ROW,
    SUCCESSOR_EVENT_START_FRAME,
    TASK_INDEX,
)
from fastwam.memory.schema import EventId


HASHES = {
    "catalog_hash": "a" * 64,
    "normalizer_hash": "b" * 64,
    "encoder_hash": "c" * 64,
    "camera_hash": "d" * 64,
}


def _digest_rows(values: list[int]) -> np.ndarray:
    return np.stack(
        [np.frombuffer(bytes.fromhex(f"{value:064x}"), dtype=np.uint8) for value in values]
    )


def _context() -> np.ndarray:
    # Three factual phases with repeated endpoints to form T+1 states.
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _episode(index: int, *, dataset_id: str = "rmbench") -> EpisodeFeatures:
    return EpisodeFeatures(
        dataset_id=dataset_id,
        dataset_index=0,
        episode_index=index,
        task_index=0,
        source_episode_sha256=f"{index + 100:064x}",
        model_actions=np.full((6, 2), float(index), dtype=np.float32),
        proprio=np.full((7, 3), float(index), dtype=np.float32),
        gripper=np.zeros((7,), dtype=np.float32),
        context_keys=_context(),
        semantic_features=np.full((7, 2, 3), float(index), dtype=np.float32),
    )


def _bank(
    *,
    corrupt_successor: bool = False,
    sparse: bool = False,
    ambiguous_keys: bool = False,
) -> EventBank:
    event_ids: list[EventId] = []
    keys: list[np.ndarray] = []
    phases: list[float] = []
    ordinals: list[int] = []
    successors: list[int] = []
    successor_starts: list[int] = []
    source_values: list[int] = []
    feature_values: list[int] = []
    starts = [0, 4] if sparse else [0, 2, 4]
    for episode in range(3):
        base = len(event_ids)
        for ordinal, start in enumerate(starts):
            event_ids.append(EventId("rmbench", 0, episode, start))
            keys.append(
                np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                if ambiguous_keys
                else _context()[start]
            )
            phases.append(float(start) / 4.0)
            ordinals.append(ordinal)
            successors.append(base + ordinal + 1 if ordinal + 1 < len(starts) else -1)
            successor_starts.append(starts[ordinal + 1] if ordinal + 1 < len(starts) else -1)
            source_values.append(episode + 1)
            feature_values.append(episode + 11)
    if corrupt_successor:
        successors[0] = -1
    count = len(event_ids)
    return EventBank(
        event_ids,
        np.asarray(keys, dtype=np.float32),
        payloads={
            MODEL_SPACE_ACTION: np.zeros((count, 2, 2), dtype=np.float32),
            EFFECT_PRE: np.zeros((count, 2, 3), dtype=np.float32),
            EFFECT_POST: np.ones((count, 2, 3), dtype=np.float32),
            START_PROPRIO: np.zeros((count, 3), dtype=np.float32),
            OBSERVED_GRIPPER_STATE: np.zeros((count, 3), dtype=np.float32),
            TASK_INDEX: np.zeros((count,), dtype=np.int64),
            EVENT_SCORE: np.ones((count,), dtype=np.float32),
            CONTAINS_FORCED_GRIPPER: np.zeros((count,), dtype=np.bool_),
            NORMALIZED_PHASE: np.asarray(phases, dtype=np.float32),
            EVENT_ORDINAL: np.asarray(ordinals, dtype=np.int64),
            SUCCESSOR_ROW: np.asarray(successors, dtype=np.int64),
            SUCCESSOR_EVENT_START_FRAME: np.asarray(successor_starts, dtype=np.int64),
            ACTION_VALID_MASK: np.ones((count, 2), dtype=np.bool_),
            SOURCE_EPISODE_SHA256: _digest_rows(source_values),
            FEATURE_EPISODE_SHA256: _digest_rows(feature_values),
        },
    )


def _dev_collection(tmp_path):
    paths = []
    for index in (20, 21):
        episode = _episode(index)
        path = tmp_path / f"dev-{index}.npz"
        save_episode_feature_cache(
            path,
            episode,
            metadata=FeatureCacheMetadata.for_episode(
                episode, split="dev", **HASHES
            ),
        )
        paths.append(path)
    return load_feature_cache_collection(paths)


def test_temporal_bank_accepts_dense_exact_chains_and_rejects_broken_links() -> None:
    report = validate_temporal_bank(_bank(), max_event_stride=2)
    assert report["episode_count"] == 3
    assert report["max_observed_event_stride"] == 2

    with pytest.raises(QualificationError, match="successor_row"):
        validate_temporal_bank(_bank(corrupt_successor=True), max_event_stride=2)
    with pytest.raises(QualificationError, match="exceeding max_event_stride"):
        validate_temporal_bank(_bank(sparse=True), max_event_stride=2)


def test_phase_recall_is_stratified_and_fails_when_k_cannot_cover_phase(tmp_path) -> None:
    bank = _bank()
    collection = _dev_collection(tmp_path)
    report = evaluate_phase_recall(
        bank,
        collection,
        action_horizon=2,
        query_stride=2,
        thresholds=[PhaseRecallThreshold(3, 1.0)],
        phase_tolerance=0.01,
    )
    assert report["query_count"] == 6
    assert report["reports"]["3"]["overall"] == 1.0
    assert set(report["reports"]["3"]["per_stratum"]) == {
        "early",
        "middle",
        "late",
    }
    assert report["per_task"]["0"]["reports"]["3"]["overall"] == 1.0

    with pytest.raises(QualificationError, match="phase-compatible candidate gate"):
        evaluate_phase_recall(
            _bank(ambiguous_keys=True),
            collection,
            action_horizon=2,
            query_stride=2,
            thresholds=[PhaseRecallThreshold(1, 1.0)],
            phase_tolerance=0.01,
        )


def test_teacher_forced_parity_detects_candidate_identity_drift(tmp_path) -> None:
    bank = _bank()
    collection = _dev_collection(tmp_path)
    query_ids = []
    candidate_rows = []
    for record in collection.records:
        episode = record.features
        for start in fixed_horizon_query_starts(6, action_horizon=2, stride=2):
            query_ids.append(QueryId("rmbench", 0, episode.episode_index, start))
            results = bank.search(
                episode.context_keys[start],
                top_k=3,
                exclude_episode=("rmbench", 0, episode.episode_index),
                exclude_source_episode_sha256=episode.source_episode_sha256,
                exclude_feature_episode_sha256=episode.feature_episode_sha256,
            )
            candidate_rows.append(
                [CachedCandidate(result.event_id, result.score) for result in results]
            )
    cache = CandidateCache(query_ids, candidate_rows)
    cache.save(
        tmp_path / "cache",
        event_bank_manifest_hash="e" * 64,
        event_bank_content_hash="f" * 64,
        query_corpus_hash=collection.content_hash,
        query_key_encoder={"name": "test", "version": 1},
        build_recipe={
            "implementation": "exact_cosine_v1",
            "action_horizon": 2,
            "query_stride": 2,
            "top_k": 3,
            "query_split": "dev",
        },
    )
    report = validate_teacher_forced_parity(
        bank, collection, cache, max_queries=100
    )
    assert report["checked_query_count"] == len(cache)

    # Simulate integration code replacing one immutable candidate row.
    rows = [list(row) for row in cache.candidates]
    rows[0] = list(reversed(rows[0]))
    cache._candidates = tuple(tuple(row) for row in rows)
    with pytest.raises(QualificationError, match="identity mismatch"):
        validate_teacher_forced_parity(bank, collection, cache, max_queries=100)
