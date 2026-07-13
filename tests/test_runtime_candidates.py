from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastwam.memory.action_contract import ActionSpaceContract
from fastwam.memory.candidate_cache import (
    MANIFEST_FILENAME as CANDIDATE_MANIFEST_FILENAME,
    CachedCandidate,
    CandidateCache,
    EpisodeLeakageError,
    QueryId,
    canonical_event_bank_content_hash,
)
from fastwam.memory.event_bank import MANIFEST_FILENAME, EventBank
from fastwam.memory.manifest import sha256_file
from fastwam.memory.payload_names import (
    CONTAINS_FORCED_GRIPPER,
    EFFECT_POST,
    EFFECT_PRE,
    EVENT_SCORE,
    FEATURE_EPISODE_SHA256,
    MODEL_SPACE_ACTION,
    OBSERVED_GRIPPER_STATE,
    SOURCE_EPISODE_SHA256,
    START_PROPRIO,
    TASK_INDEX,
)
from fastwam.memory.runtime_candidates import (
    INVALID_BANK_ROW,
    ResolvedCandidateRow,
    RuntimeCandidateContractError,
    RuntimeCandidateGatherError,
    RuntimeCandidateResolver,
)
from fastwam.memory.schema import EventId


CATALOG_HASH = "a" * 64
AUDIT_HASH = "b" * 64
QUERY_CORPUS_HASH = "c" * 64
NORMALIZER_HASH = "d" * 64
ENCODER_HASH = "e" * 64
CAMERA_HASH = "f" * 64
STATS_HASH = "1" * 64


def _action_contract() -> ActionSpaceContract:
    return ActionSpaceContract(
        action_dim=3,
        arm_dims=(0, 1),
        gripper_dims=(2,),
        gripper_threshold=0.0,
        normalization_mode="min/max",
        normalization_stats_sha256=STATS_HASH,
        control_mode="delta_eef_gripper",
        embodiment="libero_panda",
    )


def _hash_rows(values: tuple[int, ...]) -> np.ndarray:
    return np.stack(
        [np.frombuffer(bytes.fromhex(f"{value:064x}"), dtype=np.uint8) for value in values]
    )


def _write_artifacts(
    tmp_path: Path,
    *,
    catalog_hash: str = CATALOG_HASH,
    audit_hash: str = AUDIT_HASH,
    action_contract: ActionSpaceContract | None = None,
    actions: np.ndarray | None = None,
) -> tuple[Path, Path, tuple[EventId, ...]]:
    if action_contract is None:
        action_contract = _action_contract()
    events = tuple(EventId("libero", 0, episode, 0) for episode in range(3))
    count = len(events)
    if actions is None:
        actions = np.arange(count * 4 * 3, dtype=np.float32).reshape(count, 4, 3)
    else:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (count, 4, action_contract.action_dim):
            raise ValueError("test actions must have shape [3, 4, action_dim]")
    bank = EventBank.from_arrays(
        events,
        np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32
        ),
        **{
            MODEL_SPACE_ACTION: actions,
            EFFECT_PRE: np.zeros((count, 2, 2), dtype=np.float32),
            EFFECT_POST: np.ones((count, 2, 2), dtype=np.float32),
            START_PROPRIO: np.zeros((count, 8), dtype=np.float32),
            OBSERVED_GRIPPER_STATE: np.zeros((count, 5), dtype=np.float32),
            TASK_INDEX: np.zeros((count,), dtype=np.int64),
            EVENT_SCORE: np.ones((count,), dtype=np.float32),
            CONTAINS_FORCED_GRIPPER: np.zeros((count,), dtype=np.bool_),
            SOURCE_EPISODE_SHA256: _hash_rows((1, 2, 3)),
            FEATURE_EPISODE_SHA256: _hash_rows((11, 12, 13)),
        },
    )
    bank_path = tmp_path / "bank"
    manifest = bank.save(
        bank_path,
        action_normalizer={
            "file_sha256": NORMALIZER_HASH,
            "contract": action_contract.to_dict(),
        },
        encoder={"file_sha256": ENCODER_HASH, "name": "dino-test"},
        camera_layout={"file_sha256": CAMERA_HASH},
        provenance={
            "split": "train",
            "feature_collection_sha256": QUERY_CORPUS_HASH,
            "data_binding": {
                "catalog_sha256": catalog_hash,
                "audit_report_sha256": audit_hash,
                "split": "train",
            },
        },
    )
    manifest_hash = sha256_file(bank_path / MANIFEST_FILENAME)
    content_hash = canonical_event_bank_content_hash(manifest.content_hashes)

    queries = (
        QueryId("libero", 0, 0, 0),
        QueryId("libero", 0, 0, 1),
    )
    cache = CandidateCache(
        queries,
        (
            (
                CachedCandidate(events[1], 0.8),
                CachedCandidate(events[2], 0.7),
            ),
            (),
        ),
    )
    cache_path = tmp_path / "candidates"
    cache.save(
        cache_path,
        event_bank_manifest_hash=manifest_hash,
        event_bank_content_hash=content_hash,
        query_corpus_hash=QUERY_CORPUS_HASH,
        query_key_encoder=manifest.encoder,
        build_recipe={
            "implementation": "exact_cosine_v1",
            "action_horizon": 4,
            "query_stride": 1,
            "top_k": 3,
            "query_split": "train",
            "query_data_binding": {
                "catalog_sha256": catalog_hash,
                "audit_report_sha256": audit_hash,
                "split": "train",
            },
            "episode_exclusion": [
                "global_episode_identity",
                "source_episode_sha256",
                "feature_episode_sha256",
            ],
        },
    )
    return bank_path, cache_path, events


def _resolver(tmp_path: Path) -> tuple[RuntimeCandidateResolver, tuple[EventId, ...]]:
    bank_path, cache_path, events = _write_artifacts(tmp_path)
    resolver = RuntimeCandidateResolver.from_artifacts(
        bank_path,
        cache_path,
        expected_query_split="train",
        expected_query_corpus_sha256=QUERY_CORPUS_HASH,
        expected_action_space=_action_contract(),
    )
    return resolver, events


def test_resolves_fixed_width_rows_and_safely_gathers_actions(tmp_path: Path) -> None:
    bank_path, cache_path, events = _write_artifacts(tmp_path)
    resolver = RuntimeCandidateResolver.from_artifacts(
        bank_path,
        cache_path,
        expected_query_split="train",
        expected_query_corpus_sha256=QUERY_CORPUS_HASH,
        expected_action_space=_action_contract(),
    )
    row = resolver.resolve(QueryId("libero", 0, 0, 0))

    assert resolver.fixed_k == 3
    assert resolver.query_split == "train"
    assert resolver.query_stride == 1
    assert resolver.action_horizon == 4
    assert resolver.bank_manifest_sha256 == sha256_file(
        bank_path / MANIFEST_FILENAME
    )
    assert resolver.candidate_manifest_sha256 == sha256_file(
        cache_path / CANDIDATE_MANIFEST_FILENAME
    )
    assert resolver.build_recipe["query_data_binding"]["split"] == "train"
    with pytest.raises(TypeError):
        resolver.build_recipe["query_data_binding"]["split"] = "dev"
    assert row.bank_rows.tolist() == [1, 2, INVALID_BANK_ROW]
    assert row.mask.tolist() == [True, True, False]
    assert row.cosine_scores.tolist() == pytest.approx([0.8, 0.7, 0.0])
    assert row.event_ids == (events[1], events[2], None)
    assert row.valid_bank_rows.tolist() == [1, 2]
    assert not row.bank_rows.flags.writeable

    actions = resolver.gather_model_actions(row)
    assert actions.shape == (3, 4, 3)
    assert np.array_equal(actions[0], np.arange(12, 24, dtype=np.float32).reshape(4, 3))
    assert np.array_equal(actions[1], np.arange(24, 36, dtype=np.float32).reshape(4, 3))
    assert np.count_nonzero(actions[2]) == 0
    assert not actions.flags.writeable


def test_missing_and_existing_empty_rows_have_explicit_semantics(tmp_path: Path) -> None:
    resolver, _ = _resolver(tmp_path)
    missing = QueryId("libero", 0, 0, 99)
    with pytest.raises(KeyError, match="no exact row"):
        resolver.resolve(missing)

    explicit_missing = resolver.resolve(missing, allow_missing=True)
    existing_empty = resolver.resolve(QueryId("libero", 0, 0, 1))
    for row in (explicit_missing, existing_empty):
        assert row.valid_count == 0
        assert row.bank_rows.tolist() == [INVALID_BANK_ROW] * 3
        assert not row.mask.any()
        # This must return zero padding without ever indexing bank[-1].
        assert np.count_nonzero(resolver.gather_model_actions(row)) == 0


def test_invalid_valid_row_is_rejected_before_numpy_gather(tmp_path: Path) -> None:
    resolver, events = _resolver(tmp_path)
    invalid = ResolvedCandidateRow(
        query_id=QueryId("libero", 0, 0, 0),
        bank_rows=np.asarray([99, INVALID_BANK_ROW, INVALID_BANK_ROW], dtype=np.int64),
        mask=np.asarray([True, False, False], dtype=np.bool_),
        cosine_scores=np.asarray([0.8, 0.0, 0.0], dtype=np.float32),
        event_ids=(events[1], None, None),
    )
    with pytest.raises(RuntimeCandidateGatherError, match="out-of-range"):
        resolver.gather_model_actions(invalid)

    with pytest.raises(RuntimeCandidateContractError, match="non-negative"):
        ResolvedCandidateRow(
            query_id=QueryId("libero", 0, 0, 0),
            bank_rows=np.asarray(
                [INVALID_BANK_ROW, INVALID_BANK_ROW, INVALID_BANK_ROW],
                dtype=np.int64,
            ),
            mask=np.asarray([True, False, False], dtype=np.bool_),
            cosine_scores=np.asarray([0.8, 0.0, 0.0], dtype=np.float32),
            event_ids=(events[1], None, None),
        )


def test_resolver_rechecks_same_episode_isolation(tmp_path: Path) -> None:
    bank_path, cache_path, events = _write_artifacts(tmp_path)
    bank = EventBank.load(bank_path)
    cache = CandidateCache.load(cache_path)
    cache._candidates = (
        (CachedCandidate(events[0], 0.9),),
        (),
    )

    with pytest.raises(EpisodeLeakageError, match="complete episode"):
        RuntimeCandidateResolver(
            bank,
            cache,
            event_bank_manifest_sha256=sha256_file(bank_path / MANIFEST_FILENAME),
            expected_query_split="train",
            expected_query_corpus_sha256=QUERY_CORPUS_HASH,
        )


def test_resolver_rejects_wrong_bank_snapshot_hash(tmp_path: Path) -> None:
    bank_path, cache_path, _ = _write_artifacts(tmp_path)
    bank = EventBank.load(bank_path)
    cache = CandidateCache.load(cache_path)

    with pytest.raises(RuntimeCandidateContractError, match="different event-bank manifest"):
        RuntimeCandidateResolver(
            bank,
            cache,
            event_bank_manifest_sha256="0" * 64,
            expected_query_split="train",
        )
