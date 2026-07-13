from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fastwam.memory.candidate_cache import (
    CANDIDATE_CACHE_SCHEMA,
    MANIFEST_FILENAME,
    CachedCandidate,
    CandidateCache,
    CandidateCacheContractError,
    CandidateCacheIntegrityError,
    CandidateCacheManifest,
    CandidateCacheManifestError,
    EpisodeLeakageError,
    QueryId,
    canonical_event_bank_content_hash,
)
from fastwam.memory.event_bank import EventBank
from fastwam.memory.schema import EventId


BANK_MANIFEST_HASH = "a" * 64
BANK_CONTENT_HASH = "b" * 64
QUERY_CORPUS_HASH = "e" * 64
QUERY_ENCODER = {
    "name": "frozen-dino-text-proprio",
    "version": 1,
    "dimension": 256,
    "normalization": "l2",
}
BUILD_RECIPE = {
    "implementation": "exact_cosine_v1",
    "action_horizon": 16,
    "query_stride": 4,
    "top_k": 32,
    "query_split": "dev",
}


def _candidate(
    dataset_id: str,
    dataset_index: int,
    episode_index: int,
    start_frame: int,
    score: float,
) -> CachedCandidate:
    return CachedCandidate(
        EventId(dataset_id, dataset_index, episode_index, start_frame), score
    )


def _save(cache: CandidateCache, path: Path, *, overwrite: bool = False) -> None:
    cache.save(
        path,
        event_bank_manifest_hash=BANK_MANIFEST_HASH,
        event_bank_content_hash=BANK_CONTENT_HASH,
        query_corpus_hash=QUERY_CORPUS_HASH,
        query_key_encoder=QUERY_ENCODER,
        build_recipe=BUILD_RECIPE,
        overwrite=overwrite,
    )


def _load(path: Path) -> CandidateCache:
    return CandidateCache.load(
        path,
        expected_event_bank_manifest_hash=BANK_MANIFEST_HASH,
        expected_event_bank_content_hash=BANK_CONTENT_HASH,
        expected_query_corpus_hash=QUERY_CORPUS_HASH,
        expected_query_key_encoder=QUERY_ENCODER,
        expected_build_recipe=BUILD_RECIPE,
    )


def test_round_trip_variable_length_csr_and_utf8_ids(tmp_path: Path) -> None:
    queries = [
        QueryId("libero/中文", 0, 4, 12),
        QueryId("robotwin", 1, 8, 3),
        QueryId("robotwin", 1, 9, 5),
    ]
    rows = [
        [
            _candidate("libero/中文", 0, 2, 7, 0.75),
            _candidate("other", 5, 4, 1, -0.25),
        ],
        [],
        [_candidate("robotwin", 1, 10, 6, 0.125)],
    ]
    cache = CandidateCache(queries, rows)

    _save(cache, tmp_path)
    loaded = _load(tmp_path)

    assert loaded.query_ids == tuple(queries)
    assert loaded.candidates == tuple(tuple(row) for row in rows)
    assert loaded.num_candidates == 3
    assert loaded.candidates_for(queries[1]) == ()
    assert loaded.manifest is not None
    assert loaded.manifest.schema == CANDIDATE_CACHE_SCHEMA
    assert loaded.manifest.query_corpus_hash == QUERY_CORPUS_HASH
    assert dict(loaded.manifest.query_key_encoder) == QUERY_ENCODER
    assert dict(loaded.manifest.build_recipe) == BUILD_RECIPE
    assert (tmp_path / MANIFEST_FILENAME).is_file()
    assert not (tmp_path / "manifest.json").exists()

    payload = tmp_path / loaded.manifest.payload_file
    with np.load(payload, allow_pickle=False) as archive:
        assert archive["query_candidate_offsets"].tolist() == [0, 2, 2, 3]
        assert all(not archive[name].dtype.hasobject for name in archive.files)


def test_empty_candidate_rows_and_empty_cache_round_trip(tmp_path: Path) -> None:
    cache = CandidateCache([QueryId("d", 0, 0, 1)], [[]])
    _save(cache, tmp_path / "one-empty")
    loaded = _load(tmp_path / "one-empty")
    assert loaded.candidates == ((),)

    entirely_empty = CandidateCache([], [])
    _save(entirely_empty, tmp_path / "all-empty")
    loaded_empty = _load(tmp_path / "all-empty")
    assert loaded_empty.query_ids == ()
    assert loaded_empty.candidates == ()


def test_complete_query_episode_is_excluded_not_only_query_frame() -> None:
    query = QueryId("libero", 0, 11, 5)
    same_episode_different_frame = _candidate("libero", 0, 11, 999, 0.9)

    with pytest.raises(EpisodeLeakageError, match="complete episode"):
        CandidateCache([query], [[same_episode_different_frame]])


def test_same_local_episode_number_in_other_dataset_is_not_excluded() -> None:
    query = QueryId("dataset-a", 0, 7, 2)
    rows = [
        [
            _candidate("dataset-b", 0, 7, 3, 0.8),
            _candidate("dataset-a", 1, 7, 4, 0.7),
        ]
    ]

    cache = CandidateCache([query], rows)
    cache.assert_episode_isolation()
    assert cache.candidates == (tuple(rows[0]),)


def test_validate_against_event_bank_checks_every_candidate_membership() -> None:
    present = EventId("bank", 0, 2, 4)
    bank = EventBank(
        [present],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    valid = CandidateCache(
        [QueryId("query", 0, 1, 0)],
        [[CachedCandidate(present, 0.8)]],
    )
    valid.validate_against_event_bank(bank)

    missing = EventId("bank", 0, 3, 8)
    invalid = CandidateCache(
        [QueryId("query", 0, 1, 0)],
        [[CachedCandidate(present, 0.8), CachedCandidate(missing, 0.7)]],
    )
    with pytest.raises(CandidateCacheContractError, match="not present"):
        invalid.validate_against_event_bank(bank)


def test_validate_against_event_bank_rechecks_episode_isolation() -> None:
    query = QueryId("bank", 0, 2, 1)
    leaked = CachedCandidate(EventId("bank", 0, 2, 4), 0.8)
    bank = EventBank(
        [leaked.event_id],
        np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    cache = CandidateCache([query], [[]])
    # Simulate accidental internal mutation by integration code.  Validation
    # against a bank must re-prove isolation rather than trusting construction.
    cache._candidates = ((leaked,),)

    with pytest.raises(EpisodeLeakageError, match="complete episode"):
        cache.validate_against_event_bank(bank)


def test_payload_file_corruption_is_detected_before_numpy_load(tmp_path: Path) -> None:
    cache = CandidateCache(
        [QueryId("q", 0, 1, 0)],
        [[_candidate("q", 0, 2, 0, 0.5)]],
    )
    _save(cache, tmp_path)
    manifest = CandidateCacheManifest.read(tmp_path / MANIFEST_FILENAME)
    payload = tmp_path / manifest.payload_file
    with payload.open("r+b") as handle:
        handle.seek(8)
        byte = handle.read(1)
        handle.seek(8)
        handle.write(bytes([byte[0] ^ 0xFF]))

    with pytest.raises(CandidateCacheIntegrityError, match="SHA-256 mismatch"):
        _load(tmp_path)


def test_array_hash_tamper_is_detected_even_when_payload_is_intact(tmp_path: Path) -> None:
    cache = CandidateCache(
        [QueryId("q", 0, 1, 0)],
        [[_candidate("q", 0, 2, 0, 0.5)]],
    )
    _save(cache, tmp_path)
    manifest_path = tmp_path / MANIFEST_FILENAME
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["content_hashes"]["array:candidate_cosine_score"] = "0" * 64
    manifest_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(CandidateCacheIntegrityError, match="candidate_cosine_score"):
        _load(tmp_path)


@pytest.mark.parametrize(
    ("keyword", "wrong_value", "message"),
    [
        ("expected_event_bank_manifest_hash", "c" * 64, "manifest_hash"),
        ("expected_event_bank_content_hash", "d" * 64, "content_hash"),
        ("expected_query_corpus_hash", "f" * 64, "query_corpus_hash"),
        (
            "expected_query_key_encoder",
            {"name": "different", "version": 1},
            "query_key_encoder",
        ),
        (
            "expected_build_recipe",
            {"implementation": "different"},
            "build_recipe",
        ),
    ],
)
def test_contract_mismatch_is_rejected(
    tmp_path: Path,
    keyword: str,
    wrong_value: object,
    message: str,
) -> None:
    cache = CandidateCache(
        [QueryId("q", 0, 1, 0)],
        [[_candidate("q", 0, 2, 0, 0.5)]],
    )
    _save(cache, tmp_path)
    kwargs = {
        "expected_event_bank_manifest_hash": BANK_MANIFEST_HASH,
        "expected_event_bank_content_hash": BANK_CONTENT_HASH,
        "expected_query_corpus_hash": QUERY_CORPUS_HASH,
        "expected_query_key_encoder": QUERY_ENCODER,
        "expected_build_recipe": BUILD_RECIPE,
    }
    kwargs[keyword] = wrong_value

    with pytest.raises(CandidateCacheContractError, match=message):
        CandidateCache.load(tmp_path, **kwargs)


def test_schema_tamper_is_rejected(tmp_path: Path) -> None:
    cache = CandidateCache([], [])
    _save(cache, tmp_path)
    manifest_path = tmp_path / MANIFEST_FILENAME
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["schema"] = "warm.not-candidate-cache"
    manifest_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(CandidateCacheManifestError, match="unsupported schema"):
        _load(tmp_path)


def test_overwrite_publishes_new_content_addressed_snapshot(tmp_path: Path) -> None:
    first = CandidateCache([QueryId("q", 0, 1, 0)], [[]])
    _save(first, tmp_path)
    first_payload = CandidateCacheManifest.read(tmp_path / MANIFEST_FILENAME).payload_file

    second = CandidateCache(
        [QueryId("q", 0, 1, 0)],
        [[_candidate("q", 0, 2, 0, 0.25)]],
    )
    _save(second, tmp_path, overwrite=True)
    second_manifest = CandidateCacheManifest.read(tmp_path / MANIFEST_FILENAME)

    assert second_manifest.payload_file != first_payload
    assert (tmp_path / first_payload).is_file()
    assert (tmp_path / second_manifest.payload_file).is_file()
    assert _load(tmp_path).num_candidates == 1
    assert not list(tmp_path.glob(".*.tmp"))


def test_canonical_event_bank_content_hash_is_order_independent() -> None:
    hashes_a = {"events.npz": "1" * 64, "array:key": "2" * 64}
    hashes_b = {"array:key": "2" * 64, "events.npz": "1" * 64}

    assert canonical_event_bank_content_hash(hashes_a) == canonical_event_bank_content_hash(
        hashes_b
    )
