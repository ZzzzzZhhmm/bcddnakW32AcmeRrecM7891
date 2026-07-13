"""Immutable, auditable offline candidate cache for WARM retrieval.

The cache is a CSR table from stable :class:`QueryId` values to variable-length
lists of :class:`EventId` candidates and their cosine scores.  Dataset ids are
stored as explicit UTF-8 byte pools; every numeric field uses a non-object NumPy
array.  Loading always uses ``allow_pickle=False``.

The fixed ``candidate_manifest.json`` is the atomic commit point.  Payloads are
immutable and content-addressed, so a concurrent reader that observed an older
manifest continues to reference the corresponding older payload while an
overwrite is published.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from .event_bank import EventBank
from .manifest import ArraySpec, ManifestError as ArrayManifestError, sha256_array, sha256_file
from .schema import EpisodeKey, EventId


CANDIDATE_CACHE_SCHEMA = "warm.candidate-cache"
CANDIDATE_CACHE_SCHEMA_VERSION = 2
MANIFEST_FILENAME = "candidate_manifest.json"
PAYLOAD_PREFIX = "candidates"

_QUERY_CANDIDATE_OFFSETS = "query_candidate_offsets"
_QUERY_DATASET_BYTES = "query_dataset_id_utf8"
_QUERY_DATASET_OFFSETS = "query_dataset_id_offsets"
_QUERY_DATASET_INDEX = "query_dataset_index"
_QUERY_EPISODE_INDEX = "query_episode_index"
_QUERY_FRAME_INDEX = "query_frame_index"
_CANDIDATE_DATASET_BYTES = "candidate_dataset_id_utf8"
_CANDIDATE_DATASET_OFFSETS = "candidate_dataset_id_offsets"
_CANDIDATE_DATASET_INDEX = "candidate_dataset_index"
_CANDIDATE_EPISODE_INDEX = "candidate_episode_index"
_CANDIDATE_START_FRAME = "candidate_start_frame"
_CANDIDATE_COSINE_SCORE = "candidate_cosine_score"

_REQUIRED_ARRAYS = frozenset(
    {
        _QUERY_CANDIDATE_OFFSETS,
        _QUERY_DATASET_BYTES,
        _QUERY_DATASET_OFFSETS,
        _QUERY_DATASET_INDEX,
        _QUERY_EPISODE_INDEX,
        _QUERY_FRAME_INDEX,
        _CANDIDATE_DATASET_BYTES,
        _CANDIDATE_DATASET_OFFSETS,
        _CANDIDATE_DATASET_INDEX,
        _CANDIDATE_EPISODE_INDEX,
        _CANDIDATE_START_FRAME,
        _CANDIDATE_COSINE_SCORE,
    }
)
_PAYLOAD_FILE_RE = re.compile(rf"^{PAYLOAD_PREFIX}-[0-9a-f]{{64}}\.npz$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class CandidateCacheError(ValueError):
    """Base class for candidate-cache validation failures."""


class CandidateCacheManifestError(CandidateCacheError):
    """Raised when manifest metadata violates the schema."""


class CandidateCacheIntegrityError(CandidateCacheError):
    """Raised when a file or array digest does not match the manifest."""


class CandidateCacheContractError(CandidateCacheError):
    """Raised when the cache belongs to a different bank or query encoder."""


class EpisodeLeakageError(CandidateCacheError):
    """Raised when a query retrieves any event from its complete episode."""


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _dataset_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("dataset_id must be a string")
    if not value or value != value.strip():
        raise ValueError("dataset_id must be non-empty without surrounding whitespace")
    if "\x00" in value:
        raise ValueError("dataset_id must not contain NUL characters")
    # Validate encodability now, rather than during serialization.
    value.encode("utf-8", errors="strict")
    return value


def _sha256_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _HEX_DIGEST_RE.fullmatch(value) is None:
        raise CandidateCacheManifestError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _json_mapping(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise CandidateCacheManifestError(
            f"{field} must contain finite JSON-compatible values"
        ) from exc
    if not isinstance(decoded, dict) or any(not isinstance(key, str) for key in decoded):
        raise CandidateCacheManifestError(f"{field} must be a JSON object")
    return decoded


@dataclass(frozen=True, order=True, slots=True)
class QueryId:
    """Stable query identity including its complete episode namespace."""

    dataset_id: str
    dataset_index: int
    episode_index: int
    frame_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _dataset_id(self.dataset_id))
        object.__setattr__(
            self, "dataset_index", _non_negative_int(self.dataset_index, "dataset_index")
        )
        object.__setattr__(
            self, "episode_index", _non_negative_int(self.episode_index, "episode_index")
        )
        object.__setattr__(
            self, "frame_index", _non_negative_int(self.frame_index, "frame_index")
        )

    @property
    def episode_key(self) -> EpisodeKey:
        return (self.dataset_id, self.dataset_index, self.episode_index)


@dataclass(frozen=True, slots=True)
class CachedCandidate:
    """One event candidate and the cosine score produced by offline search."""

    event_id: EventId
    cosine_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, EventId):
            raise TypeError("event_id must be EventId")
        if isinstance(self.cosine_score, bool) or not isinstance(self.cosine_score, Real):
            raise TypeError("cosine_score must be a finite real number")
        score = float(self.cosine_score)
        if not np.isfinite(score):
            raise ValueError("cosine_score must be finite")
        # Exact cosine implementations may exceed one by a few ulps.
        if score < -1.0 - 1e-6 or score > 1.0 + 1e-6:
            raise ValueError("cosine_score must be in [-1, 1]")
        object.__setattr__(self, "cosine_score", min(1.0, max(-1.0, score)))


@dataclass(frozen=True, slots=True)
class CandidateCacheManifest:
    """Versioned integrity and model-contract record for a candidate cache."""

    event_bank_manifest_hash: str
    event_bank_content_hash: str
    query_corpus_hash: str
    query_key_encoder: Mapping[str, Any]
    build_recipe: Mapping[str, Any]
    arrays: Mapping[str, ArraySpec]
    content_hashes: Mapping[str, str]
    num_queries: int
    num_candidates: int
    payload_file: str
    schema: str = CANDIDATE_CACHE_SCHEMA
    version: int = CANDIDATE_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != CANDIDATE_CACHE_SCHEMA:
            raise CandidateCacheManifestError(f"unsupported schema {self.schema!r}")
        if self.version != CANDIDATE_CACHE_SCHEMA_VERSION:
            raise CandidateCacheManifestError(f"unsupported schema version {self.version!r}")
        if not isinstance(self.payload_file, str) or _PAYLOAD_FILE_RE.fullmatch(
            self.payload_file
        ) is None:
            raise CandidateCacheManifestError("payload_file is not a content-addressed cache NPZ")
        if Path(self.payload_file).name != self.payload_file:
            raise CandidateCacheManifestError("payload_file must not contain directories")

        object.__setattr__(
            self,
            "event_bank_manifest_hash",
            _sha256_digest(self.event_bank_manifest_hash, "event_bank_manifest_hash"),
        )
        object.__setattr__(
            self,
            "event_bank_content_hash",
            _sha256_digest(self.event_bank_content_hash, "event_bank_content_hash"),
        )
        object.__setattr__(
            self,
            "query_corpus_hash",
            _sha256_digest(self.query_corpus_hash, "query_corpus_hash"),
        )
        query_encoder = _json_mapping(self.query_key_encoder, "query_key_encoder")
        build_recipe = _json_mapping(self.build_recipe, "build_recipe")

        num_queries = _non_negative_int(self.num_queries, "num_queries")
        num_candidates = _non_negative_int(self.num_candidates, "num_candidates")

        if not isinstance(self.arrays, Mapping):
            raise TypeError("arrays must be a mapping")
        arrays: dict[str, ArraySpec] = {}
        for name, spec in self.arrays.items():
            if not isinstance(name, str) or not name:
                raise CandidateCacheManifestError("array names must be non-empty strings")
            if not isinstance(spec, ArraySpec):
                raise TypeError(f"array spec for {name!r} must be ArraySpec")
            arrays[name] = spec
        if set(arrays) != _REQUIRED_ARRAYS:
            raise CandidateCacheManifestError(
                "manifest arrays do not match the candidate-cache schema; "
                f"missing={sorted(_REQUIRED_ARRAYS - set(arrays))}, "
                f"extra={sorted(set(arrays) - _REQUIRED_ARRAYS)}"
            )

        if not isinstance(self.content_hashes, Mapping):
            raise TypeError("content_hashes must be a mapping")
        hashes: dict[str, str] = {}
        for name, digest in self.content_hashes.items():
            if not isinstance(name, str) or not name:
                raise CandidateCacheManifestError("content-hash names must be non-empty strings")
            hashes[name] = _sha256_digest(digest, f"content_hashes[{name!r}]")
        required_hashes = {self.payload_file} | {f"array:{name}" for name in arrays}
        if set(hashes) != required_hashes:
            raise CandidateCacheManifestError(
                "content hashes do not match payload and array members; "
                f"missing={sorted(required_hashes - set(hashes))}, "
                f"extra={sorted(set(hashes) - required_hashes)}"
            )

        object.__setattr__(self, "num_queries", num_queries)
        object.__setattr__(self, "num_candidates", num_candidates)
        object.__setattr__(self, "query_key_encoder", MappingProxyType(query_encoder))
        object.__setattr__(self, "build_recipe", MappingProxyType(build_recipe))
        object.__setattr__(self, "arrays", MappingProxyType(dict(sorted(arrays.items()))))
        object.__setattr__(self, "content_hashes", MappingProxyType(dict(sorted(hashes.items()))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "payload_file": self.payload_file,
            "num_queries": self.num_queries,
            "num_candidates": self.num_candidates,
            "event_bank_manifest_hash": self.event_bank_manifest_hash,
            "event_bank_content_hash": self.event_bank_content_hash,
            "query_corpus_hash": self.query_corpus_hash,
            "query_key_encoder": _json_mapping(
                self.query_key_encoder, "query_key_encoder"
            ),
            "build_recipe": _json_mapping(self.build_recipe, "build_recipe"),
            "arrays": {name: spec.to_dict() for name, spec in self.arrays.items()},
            "content_hashes": dict(self.content_hashes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateCacheManifest":
        if not isinstance(value, Mapping):
            raise TypeError("manifest must be a mapping")
        expected = {
            "schema",
            "version",
            "payload_file",
            "num_queries",
            "num_candidates",
            "event_bank_manifest_hash",
            "event_bank_content_hash",
            "query_corpus_hash",
            "query_key_encoder",
            "build_recipe",
            "arrays",
            "content_hashes",
        }
        if set(value) != expected:
            raise CandidateCacheManifestError(
                "invalid manifest fields; "
                f"missing={sorted(expected - set(value))}, "
                f"extra={sorted(set(value) - expected)}"
            )
        arrays_value = value["arrays"]
        if not isinstance(arrays_value, Mapping):
            raise CandidateCacheManifestError("arrays must be a JSON object")
        try:
            arrays = {
                name: ArraySpec.from_dict(spec) for name, spec in arrays_value.items()
            }
        except (ArrayManifestError, TypeError, ValueError) as exc:
            raise CandidateCacheManifestError("invalid array specification") from exc
        return cls(
            schema=value["schema"],
            version=value["version"],
            payload_file=value["payload_file"],
            num_queries=value["num_queries"],
            num_candidates=value["num_candidates"],
            event_bank_manifest_hash=value["event_bank_manifest_hash"],
            event_bank_content_hash=value["event_bank_content_hash"],
            query_corpus_hash=value["query_corpus_hash"],
            query_key_encoder=value["query_key_encoder"],
            build_recipe=value["build_recipe"],
            arrays=arrays,
            content_hashes=value["content_hashes"],
        )

    @classmethod
    def read(cls, path: str | Path) -> "CandidateCacheManifest":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CandidateCacheManifestError(f"cannot read manifest {path}") from exc
        return cls.from_dict(value)


def _encode_utf8(values: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    encoded = [value.encode("utf-8", errors="strict") for value in values]
    offsets = np.empty((len(encoded) + 1,), dtype=np.int64)
    offsets[0] = 0
    for index, item in enumerate(encoded, start=1):
        offsets[index] = offsets[index - 1] + len(item)
    data = (
        np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
        if encoded
        else np.empty((0,), dtype=np.uint8)
    )
    return np.ascontiguousarray(data), np.ascontiguousarray(offsets)


def _validate_offsets(
    offsets: np.ndarray,
    *,
    expected_count: int,
    terminal: int,
    name: str,
) -> None:
    if offsets.dtype != np.dtype(np.int64) or offsets.shape != (expected_count + 1,):
        raise CandidateCacheError(
            f"{name} must have dtype int64 and shape {(expected_count + 1,)}"
        )
    if (
        offsets[0] != 0
        or offsets[-1] != terminal
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise CandidateCacheError(f"{name} is not a valid CSR/UTF-8 offset array")


def _decode_utf8(
    data: np.ndarray,
    offsets: np.ndarray,
    *,
    count: int,
    field: str,
) -> list[str]:
    if data.dtype != np.dtype(np.uint8) or data.ndim != 1:
        raise CandidateCacheError(f"{field} byte pool must be a 1-D uint8 array")
    _validate_offsets(
        offsets,
        expected_count=count,
        terminal=int(data.size),
        name=f"{field} offsets",
    )
    raw = memoryview(data).cast("B")
    decoded: list[str] = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        try:
            decoded.append(bytes(raw[int(start) : int(end)]).decode("utf-8", errors="strict"))
        except UnicodeDecodeError as exc:
            raise CandidateCacheError(f"{field} contains invalid UTF-8") from exc
    return decoded


def _int_array(values: Sequence[int]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.int64))


class CandidateCache:
    """Immutable query-to-candidate table with whole-episode exclusion."""

    def __init__(
        self,
        query_ids: Sequence[QueryId],
        candidates: Sequence[Sequence[CachedCandidate]],
    ) -> None:
        queries = tuple(query_ids)
        rows = tuple(tuple(row) for row in candidates)
        if any(not isinstance(query, QueryId) for query in queries):
            raise TypeError("every query id must be QueryId")
        if len(set(queries)) != len(queries):
            raise CandidateCacheError("query ids must be unique")
        if len(rows) != len(queries):
            raise CandidateCacheError("candidates must contain one row per query")
        for row in rows:
            if any(not isinstance(candidate, CachedCandidate) for candidate in row):
                raise TypeError("every candidate must be CachedCandidate")
            event_ids = [candidate.event_id for candidate in row]
            if len(set(event_ids)) != len(event_ids):
                raise CandidateCacheError("candidate event ids must be unique within a query")

        self._query_ids = queries
        self._candidates = rows
        self._manifest: CandidateCacheManifest | None = None
        self.assert_episode_isolation()

    def __len__(self) -> int:
        return len(self._query_ids)

    @property
    def query_ids(self) -> tuple[QueryId, ...]:
        return self._query_ids

    @property
    def candidates(self) -> tuple[tuple[CachedCandidate, ...], ...]:
        return self._candidates

    @property
    def manifest(self) -> CandidateCacheManifest | None:
        return self._manifest

    @property
    def num_candidates(self) -> int:
        return sum(len(row) for row in self._candidates)

    def candidates_for(self, query: int | QueryId) -> tuple[CachedCandidate, ...]:
        if isinstance(query, bool):
            raise TypeError("query must be an integer index or QueryId")
        if isinstance(query, Integral):
            index = int(query)
            if index < 0 or index >= len(self):
                raise IndexError(index)
            return self._candidates[index]
        if isinstance(query, QueryId):
            try:
                return self._candidates[self._query_ids.index(query)]
            except ValueError as exc:
                raise KeyError(query) from exc
        raise TypeError("query must be an integer index or QueryId")

    def assert_episode_isolation(self) -> None:
        """Prove every row excludes all frames of the query's complete episode."""

        for query, row in zip(self._query_ids, self._candidates, strict=True):
            for candidate in row:
                if candidate.event_id.episode_key == query.episode_key:
                    raise EpisodeLeakageError(
                        "candidate cache leaks the query's complete episode: "
                        f"query={query!r}, candidate={candidate.event_id!r}"
                    )

    def validate_against_event_bank(self, bank: EventBank) -> None:
        """Validate episode isolation and candidate referential integrity.

        The caller remains responsible for comparing this cache manifest's
        event-bank hashes with the loaded bank artifact.  This method proves
        that every cached :class:`EventId` resolves in that already-validated
        bank and rechecks the whole-episode exclusion invariant for every row.
        """

        if not isinstance(bank, EventBank):
            raise TypeError("bank must be EventBank")
        self.assert_episode_isolation()
        bank_event_ids = set(bank.event_ids)
        missing = sorted(
            {
                candidate.event_id
                for row in self._candidates
                for candidate in row
                if candidate.event_id not in bank_event_ids
            }
        )
        if missing:
            preview = ", ".join(repr(event_id) for event_id in missing[:3])
            suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
            raise CandidateCacheContractError(
                "candidate cache references EventId values not present in the "
                f"event bank: {preview}{suffix}"
            )

    def _storage_arrays(self) -> dict[str, np.ndarray]:
        flat = [candidate for row in self._candidates for candidate in row]
        candidate_offsets = np.empty((len(self) + 1,), dtype=np.int64)
        candidate_offsets[0] = 0
        for index, row in enumerate(self._candidates, start=1):
            candidate_offsets[index] = candidate_offsets[index - 1] + len(row)

        query_dataset_bytes, query_dataset_offsets = _encode_utf8(
            [query.dataset_id for query in self._query_ids]
        )
        candidate_dataset_bytes, candidate_dataset_offsets = _encode_utf8(
            [candidate.event_id.dataset_id for candidate in flat]
        )
        arrays = {
            _QUERY_CANDIDATE_OFFSETS: np.ascontiguousarray(candidate_offsets),
            _QUERY_DATASET_BYTES: query_dataset_bytes,
            _QUERY_DATASET_OFFSETS: query_dataset_offsets,
            _QUERY_DATASET_INDEX: _int_array(
                [query.dataset_index for query in self._query_ids]
            ),
            _QUERY_EPISODE_INDEX: _int_array(
                [query.episode_index for query in self._query_ids]
            ),
            _QUERY_FRAME_INDEX: _int_array([query.frame_index for query in self._query_ids]),
            _CANDIDATE_DATASET_BYTES: candidate_dataset_bytes,
            _CANDIDATE_DATASET_OFFSETS: candidate_dataset_offsets,
            _CANDIDATE_DATASET_INDEX: _int_array(
                [candidate.event_id.dataset_index for candidate in flat]
            ),
            _CANDIDATE_EPISODE_INDEX: _int_array(
                [candidate.event_id.episode_index for candidate in flat]
            ),
            _CANDIDATE_START_FRAME: _int_array(
                [candidate.event_id.start_frame for candidate in flat]
            ),
            _CANDIDATE_COSINE_SCORE: np.ascontiguousarray(
                np.asarray([candidate.cosine_score for candidate in flat], dtype=np.float32)
            ),
        }
        return dict(sorted(arrays.items()))

    def save(
        self,
        directory: str | Path,
        *,
        event_bank_manifest_hash: str,
        event_bank_content_hash: str,
        query_corpus_hash: str,
        query_key_encoder: Mapping[str, Any],
        build_recipe: Mapping[str, Any],
        overwrite: bool = False,
    ) -> CandidateCacheManifest:
        """Atomically publish a content-addressed NPZ and commit manifest."""

        self.assert_episode_isolation()
        # Validate the external contract before touching disk.
        event_bank_manifest_hash = _sha256_digest(
            event_bank_manifest_hash, "event_bank_manifest_hash"
        )
        event_bank_content_hash = _sha256_digest(
            event_bank_content_hash, "event_bank_content_hash"
        )
        query_corpus_hash = _sha256_digest(query_corpus_hash, "query_corpus_hash")
        query_key_encoder = _json_mapping(query_key_encoder, "query_key_encoder")
        build_recipe = _json_mapping(build_recipe, "build_recipe")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        manifest_path = directory / MANIFEST_FILENAME
        if manifest_path.exists() and not overwrite:
            raise FileExistsError(f"candidate cache already exists in {directory}")

        arrays = self._storage_arrays()
        temporary_payload = directory / f".{PAYLOAD_PREFIX}.{uuid4().hex}.npz.tmp"
        temporary_manifest = directory / f".{MANIFEST_FILENAME}.{uuid4().hex}.tmp"
        manifest: CandidateCacheManifest | None = None
        try:
            with temporary_payload.open("wb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())

            payload_hash = sha256_file(temporary_payload)
            payload_file = f"{PAYLOAD_PREFIX}-{payload_hash}.npz"
            payload_path = directory / payload_file
            hashes = {f"array:{name}": sha256_array(array) for name, array in arrays.items()}
            hashes[payload_file] = payload_hash
            manifest = CandidateCacheManifest(
                event_bank_manifest_hash=event_bank_manifest_hash,
                event_bank_content_hash=event_bank_content_hash,
                query_corpus_hash=query_corpus_hash,
                query_key_encoder=query_key_encoder,
                build_recipe=build_recipe,
                arrays={name: ArraySpec.from_array(array) for name, array in arrays.items()},
                content_hashes=hashes,
                num_queries=len(self),
                num_candidates=self.num_candidates,
                payload_file=payload_file,
            )

            if payload_path.exists():
                if sha256_file(payload_path) != payload_hash:
                    raise CandidateCacheIntegrityError(
                        f"existing content-addressed payload {payload_path} has wrong hash"
                    )
                temporary_payload.unlink()
            else:
                os.replace(temporary_payload, payload_path)

            encoded_manifest = json.dumps(
                manifest.to_dict(),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            with temporary_manifest.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded_manifest)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            # Atomic commit point. Older content-addressed payloads are retained
            # so readers holding an older manifest continue to have a valid snapshot.
            os.replace(temporary_manifest, manifest_path)
        finally:
            temporary_payload.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)

        assert manifest is not None
        self._manifest = manifest
        return manifest

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_event_bank_manifest_hash: str | None = None,
        expected_event_bank_content_hash: str | None = None,
        expected_query_corpus_hash: str | None = None,
        expected_query_key_encoder: Mapping[str, Any] | None = None,
        expected_build_recipe: Mapping[str, Any] | None = None,
    ) -> "CandidateCache":
        """Load after schema, contract, file, member, dtype and array-hash checks."""

        directory = Path(directory)
        manifest = CandidateCacheManifest.read(directory / MANIFEST_FILENAME)

        for field, expected, actual in (
            (
                "event_bank_manifest_hash",
                expected_event_bank_manifest_hash,
                manifest.event_bank_manifest_hash,
            ),
            (
                "event_bank_content_hash",
                expected_event_bank_content_hash,
                manifest.event_bank_content_hash,
            ),
            (
                "query_corpus_hash",
                expected_query_corpus_hash,
                manifest.query_corpus_hash,
            ),
        ):
            if expected is not None:
                expected_digest = _sha256_digest(expected, field)
                if expected_digest != actual:
                    raise CandidateCacheContractError(
                        f"{field} does not match the requested event-bank contract"
                    )
        if expected_query_key_encoder is not None:
            expected_encoder = _json_mapping(
                expected_query_key_encoder, "expected_query_key_encoder"
            )
            if expected_encoder != dict(manifest.query_key_encoder):
                raise CandidateCacheContractError(
                    "query_key_encoder does not match the requested model contract"
                )
        if expected_build_recipe is not None:
            expected_recipe = _json_mapping(
                expected_build_recipe, "expected_build_recipe"
            )
            if expected_recipe != dict(manifest.build_recipe):
                raise CandidateCacheContractError(
                    "build_recipe does not match the requested cache recipe"
                )

        payload_path = directory / manifest.payload_file
        if not payload_path.is_file():
            raise FileNotFoundError(payload_path)
        actual_file_hash = sha256_file(payload_path)
        expected_file_hash = manifest.content_hashes[manifest.payload_file]
        if actual_file_hash != expected_file_hash:
            raise CandidateCacheIntegrityError(
                f"SHA-256 mismatch for {manifest.payload_file}: "
                f"expected {expected_file_hash}, got {actual_file_hash}"
            )

        arrays: dict[str, np.ndarray] = {}
        try:
            with np.load(payload_path, allow_pickle=False) as archive:
                actual_names = set(archive.files)
                expected_names = set(manifest.arrays)
                if actual_names != expected_names:
                    raise CandidateCacheManifestError(
                        "NPZ members do not match manifest; "
                        f"missing={sorted(expected_names - actual_names)}, "
                        f"extra={sorted(actual_names - expected_names)}"
                    )
                for name, spec in manifest.arrays.items():
                    array = archive[name]
                    try:
                        spec.validate(name, array)
                    except ArrayManifestError as exc:
                        raise CandidateCacheManifestError(
                            f"array {name!r} violates its manifest spec"
                        ) from exc
                    expected_array_hash = manifest.content_hashes[f"array:{name}"]
                    actual_array_hash = sha256_array(array)
                    if actual_array_hash != expected_array_hash:
                        raise CandidateCacheIntegrityError(
                            f"SHA-256 mismatch for array {name!r}: "
                            f"expected {expected_array_hash}, got {actual_array_hash}"
                        )
                    arrays[name] = np.array(array, copy=True, order="C")
        except (OSError, ValueError, EOFError) as exc:
            if isinstance(
                exc,
                (
                    CandidateCacheError,
                    CandidateCacheManifestError,
                    CandidateCacheIntegrityError,
                ),
            ):
                raise
            raise CandidateCacheError(f"cannot load NumPy payload {payload_path}") from exc

        query_count = manifest.num_queries
        candidate_count = manifest.num_candidates
        for name in (
            _QUERY_DATASET_INDEX,
            _QUERY_EPISODE_INDEX,
            _QUERY_FRAME_INDEX,
        ):
            if arrays[name].dtype != np.dtype(np.int64) or arrays[name].shape != (
                query_count,
            ):
                raise CandidateCacheError(
                    f"{name} must have dtype int64 and shape {(query_count,)}"
                )
        for name in (
            _CANDIDATE_DATASET_INDEX,
            _CANDIDATE_EPISODE_INDEX,
            _CANDIDATE_START_FRAME,
        ):
            if arrays[name].dtype != np.dtype(np.int64) or arrays[name].shape != (
                candidate_count,
            ):
                raise CandidateCacheError(
                    f"{name} must have dtype int64 and shape {(candidate_count,)}"
                )
        scores = arrays[_CANDIDATE_COSINE_SCORE]
        if scores.dtype != np.dtype(np.float32) or scores.shape != (candidate_count,):
            raise CandidateCacheError(
                f"{_CANDIDATE_COSINE_SCORE} must have dtype float32 and shape "
                f"{(candidate_count,)}"
            )
        if not np.isfinite(scores).all() or np.any(scores < -1.0) or np.any(scores > 1.0):
            raise CandidateCacheError("candidate cosine scores must be finite in [-1, 1]")

        csr_offsets = arrays[_QUERY_CANDIDATE_OFFSETS]
        _validate_offsets(
            csr_offsets,
            expected_count=query_count,
            terminal=candidate_count,
            name=_QUERY_CANDIDATE_OFFSETS,
        )
        query_dataset_ids = _decode_utf8(
            arrays[_QUERY_DATASET_BYTES],
            arrays[_QUERY_DATASET_OFFSETS],
            count=query_count,
            field="query dataset ids",
        )
        candidate_dataset_ids = _decode_utf8(
            arrays[_CANDIDATE_DATASET_BYTES],
            arrays[_CANDIDATE_DATASET_OFFSETS],
            count=candidate_count,
            field="candidate dataset ids",
        )

        queries = tuple(
            QueryId(dataset_id, dataset_index, episode_index, frame_index)
            for dataset_id, dataset_index, episode_index, frame_index in zip(
                query_dataset_ids,
                arrays[_QUERY_DATASET_INDEX].tolist(),
                arrays[_QUERY_EPISODE_INDEX].tolist(),
                arrays[_QUERY_FRAME_INDEX].tolist(),
                strict=True,
            )
        )
        flat_candidates = tuple(
            CachedCandidate(
                EventId(dataset_id, dataset_index, episode_index, start_frame),
                score,
            )
            for dataset_id, dataset_index, episode_index, start_frame, score in zip(
                candidate_dataset_ids,
                arrays[_CANDIDATE_DATASET_INDEX].tolist(),
                arrays[_CANDIDATE_EPISODE_INDEX].tolist(),
                arrays[_CANDIDATE_START_FRAME].tolist(),
                scores.tolist(),
                strict=True,
            )
        )
        rows = tuple(
            flat_candidates[int(start) : int(end)]
            for start, end in zip(csr_offsets[:-1], csr_offsets[1:], strict=True)
        )
        cache = cls(queries, rows)
        cache._manifest = manifest
        return cache


def canonical_event_bank_content_hash(content_hashes: Mapping[str, str]) -> str:
    """Create a deterministic root digest from an event-bank hash mapping."""

    if not isinstance(content_hashes, Mapping) or not content_hashes:
        raise TypeError("content_hashes must be a non-empty mapping")
    normalized: dict[str, str] = {}
    for name, digest in content_hashes.items():
        if not isinstance(name, str) or not name:
            raise TypeError("content hash names must be non-empty strings")
        normalized[name] = _sha256_digest(digest, f"content_hashes[{name!r}]")
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CANDIDATE_CACHE_SCHEMA",
    "CANDIDATE_CACHE_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "CachedCandidate",
    "CandidateCache",
    "CandidateCacheContractError",
    "CandidateCacheError",
    "CandidateCacheIntegrityError",
    "CandidateCacheManifest",
    "CandidateCacheManifestError",
    "EpisodeLeakageError",
    "QueryId",
    "canonical_event_bank_content_hash",
]
