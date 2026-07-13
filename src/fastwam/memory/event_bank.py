"""Immutable NumPy event bank with exact cosine retrieval.

This backend is intentionally small and dependency-free.  It is the reference
implementation for M1 correctness tests; an ANN index can later be rebuilt
from the same manifest and payload without changing the source-of-truth bank.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from .manifest import (
    ArraySpec,
    EventBankManifest,
    ManifestError,
    sha256_array,
    sha256_file,
)
from .schema import (
    EVENT_BANK_SCHEMA,
    EVENT_BANK_SCHEMA_VERSION,
    EpisodeKey,
    EventId,
    coerce_episode_key,
)


MANIFEST_FILENAME = "manifest.json"
PAYLOAD_FILENAME = "events.npz"

_CONTEXT_KEY = "context_key"
_ID_DATASET_BYTES = "_event_dataset_id_utf8"
_ID_DATASET_OFFSETS = "_event_dataset_id_offsets"
_ID_DATASET_INDEX = "_event_dataset_index"
_ID_EPISODE_INDEX = "_event_episode_index"
_ID_START_FRAME = "_event_start_frame"
_REQUIRED_ARRAYS = frozenset(
    {
        _CONTEXT_KEY,
        _ID_DATASET_BYTES,
        _ID_DATASET_OFFSETS,
        _ID_DATASET_INDEX,
        _ID_EPISODE_INDEX,
        _ID_START_FRAME,
    }
)
_PAYLOAD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class EventBankError(ValueError):
    """Raised when event-bank payload or retrieval inputs are invalid."""


class IntegrityError(EventBankError):
    """Raised when stored content does not match the manifest hash."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    index: int
    event_id: EventId
    score: float


def _readonly_contiguous(array: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.dtype.hasobject or array.dtype.fields is not None or array.dtype.subdtype is not None:
        raise TypeError(f"{name} has an unsupported dtype {array.dtype}")
    if array.dtype.kind not in "biuf":
        raise TypeError(f"{name} has an unsupported dtype {array.dtype}")
    result = np.array(array, copy=True, order="C")
    if result.dtype.kind == "f" and not np.isfinite(result).all():
        raise EventBankError(f"{name} must contain only finite values")
    result.flags.writeable = False
    return result


def _encode_dataset_ids(event_ids: Sequence[EventId]) -> tuple[np.ndarray, np.ndarray]:
    encoded = [event_id.dataset_id.encode("utf-8") for event_id in event_ids]
    offsets = np.empty(len(encoded) + 1, dtype=np.int64)
    offsets[0] = 0
    for index, value in enumerate(encoded, start=1):
        offsets[index] = offsets[index - 1] + len(value)
    if encoded:
        data = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
    else:
        data = np.empty((0,), dtype=np.uint8)
    return np.ascontiguousarray(data), np.ascontiguousarray(offsets)


def _decode_dataset_ids(data: np.ndarray, offsets: np.ndarray, count: int) -> list[str]:
    if data.dtype != np.dtype(np.uint8) or data.ndim != 1:
        raise EventBankError(f"{_ID_DATASET_BYTES} must be a 1-D uint8 array")
    if offsets.dtype != np.dtype(np.int64) or offsets.shape != (count + 1,):
        raise EventBankError(
            f"{_ID_DATASET_OFFSETS} must have dtype int64 and shape {(count + 1,)}"
        )
    if offsets[0] != 0 or offsets[-1] != data.size or np.any(offsets[1:] < offsets[:-1]):
        raise EventBankError("dataset-id UTF-8 offsets are invalid")
    raw = memoryview(data).cast("B")
    result: list[str] = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        try:
            result.append(bytes(raw[int(start) : int(end)]).decode("utf-8", errors="strict"))
        except UnicodeDecodeError as exc:
            raise EventBankError("dataset id payload is not valid UTF-8") from exc
    return result


class EventBank:
    """A fixed event table and exact cosine-search reference backend.

    Parameters
    ----------
    event_ids:
        Stable identities in row order.
    context_keys:
        Float32 matrix ``[N, key_dim]``. Rows need not be pre-normalized.
    payloads:
        Named C-compatible NumPy tensors whose first dimension is ``N``.
        Typical names are ``model_space_action``, ``effect_tokens`` and
        ``start_proprio``. Values are copied into immutable contiguous arrays.
    """

    def __init__(
        self,
        event_ids: Sequence[EventId],
        context_keys: np.ndarray,
        payloads: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        ids = tuple(event_ids)
        if any(not isinstance(event_id, EventId) for event_id in ids):
            raise TypeError("every event id must be EventId")
        if len(set(ids)) != len(ids):
            raise EventBankError("event ids must be unique")

        keys = _readonly_contiguous(context_keys, "context_keys")
        if keys.dtype != np.dtype(np.float32):
            raise TypeError(f"context_keys must have dtype float32, got {keys.dtype}")
        if keys.ndim != 2 or keys.shape[1] == 0:
            raise EventBankError("context_keys must have shape [N, key_dim] with key_dim > 0")
        if keys.shape[0] != len(ids):
            raise EventBankError(
                f"context_keys row count {keys.shape[0]} does not match {len(ids)} event ids"
            )
        norms = np.linalg.norm(keys.astype(np.float64), axis=1)
        if np.any(norms <= 0.0):
            raise EventBankError("context_keys rows must have non-zero cosine norm")

        stored_payloads: dict[str, np.ndarray] = {}
        for name, array in (payloads or {}).items():
            if not isinstance(name, str) or not _PAYLOAD_NAME.fullmatch(name):
                raise EventBankError(
                    f"payload name {name!r} must match {_PAYLOAD_NAME.pattern!r}"
                )
            if name in _REQUIRED_ARRAYS:
                raise EventBankError(f"payload name {name!r} is reserved")
            stored = _readonly_contiguous(array, name)
            if stored.ndim == 0 or stored.shape[0] != len(ids):
                raise EventBankError(f"{name} must have leading event dimension {len(ids)}")
            stored_payloads[name] = stored

        self._event_ids = ids
        self._context_keys = keys
        self._key_norms = norms
        self._payloads = MappingProxyType(dict(sorted(stored_payloads.items())))
        self._manifest: EventBankManifest | None = None

    @classmethod
    def from_arrays(
        cls,
        event_ids: Sequence[EventId],
        context_keys: np.ndarray,
        **payloads: np.ndarray,
    ) -> "EventBank":
        """Convenient named-array constructor."""

        return cls(event_ids=event_ids, context_keys=context_keys, payloads=payloads)

    def __len__(self) -> int:
        return len(self._event_ids)

    @property
    def event_ids(self) -> tuple[EventId, ...]:
        return self._event_ids

    @property
    def context_keys(self) -> np.ndarray:
        return self._context_keys

    @property
    def payloads(self) -> Mapping[str, np.ndarray]:
        return self._payloads

    @property
    def manifest(self) -> EventBankManifest | None:
        return self._manifest

    def payload(self, name: str) -> np.ndarray:
        try:
            return self._payloads[name]
        except KeyError as exc:
            raise KeyError(f"unknown event payload {name!r}") from exc

    def payload_row(self, index: int) -> Mapping[str, np.ndarray | np.generic]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("event index must be an integer")
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return MappingProxyType({name: value[index] for name, value in self._payloads.items()})

    def search(
        self,
        query: np.ndarray | Sequence[float],
        top_k: int = 32,
        *,
        exclude_episode: EventId | EpisodeKey | None = None,
    ) -> tuple[SearchResult, ...]:
        """Return exact cosine neighbors, optionally leaving one episode out.

        Exclusion uses ``(dataset_id, dataset_index, episode_index)`` and thus
        removes every start frame from the query episode, not just the query
        event itself.
        """

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be a positive integer")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_array = np.asarray(query)
        if query_array.ndim != 1 or query_array.shape[0] != self._context_keys.shape[1]:
            raise EventBankError(
                f"query must have shape {(self._context_keys.shape[1],)}, got {query_array.shape}"
            )
        if query_array.dtype.kind not in "iuf":
            raise TypeError("query must be numeric")
        query_float = query_array.astype(np.float64, copy=False)
        if not np.isfinite(query_float).all():
            raise EventBankError("query must contain only finite values")
        query_norm = float(np.linalg.norm(query_float))
        if query_norm <= 0.0:
            raise EventBankError("query must have non-zero cosine norm")

        episode_key = (
            None if exclude_episode is None else coerce_episode_key(exclude_episode)
        )
        allowed = np.fromiter(
            (
                episode_key is None or event_id.episode_key != episode_key
                for event_id in self._event_ids
            ),
            dtype=np.bool_,
            count=len(self),
        )
        candidate_indices = np.flatnonzero(allowed)
        if candidate_indices.size == 0:
            return ()

        scores = (
            self._context_keys.astype(np.float64, copy=False) @ query_float
        ) / (self._key_norms * query_norm)
        candidate_scores = scores[candidate_indices]
        # Stable sorting makes equal-score behavior reproducible in bank row order.
        order = np.argsort(-candidate_scores, kind="stable")[:top_k]
        return tuple(
            SearchResult(
                index=int(candidate_indices[position]),
                event_id=self._event_ids[int(candidate_indices[position])],
                score=float(candidate_scores[position]),
            )
            for position in order
        )

    def _storage_arrays(self) -> dict[str, np.ndarray]:
        dataset_bytes, dataset_offsets = _encode_dataset_ids(self._event_ids)
        arrays: dict[str, np.ndarray] = {
            _ID_DATASET_BYTES: dataset_bytes,
            _ID_DATASET_OFFSETS: dataset_offsets,
            _ID_DATASET_INDEX: np.ascontiguousarray(
                np.array([event_id.dataset_index for event_id in self._event_ids], dtype=np.int64)
            ),
            _ID_EPISODE_INDEX: np.ascontiguousarray(
                np.array([event_id.episode_index for event_id in self._event_ids], dtype=np.int64)
            ),
            _ID_START_FRAME: np.ascontiguousarray(
                np.array([event_id.start_frame for event_id in self._event_ids], dtype=np.int64)
            ),
            _CONTEXT_KEY: np.ascontiguousarray(self._context_keys),
        }
        arrays.update({name: np.ascontiguousarray(value) for name, value in self._payloads.items()})
        return dict(sorted(arrays.items()))

    def save(
        self,
        directory: str | Path,
        *,
        action_normalizer: Mapping[str, Any],
        encoder: Mapping[str, Any],
        camera_layout: Mapping[str, Any],
        overwrite: bool = False,
    ) -> EventBankManifest:
        """Atomically write the reference NPZ payload and its manifest."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload_path = directory / PAYLOAD_FILENAME
        manifest_path = directory / MANIFEST_FILENAME
        if not overwrite and (payload_path.exists() or manifest_path.exists()):
            raise FileExistsError(f"event bank already exists in {directory}")

        arrays = self._storage_arrays()
        temporary_payload = directory / f".{PAYLOAD_FILENAME}.{uuid4().hex}.tmp"
        temporary_manifest = directory / f".{MANIFEST_FILENAME}.{uuid4().hex}.tmp"
        try:
            with temporary_payload.open("wb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())

            hashes = {f"array:{name}": sha256_array(array) for name, array in arrays.items()}
            hashes[PAYLOAD_FILENAME] = sha256_file(temporary_payload)
            manifest = EventBankManifest(
                schema=EVENT_BANK_SCHEMA,
                version=EVENT_BANK_SCHEMA_VERSION,
                payload_file=PAYLOAD_FILENAME,
                num_events=len(self),
                action_normalizer=action_normalizer,
                encoder=encoder,
                camera_layout=camera_layout,
                arrays={name: ArraySpec.from_array(array) for name, array in arrays.items()},
                content_hashes=hashes,
            )
            manifest.write(temporary_manifest)
            # Validate every piece before publishing either final filename.
            # The manifest is replaced last so readers never accept a payload
            # under a not-yet-written contract.
            os.replace(temporary_payload, payload_path)
            os.replace(temporary_manifest, manifest_path)
        finally:
            temporary_payload.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)

        self._manifest = manifest
        return manifest

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        expected_action_normalizer: Mapping[str, Any] | None = None,
        expected_encoder: Mapping[str, Any] | None = None,
        expected_camera_layout: Mapping[str, Any] | None = None,
    ) -> "EventBank":
        """Load only after schema, hash, member, dtype and shape validation."""

        directory = Path(directory)
        manifest = EventBankManifest.read(directory / MANIFEST_FILENAME)
        if manifest.schema != EVENT_BANK_SCHEMA or manifest.version != EVENT_BANK_SCHEMA_VERSION:
            raise ManifestError(
                f"unsupported event-bank schema {manifest.schema!r} version {manifest.version}"
            )

        for field, expected, actual in (
            ("action_normalizer", expected_action_normalizer, manifest.action_normalizer),
            ("encoder", expected_encoder, manifest.encoder),
            ("camera_layout", expected_camera_layout, manifest.camera_layout),
        ):
            if expected is not None:
                try:
                    canonical = json.loads(
                        json.dumps(expected, sort_keys=True, allow_nan=False)
                    )
                except (TypeError, ValueError) as exc:
                    raise TypeError(f"expected {field} must be finite JSON metadata") from exc
                if canonical != dict(actual):
                    raise ManifestError(f"{field} does not match the requested model contract")

        payload_path = directory / manifest.payload_file
        if not payload_path.is_file():
            raise FileNotFoundError(payload_path)
        actual_file_hash = sha256_file(payload_path)
        if actual_file_hash != manifest.content_hashes[manifest.payload_file]:
            raise IntegrityError(
                f"SHA-256 mismatch for {manifest.payload_file}: "
                f"expected {manifest.content_hashes[manifest.payload_file]}, got {actual_file_hash}"
            )

        arrays: dict[str, np.ndarray] = {}
        try:
            with np.load(payload_path, allow_pickle=False) as archive:
                actual_names = set(archive.files)
                expected_names = set(manifest.arrays)
                if actual_names != expected_names:
                    raise ManifestError(
                        "NPZ members do not match manifest; "
                        f"missing={sorted(expected_names - actual_names)}, "
                        f"extra={sorted(actual_names - expected_names)}"
                    )
                for name, spec in manifest.arrays.items():
                    array = archive[name]
                    spec.validate(name, array)
                    actual_hash = sha256_array(array)
                    expected_hash = manifest.content_hashes[f"array:{name}"]
                    if actual_hash != expected_hash:
                        raise IntegrityError(
                            f"SHA-256 mismatch for array {name!r}: "
                            f"expected {expected_hash}, got {actual_hash}"
                        )
                    arrays[name] = np.array(array, copy=True, order="C")
        except (OSError, ValueError) as exc:
            if isinstance(exc, (ManifestError, IntegrityError)):
                raise
            raise EventBankError(f"cannot load NumPy payload {payload_path}") from exc

        missing_required = _REQUIRED_ARRAYS - set(arrays)
        if missing_required:
            raise ManifestError(f"payload is missing required arrays {sorted(missing_required)}")
        count = manifest.num_events
        for name in (_ID_DATASET_INDEX, _ID_EPISODE_INDEX, _ID_START_FRAME):
            if arrays[name].dtype != np.dtype(np.int64) or arrays[name].shape != (count,):
                raise EventBankError(f"{name} must have dtype int64 and shape {(count,)}")
        dataset_ids = _decode_dataset_ids(
            arrays[_ID_DATASET_BYTES], arrays[_ID_DATASET_OFFSETS], count
        )
        event_ids = tuple(
            EventId(dataset_id, dataset_index, episode_index, start_frame)
            for dataset_id, dataset_index, episode_index, start_frame in zip(
                dataset_ids,
                arrays[_ID_DATASET_INDEX].tolist(),
                arrays[_ID_EPISODE_INDEX].tolist(),
                arrays[_ID_START_FRAME].tolist(),
                strict=True,
            )
        )
        payloads = {
            name: array
            for name, array in arrays.items()
            if name not in _REQUIRED_ARRAYS
        }
        bank = cls(event_ids, arrays[_CONTEXT_KEY], payloads)
        bank._manifest = manifest
        return bank
