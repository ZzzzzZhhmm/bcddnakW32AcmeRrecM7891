"""Auditable, episode-level feature caches for offline WARM preprocessing.

This module defines only the cache contract.  It does not run DINO, a VAE, or
any other encoder.  A cache consists of one numeric ``.npz`` payload and one
``.manifest.json`` sidecar.  The manifest is published last so a reader never
accepts a payload before its complete integrity contract is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from .bank_builder import EpisodeFeatures
from .manifest import ArraySpec, ManifestError, sha256_array, sha256_file
from .schema import EventId


FEATURE_CACHE_SCHEMA = "warm.episode-feature-cache"
FEATURE_CACHE_SCHEMA_VERSION = 3

_METADATA_BYTES = "_metadata_json_utf8"
_MODEL_ACTIONS = "model_actions"
_PROPRIO = "proprio"
_GRIPPER = "gripper"
_CONTEXT_KEYS = "context_keys"
_SEMANTIC_FEATURES = "semantic_features"
_VAE_FEATURES = "vae_features"
_REQUIRED_ARRAYS = frozenset(
    {
        _METADATA_BYTES,
        _MODEL_ACTIONS,
        _PROPRIO,
        _GRIPPER,
        _CONTEXT_KEYS,
        _SEMANTIC_FEATURES,
    }
)
_ALLOWED_ARRAYS = _REQUIRED_ARRAYS | {_VAE_FEATURES}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FeatureCacheError(ValueError):
    """Raised when an episode cache violates its schema or array contract."""


class FeatureCacheIntegrityError(FeatureCacheError):
    """Raised when file or per-array content differs from its SHA-256."""


class FeatureCacheMismatchError(FeatureCacheError):
    """Raised when cache provenance differs from the requested contract."""


def _validated_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class FeatureCacheMetadata:
    """Identity and model provenance required to reuse one episode cache."""

    dataset_id: str
    dataset_index: int
    episode_index: int
    task_index: int
    split: str
    source_episode_sha256: str
    catalog_hash: str
    normalizer_hash: str
    encoder_hash: str
    camera_hash: str

    def __post_init__(self) -> None:
        identity = EventId(self.dataset_id, self.dataset_index, self.episode_index, 0)
        if isinstance(self.task_index, bool) or not isinstance(
            self.task_index, (int, np.integer)
        ):
            raise TypeError("task_index must be a non-negative integer")
        task_index = int(self.task_index)
        if task_index < 0:
            raise ValueError("task_index must be non-negative")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")

        object.__setattr__(self, "dataset_id", identity.dataset_id)
        object.__setattr__(self, "dataset_index", identity.dataset_index)
        object.__setattr__(self, "episode_index", identity.episode_index)
        object.__setattr__(self, "task_index", task_index)
        object.__setattr__(self, "split", self.split)
        for field in (
            "source_episode_sha256",
            "catalog_hash",
            "normalizer_hash",
            "encoder_hash",
            "camera_hash",
        ):
            object.__setattr__(self, field, _validated_sha256(getattr(self, field), field))

    @classmethod
    def for_episode(
        cls,
        episode: EpisodeFeatures,
        *,
        split: str,
        catalog_hash: str,
        normalizer_hash: str,
        encoder_hash: str,
        camera_hash: str,
    ) -> "FeatureCacheMetadata":
        if not isinstance(episode, EpisodeFeatures):
            raise TypeError("episode must be EpisodeFeatures")
        return cls(
            dataset_id=episode.dataset_id,
            dataset_index=episode.dataset_index,
            episode_index=episode.episode_index,
            task_index=episode.task_index,
            split=split,
            source_episode_sha256=episode.source_episode_sha256,
            catalog_hash=catalog_hash,
            normalizer_hash=normalizer_hash,
            encoder_hash=encoder_hash,
            camera_hash=camera_hash,
        )

    def validate_episode(self, episode: EpisodeFeatures) -> None:
        if not isinstance(episode, EpisodeFeatures):
            raise TypeError("episode must be EpisodeFeatures")
        actual = (
            episode.dataset_id,
            episode.dataset_index,
            episode.episode_index,
            episode.task_index,
            episode.source_episode_sha256,
        )
        expected = (
            self.dataset_id,
            self.dataset_index,
            self.episode_index,
            self.task_index,
            self.source_episode_sha256,
        )
        if actual != expected:
            raise FeatureCacheMismatchError(
                f"metadata episode identity {expected!r} does not match features {actual!r}"
            )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_index": self.dataset_index,
            "episode_index": self.episode_index,
            "task_index": self.task_index,
            "split": self.split,
            "source_episode_sha256": self.source_episode_sha256,
            "catalog_hash": self.catalog_hash,
            "normalizer_hash": self.normalizer_hash,
            "encoder_hash": self.encoder_hash,
            "camera_hash": self.camera_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureCacheMetadata":
        if not isinstance(value, Mapping):
            raise TypeError("feature-cache metadata must be a mapping")
        expected = {
            "dataset_id",
            "dataset_index",
            "episode_index",
            "task_index",
            "split",
            "source_episode_sha256",
            "catalog_hash",
            "normalizer_hash",
            "encoder_hash",
            "camera_hash",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise FeatureCacheError(
                f"invalid metadata fields; missing={missing}, extra={extra}"
            )
        return cls(**{name: value[name] for name in expected})


def _metadata_bytes(metadata: FeatureCacheMetadata) -> bytes:
    return json.dumps(
        metadata.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_feature_content_hash(array_hashes: Mapping[str, str]) -> str:
    """Combine data-array digests into an identity-independent episode hash.

    The embedded metadata member is deliberately excluded, so byte-identical
    episode features stored under different dataset/episode identities produce
    the same digest.  Names, optional-VAE presence, and every data-array digest
    are domain-separated and canonically JSON encoded.
    """

    if not isinstance(array_hashes, Mapping):
        raise TypeError("array_hashes must be a mapping")
    data_hashes: dict[str, str] = {}
    for name, digest in array_hashes.items():
        if not isinstance(name, str) or not name:
            raise FeatureCacheError("array-hash names must be non-empty strings")
        if name == _METADATA_BYTES:
            _validated_sha256(digest, f"array_hashes[{name!r}]")
            continue
        if name not in _ALLOWED_ARRAYS:
            raise FeatureCacheError(f"unknown feature data array {name!r}")
        data_hashes[name] = _validated_sha256(
            digest, f"array_hashes[{name!r}]"
        )

    required_data = _REQUIRED_ARRAYS - {_METADATA_BYTES}
    names = set(data_hashes)
    if not required_data.issubset(names) or not names.issubset(
        _ALLOWED_ARRAYS - {_METADATA_BYTES}
    ):
        raise FeatureCacheError(
            "invalid feature data hashes; "
            f"missing={sorted(required_data - names)}, "
            f"extra={sorted(names - (_ALLOWED_ARRAYS - {_METADATA_BYTES}))}"
        )

    encoded = json.dumps(
        {
            "domain": "warm.episode-feature-content.v1",
            "arrays": dict(sorted(data_hashes.items())),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureCacheManifest:
    """Sidecar contract for an immutable episode feature payload."""

    payload_file: str
    metadata: FeatureCacheMetadata
    arrays: Mapping[str, ArraySpec]
    array_hashes: Mapping[str, str]
    episode_content_hash: str
    payload_hash: str
    schema: str = FEATURE_CACHE_SCHEMA
    version: int = FEATURE_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != FEATURE_CACHE_SCHEMA:
            raise FeatureCacheError(f"unsupported feature-cache schema {self.schema!r}")
        if self.version != FEATURE_CACHE_SCHEMA_VERSION:
            raise FeatureCacheError(
                f"unsupported feature-cache version {self.version!r}"
            )
        if not isinstance(self.payload_file, str) or not self.payload_file:
            raise FeatureCacheError("payload_file must be a non-empty filename")
        if Path(self.payload_file).name != self.payload_file:
            raise FeatureCacheError("payload_file must not contain directories")
        if Path(self.payload_file).suffix != ".npz":
            raise FeatureCacheError("payload_file must use the .npz suffix")
        if not isinstance(self.metadata, FeatureCacheMetadata):
            raise TypeError("metadata must be FeatureCacheMetadata")

        if not isinstance(self.arrays, Mapping):
            raise TypeError("arrays must be a mapping")
        arrays: dict[str, ArraySpec] = {}
        for name, spec in self.arrays.items():
            if not isinstance(name, str) or not name:
                raise FeatureCacheError("array names must be non-empty strings")
            if not isinstance(spec, ArraySpec):
                raise TypeError(f"array spec for {name!r} must be ArraySpec")
            arrays[name] = spec
        names = set(arrays)
        if not _REQUIRED_ARRAYS.issubset(names) or not names.issubset(_ALLOWED_ARRAYS):
            raise FeatureCacheError(
                "invalid cache arrays; "
                f"missing={sorted(_REQUIRED_ARRAYS - names)}, "
                f"extra={sorted(names - _ALLOWED_ARRAYS)}"
            )

        if not isinstance(self.array_hashes, Mapping):
            raise TypeError("array_hashes must be a mapping")
        if set(self.array_hashes) != names:
            raise FeatureCacheError("array_hashes keys must exactly match arrays")
        array_hashes = {
            name: _validated_sha256(self.array_hashes[name], f"array_hashes[{name!r}]")
            for name in sorted(names)
        }
        episode_content_hash = _validated_sha256(
            self.episode_content_hash, "episode_content_hash"
        )
        expected_content_hash = canonical_feature_content_hash(array_hashes)
        if episode_content_hash != expected_content_hash:
            raise FeatureCacheIntegrityError(
                "episode_content_hash does not match the canonical data-array hashes"
            )

        metadata_spec = arrays[_METADATA_BYTES]
        if metadata_spec.dtype != np.dtype(np.uint8).str or len(metadata_spec.shape) != 1:
            raise FeatureCacheError(f"{_METADATA_BYTES} must be a one-dimensional uint8 array")

        object.__setattr__(self, "arrays", MappingProxyType(dict(sorted(arrays.items()))))
        object.__setattr__(self, "array_hashes", MappingProxyType(array_hashes))
        object.__setattr__(self, "episode_content_hash", episode_content_hash)
        object.__setattr__(self, "payload_hash", _validated_sha256(self.payload_hash, "payload_hash"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "payload_file": self.payload_file,
            "metadata": self.metadata.to_dict(),
            "arrays": {name: spec.to_dict() for name, spec in self.arrays.items()},
            "array_hashes": dict(self.array_hashes),
            "episode_content_hash": self.episode_content_hash,
            "payload_hash": self.payload_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeatureCacheManifest":
        if not isinstance(value, Mapping):
            raise TypeError("feature-cache manifest must be a mapping")
        expected = {
            "schema",
            "version",
            "payload_file",
            "metadata",
            "arrays",
            "array_hashes",
            "episode_content_hash",
            "payload_hash",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise FeatureCacheError(
                f"invalid manifest fields; missing={missing}, extra={extra}"
            )
        arrays_value = value["arrays"]
        if not isinstance(arrays_value, Mapping):
            raise FeatureCacheError("arrays must be a JSON object")
        try:
            arrays = {
                name: ArraySpec.from_dict(spec) for name, spec in arrays_value.items()
            }
        except (ManifestError, TypeError, ValueError) as exc:
            raise FeatureCacheError("invalid feature-cache array specification") from exc
        return cls(
            schema=value["schema"],
            version=value["version"],
            payload_file=value["payload_file"],
            metadata=FeatureCacheMetadata.from_dict(value["metadata"]),
            arrays=arrays,
            array_hashes=value["array_hashes"],
            episode_content_hash=value["episode_content_hash"],
            payload_hash=value["payload_hash"],
        )

    def write(self, path: str | Path) -> None:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        with Path(path).open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    @classmethod
    def read(cls, path: str | Path) -> "FeatureCacheManifest":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls.from_dict(value)
        except FeatureCacheError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise FeatureCacheError(f"cannot read feature-cache manifest {path}") from exc


@dataclass(frozen=True, slots=True)
class LoadedFeatureCache:
    """One verified cache together with its immutable provenance contract."""

    features: EpisodeFeatures
    metadata: FeatureCacheMetadata
    manifest: FeatureCacheManifest
    payload_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.features, EpisodeFeatures):
            raise TypeError("features must be EpisodeFeatures")
        if not isinstance(self.metadata, FeatureCacheMetadata):
            raise TypeError("metadata must be FeatureCacheMetadata")
        if not isinstance(self.manifest, FeatureCacheManifest):
            raise TypeError("manifest must be FeatureCacheManifest")
        if self.manifest.metadata != self.metadata:
            raise FeatureCacheMismatchError(
                "loaded metadata does not match the feature-cache manifest"
            )
        self.metadata.validate_episode(self.features)
        if self.features.feature_episode_sha256 != self.manifest.episode_content_hash:
            raise FeatureCacheMismatchError(
                "loaded features are not bound to the manifest episode_content_hash"
            )

        path = Path(self.payload_path).resolve()
        if path.name != self.manifest.payload_file:
            raise FeatureCacheMismatchError(
                "loaded payload path does not match the feature-cache manifest"
            )
        for name in (
            "model_actions",
            "proprio",
            "gripper",
            "context_keys",
            "semantic_features",
        ):
            if getattr(self.features, name).flags.writeable:
                raise FeatureCacheError(f"loaded feature array {name!r} must be read-only")
        if (
            self.features.vae_features is not None
            and self.features.vae_features.flags.writeable
        ):
            raise FeatureCacheError("loaded feature array 'vae_features' must be read-only")
        object.__setattr__(self, "payload_path", path)

    @property
    def episode_content_hash(self) -> str:
        return self.manifest.episode_content_hash


# Descriptive alias for callers that prefer the data-centric name.
CachedEpisodeFeatures = LoadedFeatureCache


def feature_cache_manifest_path(payload_path: str | Path) -> Path:
    """Return ``<stem>.manifest.json`` for a strict ``.npz`` payload path."""

    path = Path(payload_path)
    if path.suffix != ".npz":
        raise ValueError("episode feature-cache payload path must use the .npz suffix")
    return path.with_suffix(".manifest.json")


def _validated_episode_snapshot(episode: EpisodeFeatures) -> EpisodeFeatures:
    """Copy and reconstruct an episode to catch mutation after construction."""

    if not isinstance(episode, EpisodeFeatures):
        raise TypeError("episode must be EpisodeFeatures")
    return EpisodeFeatures(
        dataset_id=episode.dataset_id,
        dataset_index=episode.dataset_index,
        episode_index=episode.episode_index,
        task_index=episode.task_index,
        source_episode_sha256=episode.source_episode_sha256,
        model_actions=np.array(episode.model_actions, copy=True, order="C"),
        proprio=np.array(episode.proprio, copy=True, order="C"),
        gripper=np.array(episode.gripper, copy=True, order="C"),
        context_keys=np.array(episode.context_keys, copy=True, order="C"),
        semantic_features=np.array(episode.semantic_features, copy=True, order="C"),
        vae_features=(
            None
            if episode.vae_features is None
            else np.array(episode.vae_features, copy=True, order="C")
        ),
        feature_episode_sha256=episode.feature_episode_sha256,
    )


def _storage_arrays(
    episode: EpisodeFeatures,
    metadata: FeatureCacheMetadata,
) -> dict[str, np.ndarray]:
    arrays = {
        _METADATA_BYTES: np.frombuffer(_metadata_bytes(metadata), dtype=np.uint8).copy(),
        _MODEL_ACTIONS: np.ascontiguousarray(episode.model_actions),
        _PROPRIO: np.ascontiguousarray(episode.proprio),
        _GRIPPER: np.ascontiguousarray(episode.gripper),
        _CONTEXT_KEYS: np.ascontiguousarray(episode.context_keys),
        _SEMANTIC_FEATURES: np.ascontiguousarray(episode.semantic_features),
    }
    if episode.vae_features is not None:
        arrays[_VAE_FEATURES] = np.ascontiguousarray(episode.vae_features)
    return dict(sorted(arrays.items()))


def save_episode_feature_cache(
    payload_path: str | Path,
    episode: EpisodeFeatures,
    *,
    metadata: FeatureCacheMetadata,
) -> FeatureCacheManifest:
    """Atomically publish one validated, immutable episode cache."""

    payload_path = Path(payload_path)
    manifest_path = feature_cache_manifest_path(payload_path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    if payload_path.exists() or manifest_path.exists():
        raise FileExistsError(f"episode feature cache already exists at {payload_path}")
    if not isinstance(metadata, FeatureCacheMetadata):
        raise TypeError("metadata must be FeatureCacheMetadata")

    snapshot = _validated_episode_snapshot(episode)
    metadata.validate_episode(snapshot)
    arrays = _storage_arrays(snapshot, metadata)

    temporary_payload = payload_path.parent / f".{payload_path.name}.{uuid4().hex}.tmp"
    temporary_manifest = manifest_path.parent / f".{manifest_path.name}.{uuid4().hex}.tmp"
    try:
        with temporary_payload.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())

        array_hashes = {
            name: sha256_array(array) for name, array in arrays.items()
        }
        manifest = FeatureCacheManifest(
            payload_file=payload_path.name,
            metadata=metadata,
            arrays={name: ArraySpec.from_array(array) for name, array in arrays.items()},
            array_hashes=array_hashes,
            episode_content_hash=canonical_feature_content_hash(array_hashes),
            payload_hash=sha256_file(temporary_payload),
        )
        if (
            snapshot.feature_episode_sha256 is not None
            and snapshot.feature_episode_sha256 != manifest.episode_content_hash
        ):
            raise FeatureCacheMismatchError(
                "episode feature_episode_sha256 does not match encoded array content"
            )
        manifest.write(temporary_manifest)

        # Publish the contract last.  A crash between replacements leaves an
        # obvious hash mismatch rather than a silently accepted partial cache.
        os.replace(temporary_payload, payload_path)
        os.replace(temporary_manifest, manifest_path)
        return manifest
    finally:
        temporary_payload.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


def _metadata_from_payload(array: np.ndarray) -> FeatureCacheMetadata:
    if array.dtype != np.dtype(np.uint8) or array.ndim != 1:
        raise FeatureCacheError(f"{_METADATA_BYTES} must be a one-dimensional uint8 array")
    try:
        raw = bytes(memoryview(np.ascontiguousarray(array)).cast("B"))
        value = json.loads(raw.decode("utf-8", errors="strict"))
        return FeatureCacheMetadata.from_dict(value)
    except FeatureCacheError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise FeatureCacheError("embedded feature-cache metadata is invalid") from exc


def load_episode_feature_cache(
    payload_path: str | Path,
    *,
    expected_metadata: FeatureCacheMetadata | None = None,
) -> LoadedFeatureCache:
    """Load after schema, provenance, file, member, dtype, and hash checks."""

    payload_path = Path(payload_path)
    manifest_path = feature_cache_manifest_path(payload_path)
    manifest = FeatureCacheManifest.read(manifest_path)
    if manifest.payload_file != payload_path.name:
        raise FeatureCacheMismatchError(
            f"manifest payload {manifest.payload_file!r} does not match {payload_path.name!r}"
        )
    if expected_metadata is not None:
        if not isinstance(expected_metadata, FeatureCacheMetadata):
            raise TypeError("expected_metadata must be FeatureCacheMetadata or None")
        if manifest.metadata != expected_metadata:
            raise FeatureCacheMismatchError(
                "feature-cache metadata does not match the requested provenance"
            )
    if not payload_path.is_file():
        raise FileNotFoundError(payload_path)

    actual_payload_hash = sha256_file(payload_path)
    if actual_payload_hash != manifest.payload_hash:
        raise FeatureCacheIntegrityError(
            f"SHA-256 mismatch for {payload_path.name}: "
            f"expected {manifest.payload_hash}, got {actual_payload_hash}"
        )

    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(payload_path, allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise FeatureCacheError("NPZ payload contains duplicate member names")
            actual_names = set(archive.files)
            expected_names = set(manifest.arrays)
            if actual_names != expected_names:
                raise FeatureCacheError(
                    "NPZ members do not match manifest; "
                    f"missing={sorted(expected_names - actual_names)}, "
                    f"extra={sorted(actual_names - expected_names)}"
                )
            for name, spec in manifest.arrays.items():
                array = archive[name]
                try:
                    spec.validate(name, array)
                except ManifestError as exc:
                    raise FeatureCacheError(
                        f"array {name!r} violates its dtype/shape contract"
                    ) from exc
                actual_array_hash = sha256_array(array)
                if actual_array_hash != manifest.array_hashes[name]:
                    raise FeatureCacheIntegrityError(
                        f"SHA-256 mismatch for array {name!r}: "
                        f"expected {manifest.array_hashes[name]}, got {actual_array_hash}"
                    )
                arrays[name] = np.array(array, copy=True, order="C")
    except (FeatureCacheError, FeatureCacheIntegrityError):
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise FeatureCacheError(f"cannot load NumPy payload {payload_path}") from exc

    embedded_metadata = _metadata_from_payload(arrays.pop(_METADATA_BYTES))
    if embedded_metadata != manifest.metadata:
        raise FeatureCacheMismatchError(
            "embedded metadata does not match the feature-cache manifest"
        )

    try:
        episode = EpisodeFeatures(
            dataset_id=manifest.metadata.dataset_id,
            dataset_index=manifest.metadata.dataset_index,
            episode_index=manifest.metadata.episode_index,
            task_index=manifest.metadata.task_index,
            source_episode_sha256=manifest.metadata.source_episode_sha256,
            model_actions=arrays[_MODEL_ACTIONS],
            proprio=arrays[_PROPRIO],
            gripper=arrays[_GRIPPER],
            context_keys=arrays[_CONTEXT_KEYS],
            semantic_features=arrays[_SEMANTIC_FEATURES],
            vae_features=arrays.get(_VAE_FEATURES),
            feature_episode_sha256=manifest.episode_content_hash,
        )
    except (TypeError, ValueError) as exc:
        raise FeatureCacheError("cached arrays fail EpisodeFeatures validation") from exc
    manifest.metadata.validate_episode(episode)
    return LoadedFeatureCache(
        features=episode,
        metadata=manifest.metadata,
        manifest=manifest,
        payload_path=payload_path,
    )


__all__ = [
    "FEATURE_CACHE_SCHEMA",
    "FEATURE_CACHE_SCHEMA_VERSION",
    "CachedEpisodeFeatures",
    "FeatureCacheError",
    "FeatureCacheIntegrityError",
    "FeatureCacheManifest",
    "FeatureCacheMetadata",
    "FeatureCacheMismatchError",
    "LoadedFeatureCache",
    "canonical_feature_content_hash",
    "feature_cache_manifest_path",
    "load_episode_feature_cache",
    "save_episode_feature_cache",
]
