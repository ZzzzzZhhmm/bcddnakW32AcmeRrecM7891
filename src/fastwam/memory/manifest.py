"""Versioned, self-validating manifest for a small WARM event bank."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .schema import EVENT_BANK_SCHEMA, EVENT_BANK_SCHEMA_VERSION


class ManifestError(ValueError):
    """Raised when a manifest violates the versioned event-bank contract."""


def _plain_json_mapping(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    try:
        # ``EventBankManifest`` exposes immutable ``mappingproxy`` views; turn
        # the top level back into a plain object before canonical JSON copying.
        serialized = json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        result = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field} must contain only finite JSON values") from exc
    if not isinstance(result, dict) or any(not isinstance(key, str) for key in result):
        raise ManifestError(f"{field} must be a JSON object with string keys")
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path_tree(path: str | Path) -> tuple[str, int]:
    """Hash one file or a deterministic directory tree.

    Directory identity is domain-separated and includes every regular file's
    POSIX relative path, byte size, and SHA-256.  This is the canonical WARM
    checkpoint-tree recipe used by both offline feature precompute and online
    frozen-teacher retrieval.  Callers that load a mutable tree must compare a
    second hash after loading to close the ordinary replacement window.
    """

    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        return sha256_file(resolved), 1
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)

    rows: list[dict[str, object]] = []
    for candidate in sorted(
        resolved.rglob("*"),
        key=lambda item: item.relative_to(resolved).as_posix(),
    ):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(resolved).as_posix()
        before = candidate.stat()
        file_sha256 = sha256_file(candidate)
        after = candidate.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ManifestError(
                f"artifact file changed while it was hashed: {candidate}"
            )
        rows.append(
            {
                "path": relative,
                "size": after.st_size,
                "sha256": file_sha256,
            }
        )
    if not rows:
        raise ManifestError(f"artifact tree is empty: {resolved}")
    encoded = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(b"warm.checkpoint-tree.v1\0" + encoded).hexdigest()
    return digest, len(rows)


def sha256_array(array: np.ndarray) -> str:
    """Hash array content together with its exact dtype and shape."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def sha256_canonical_json(value: Mapping[str, Any]) -> str:
    """Hash one finite JSON object using WARM's canonical serialization."""

    plain = _plain_json_mapping(value, "value")
    encoded = json.dumps(
        plain,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """The exact on-disk contract for one NPZ member."""

    dtype: str
    shape: tuple[int, ...]
    order: str = "C"

    def __post_init__(self) -> None:
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ManifestError(f"invalid NumPy dtype {self.dtype!r}") from exc
        if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
            raise ManifestError("object, structured, and subarray dtypes are not supported")
        if dtype.kind not in "biuf":
            raise ManifestError(f"unsupported array dtype {dtype}")
        if not isinstance(self.shape, tuple):
            raise TypeError("array shape must be a tuple")
        normalized_shape: list[int] = []
        for dimension in self.shape:
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError("array dimensions must be integers")
            if dimension < 0:
                raise ManifestError("array dimensions must be non-negative")
            normalized_shape.append(dimension)
        if self.order != "C":
            raise ManifestError("only contiguous C-order arrays are supported")
        object.__setattr__(self, "dtype", dtype.str)
        object.__setattr__(self, "shape", tuple(normalized_shape))

    @classmethod
    def from_array(cls, array: np.ndarray) -> "ArraySpec":
        if not isinstance(array, np.ndarray):
            raise TypeError("array spec source must be a NumPy array")
        if not array.flags.c_contiguous:
            raise ManifestError("array spec source must be C-contiguous")
        return cls(dtype=array.dtype.str, shape=tuple(array.shape))

    def validate(self, name: str, array: np.ndarray) -> None:
        if not isinstance(array, np.ndarray):
            raise ManifestError(f"{name} is not a NumPy array")
        if array.dtype.str != self.dtype:
            raise ManifestError(
                f"{name} dtype mismatch: manifest={self.dtype}, payload={array.dtype.str}"
            )
        if tuple(array.shape) != self.shape:
            raise ManifestError(
                f"{name} shape mismatch: manifest={self.shape}, payload={tuple(array.shape)}"
            )
        if not array.flags.c_contiguous:
            raise ManifestError(f"{name} must be C-contiguous")

    def to_dict(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "shape": list(self.shape), "order": self.order}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArraySpec":
        if not isinstance(value, Mapping):
            raise TypeError("array spec must be a mapping")
        expected = {"dtype", "shape", "order"}
        if set(value) != expected:
            raise ManifestError(f"array spec fields must be exactly {sorted(expected)}")
        shape = value["shape"]
        if not isinstance(shape, list):
            raise ManifestError("array spec shape must be a JSON list")
        return cls(dtype=value["dtype"], shape=tuple(shape), order=value["order"])


@dataclass(frozen=True, slots=True)
class EventBankManifest:
    """Portable metadata and integrity contract for one immutable bank."""

    action_normalizer: Mapping[str, Any]
    encoder: Mapping[str, Any]
    camera_layout: Mapping[str, Any]
    provenance: Mapping[str, Any]
    arrays: Mapping[str, ArraySpec]
    content_hashes: Mapping[str, str]
    num_events: int
    schema: str = EVENT_BANK_SCHEMA
    version: int = EVENT_BANK_SCHEMA_VERSION
    payload_file: str = "events.npz"

    def __post_init__(self) -> None:
        if not isinstance(self.schema, str) or not self.schema:
            raise ManifestError("schema must be a non-empty string")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ManifestError("version must be a positive integer")
        if isinstance(self.num_events, bool) or not isinstance(self.num_events, int):
            raise TypeError("num_events must be an integer")
        if self.num_events < 0:
            raise ManifestError("num_events must be non-negative")
        if not isinstance(self.payload_file, str) or not self.payload_file:
            raise ManifestError("payload_file must be a non-empty filename")
        if Path(self.payload_file).name != self.payload_file:
            raise ManifestError("payload_file must not contain directories")

        action_normalizer = _plain_json_mapping(self.action_normalizer, "action_normalizer")
        encoder = _plain_json_mapping(self.encoder, "encoder")
        camera_layout = _plain_json_mapping(self.camera_layout, "camera_layout")
        provenance = _plain_json_mapping(self.provenance, "provenance")

        if not isinstance(self.arrays, Mapping) or not self.arrays:
            raise ManifestError("arrays must be a non-empty mapping")
        arrays: dict[str, ArraySpec] = {}
        for name, spec in self.arrays.items():
            if not isinstance(name, str) or not name:
                raise ManifestError("array names must be non-empty strings")
            if not isinstance(spec, ArraySpec):
                raise TypeError(f"array spec for {name!r} must be ArraySpec")
            arrays[name] = spec

        if not isinstance(self.content_hashes, Mapping):
            raise TypeError("content_hashes must be a mapping")
        hashes: dict[str, str] = {}
        for name, digest in self.content_hashes.items():
            if not isinstance(name, str) or not name:
                raise ManifestError("content hash names must be non-empty strings")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ManifestError(f"invalid SHA-256 digest for {name!r}")
            hashes[name] = digest
        if self.payload_file not in hashes:
            raise ManifestError(f"content_hashes must include {self.payload_file!r}")
        for name in arrays:
            if f"array:{name}" not in hashes:
                raise ManifestError(f"content_hashes must include array:{name!s}")

        object.__setattr__(self, "action_normalizer", MappingProxyType(action_normalizer))
        object.__setattr__(self, "encoder", MappingProxyType(encoder))
        object.__setattr__(self, "camera_layout", MappingProxyType(camera_layout))
        object.__setattr__(self, "provenance", MappingProxyType(provenance))
        object.__setattr__(self, "arrays", MappingProxyType(dict(sorted(arrays.items()))))
        object.__setattr__(self, "content_hashes", MappingProxyType(dict(sorted(hashes.items()))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "payload_file": self.payload_file,
            "num_events": self.num_events,
            "action_normalizer": _plain_json_mapping(
                self.action_normalizer, "action_normalizer"
            ),
            "encoder": _plain_json_mapping(self.encoder, "encoder"),
            "camera_layout": _plain_json_mapping(self.camera_layout, "camera_layout"),
            "provenance": _plain_json_mapping(self.provenance, "provenance"),
            "arrays": {name: spec.to_dict() for name, spec in self.arrays.items()},
            "content_hashes": dict(self.content_hashes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EventBankManifest":
        if not isinstance(value, Mapping):
            raise TypeError("manifest must be a mapping")
        expected = {
            "schema",
            "version",
            "payload_file",
            "num_events",
            "action_normalizer",
            "encoder",
            "camera_layout",
            "provenance",
            "arrays",
            "content_hashes",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise ManifestError(f"invalid manifest fields; missing={missing}, extra={extra}")
        arrays_value = value["arrays"]
        if not isinstance(arrays_value, Mapping):
            raise ManifestError("arrays must be a JSON object")
        arrays = {name: ArraySpec.from_dict(spec) for name, spec in arrays_value.items()}
        return cls(
            schema=value["schema"],
            version=value["version"],
            payload_file=value["payload_file"],
            num_events=value["num_events"],
            action_normalizer=value["action_normalizer"],
            encoder=value["encoder"],
            camera_layout=value["camera_layout"],
            provenance=value["provenance"],
            arrays=arrays,
            content_hashes=value["content_hashes"],
        )

    def write(self, path: str | Path) -> None:
        path = Path(path)
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        )
        with path.open("wb") as handle:
            handle.write((encoded + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())

    @classmethod
    def read(cls, path: str | Path) -> "EventBankManifest":
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ManifestError(f"cannot read manifest {path}") from exc
        return cls.from_dict(value)
