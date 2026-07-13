"""Strict readers for the official RMBench HDF5/JPEG episode layout.

The module intentionally imports HDF5 and image dependencies only inside the
functions that need them.  This keeps metadata tooling importable on CPU-only
machines while failing with a precise message when conversion dependencies are
missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .constants import (
    ACTION_DIM,
    OFFICIAL_CAMERA_KEYS,
    SOURCE_CAMERA_PATHS,
    SOURCE_QPOS_PATH,
)


class RMBenchSourceError(ValueError):
    """Raised when source data violates the frozen RMBench contract."""


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _normalized_instruction(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise RMBenchSourceError(f"{label} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise RMBenchSourceError(f"{label} must not be empty")
    return normalized


def read_episode_instruction(path: str | Path) -> tuple[str, dict[str, tuple[str, ...]], str]:
    """Read an episode instruction deterministically from its exact JSON bytes.

    Official demonstrations provide ``seen`` and ``unseen`` lists.  Training
    uses the first normalized ``seen`` instruction; all variants remain bound
    into the conversion manifest through the source JSON SHA-256.
    """

    source = Path(path)
    payload = source.read_bytes()
    digest = sha256(payload).hexdigest()
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RMBenchSourceError(f"invalid UTF-8 instruction JSON: {source}") from exc
    if not isinstance(document, dict):
        raise RMBenchSourceError(f"instruction JSON must contain an object: {source}")
    unexpected = set(document) - {"seen", "unseen"}
    if unexpected:
        raise RMBenchSourceError(
            f"instruction JSON has unsupported keys {sorted(unexpected)}: {source}"
        )
    variants: dict[str, tuple[str, ...]] = {}
    for kind in ("seen", "unseen"):
        raw = document.get(kind, [])
        if not isinstance(raw, list):
            raise RMBenchSourceError(f"instruction field {kind!r} must be a list")
        variants[kind] = tuple(
            _normalized_instruction(item, label=f"{source}:{kind}[{index}]")
            for index, item in enumerate(raw)
        )
    if not variants["seen"]:
        raise RMBenchSourceError(f"instruction JSON has no seen instruction: {source}")
    return variants["seen"][0], variants, digest


def _jpeg_payload(value: Any, *, label: str) -> bytes:
    if isinstance(value, np.ndarray):
        if value.dtype != np.uint8 or value.ndim != 1:
            raise RMBenchSourceError(f"{label} must be encoded JPEG bytes")
        raw = value.tobytes()
    elif isinstance(value, np.void):
        raw = value.tobytes()
    elif isinstance(value, np.generic):
        raw = value.tobytes()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    else:
        raise RMBenchSourceError(f"{label} must be encoded JPEG bytes")

    if not raw.startswith(b"\xff\xd8"):
        raise RMBenchSourceError(f"{label} is missing the JPEG SOI marker")
    eoi = raw.rfind(b"\xff\xd9")
    if eoi < 2:
        raise RMBenchSourceError(f"{label} is missing the JPEG EOI marker")
    trailing = raw[eoi + 2 :]
    if any(byte != 0 for byte in trailing):
        raise RMBenchSourceError(f"{label} has non-zero bytes after JPEG EOI")
    return raw[: eoi + 2]


def decode_official_jpeg(
    value: Any,
    *,
    label: str,
    expected_height: int,
    expected_width: int,
) -> tuple[np.ndarray, bytes]:
    """Strictly decode one padded JPEG cell from official RMBench HDF5.

    RoboTwin encodes its RGB NumPy buffer with OpenCV.  Decoding with OpenCV
    and retaining the returned numeric channel order reverses that writer-side
    convention and recovers the original RGB-valued buffer used by RoboTwin's
    existing policy converters.
    """

    jpeg = _jpeg_payload(value, label=label)
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Pillow is required for strict RMBench JPEG validation") from exc
    try:
        with Image.open(BytesIO(jpeg)) as image:
            image.verify()
            pil_size = image.size
    except Exception as exc:
        raise RMBenchSourceError(f"{label} failed strict JPEG verification") from exc
    if pil_size != (expected_width, expected_height):
        raise RMBenchSourceError(
            f"{label} has JPEG size {pil_size}, expected "
            f"{(expected_width, expected_height)}"
        )

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("opencv-python is required to decode RMBench JPEG cells") from exc
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RMBenchSourceError(f"{label} could not be decoded by OpenCV")
    expected_shape = (expected_height, expected_width, 3)
    if image.shape != expected_shape or image.dtype != np.uint8:
        raise RMBenchSourceError(
            f"{label} decoded to {image.shape}/{image.dtype}, expected "
            f"{expected_shape}/uint8"
        )
    return np.ascontiguousarray(image), jpeg


@dataclass(frozen=True)
class SourceEpisodeSpec:
    task_name: str
    source_episode_index: int
    hdf5_path: Path
    instruction_path: Path
    instruction: str
    instruction_variants: Mapping[str, tuple[str, ...]]
    instruction_sha256: str
    observation_count: int

    @property
    def transition_count(self) -> int:
        return self.observation_count - 1


def inspect_source_episode(
    *,
    task_name: str,
    source_episode_index: int,
    hdf5_path: str | Path,
    instruction_path: str | Path,
    camera_paths: Mapping[str, str] = SOURCE_CAMERA_PATHS,
) -> SourceEpisodeSpec:
    """Validate source shapes without decoding the large JPEG payloads."""

    hdf5_source = Path(hdf5_path).resolve(strict=True)
    instruction_source = Path(instruction_path).resolve(strict=True)
    instruction, variants, instruction_digest = read_episode_instruction(
        instruction_source
    )
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("h5py is required to inspect RMBench demonstrations") from exc

    with h5py.File(hdf5_source, "r") as episode:
        if SOURCE_QPOS_PATH not in episode:
            raise RMBenchSourceError(
                f"missing {SOURCE_QPOS_PATH} in {hdf5_source}"
            )
        qpos = episode[SOURCE_QPOS_PATH]
        if qpos.ndim != 2 or qpos.shape[1] != ACTION_DIM:
            raise RMBenchSourceError(
                f"{SOURCE_QPOS_PATH} in {hdf5_source} must have shape [N,{ACTION_DIM}]"
            )
        observations = int(qpos.shape[0])
        if observations < 2:
            raise RMBenchSourceError(f"{hdf5_source} must contain at least two observations")
        if not np.issubdtype(qpos.dtype, np.number):
            raise RMBenchSourceError(f"{SOURCE_QPOS_PATH} must be numeric")
        if tuple(camera_paths) != OFFICIAL_CAMERA_KEYS:
            raise RMBenchSourceError("camera contract must use the official three-camera order")
        for camera_key in OFFICIAL_CAMERA_KEYS:
            dataset_path = camera_paths[camera_key]
            if dataset_path not in episode:
                raise RMBenchSourceError(f"missing {dataset_path} in {hdf5_source}")
            camera = episode[dataset_path]
            if camera.ndim != 1 or int(camera.shape[0]) != observations:
                raise RMBenchSourceError(
                    f"{dataset_path} must contain exactly {observations} JPEG cells"
                )
            if camera.dtype.kind not in {"S", "O", "V"}:
                raise RMBenchSourceError(f"{dataset_path} must contain encoded byte cells")

    return SourceEpisodeSpec(
        task_name=task_name,
        source_episode_index=source_episode_index,
        hdf5_path=hdf5_source,
        instruction_path=instruction_source,
        instruction=instruction,
        instruction_variants=variants,
        instruction_sha256=instruction_digest,
        observation_count=observations,
    )


def read_qpos(episode: Any, *, source_label: str) -> np.ndarray:
    qpos = np.asarray(episode[SOURCE_QPOS_PATH][()], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != ACTION_DIM or qpos.shape[0] < 2:
        raise RMBenchSourceError(
            f"{SOURCE_QPOS_PATH} in {source_label} must have shape [N,{ACTION_DIM}]"
        )
    if not np.isfinite(qpos).all():
        raise RMBenchSourceError(f"{SOURCE_QPOS_PATH} in {source_label} is non-finite")
    return np.ascontiguousarray(qpos)


__all__ = [
    "RMBenchSourceError",
    "SourceEpisodeSpec",
    "canonical_json_bytes",
    "decode_official_jpeg",
    "file_sha256",
    "inspect_source_episode",
    "read_episode_instruction",
    "read_qpos",
]
