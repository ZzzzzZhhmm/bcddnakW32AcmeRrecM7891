"""Strict full-episode LeRobot reader for WARM offline preprocessing.

Unlike the training dataset, this reader never samples temporal windows.  It
loads one catalogued parquet payload into memory once, verifies that exact
byte snapshot against the audit proof, and parses the verified bytes.  Heavy
runtime dependencies are imported only by the default adapters so this module
remains importable in lightweight CPU contract-test environments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from .audit import (
    CameraVideoAuditProof,
    EpisodeAuditProof,
    compute_source_bundle_sha256,
    resolve_episode_video_paths,
)
from .episode_catalog import EpisodeRecord


DEFAULT_TIMESTAMP_TOLERANCE_S = 1e-4


class TableReader(Protocol):
    """Parse one already-verified parquet byte snapshot."""

    def __call__(self, payload: bytes) -> Mapping[str, Any]: ...


class VideoDecoder(Protocol):
    """Decode one camera stream at the requested episode timestamps."""

    def __call__(
        self,
        video_path: Path,
        timestamps: Sequence[float],
        tolerance_s: float,
    ) -> Any: ...


def _immutable_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Return an array backed by immutable bytes, not merely a cleared flag."""

    contiguous = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return frozen.reshape(contiguous.shape)


@dataclass(frozen=True, slots=True)
class FullLerobotEpisode:
    """One validated, immutable, unsampled LeRobot episode snapshot."""

    record: EpisodeRecord
    source_episode_sha256: str
    data_path: Path
    actions: np.ndarray
    states: np.ndarray
    images: Mapping[str, np.ndarray]
    timestamps: np.ndarray
    task_indices: np.ndarray

    def __post_init__(self) -> None:
        expected = self.record.length
        if self.actions.shape[0] != expected:
            raise ValueError("actions must retain all N raw parquet rows")
        if self.states.shape[0] != expected:
            raise ValueError("states must retain all N parquet rows")
        if self.timestamps.shape != (expected,):
            raise ValueError("timestamps must have shape [N]")
        if self.task_indices.shape != (expected,):
            raise ValueError("task_indices must have shape [N]")
        if not isinstance(self.images, MappingProxyType):
            raise TypeError("images must be an immutable mapping")
        for camera, frames in self.images.items():
            if not camera:
                raise ValueError("camera names must be non-empty")
            if frames.ndim != 4 or frames.shape[:2] != (expected, 3):
                raise ValueError("each camera must contain float32 frames [N, 3, H, W]")
            if frames.dtype != np.float32 or frames.flags.writeable:
                raise ValueError("camera frames must be immutable float32 arrays")
        for name in ("actions", "states", "timestamps", "task_indices"):
            if getattr(self, name).flags.writeable:
                raise ValueError(f"{name} must be immutable")


def _default_table_reader(payload: bytes) -> Mapping[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised on the GPU server
        raise RuntimeError(
            "Reading parquet requires pyarrow; install the server dependencies "
            "or inject table_reader for contract tests"
        ) from exc

    table = pq.read_table(pa.BufferReader(payload))
    return {name: table[name] for name in table.column_names}


def _default_video_decoder(
    video_path: Path,
    timestamps: Sequence[float],
    tolerance_s: float,
    *,
    backend: str,
) -> Any:
    try:
        from .lerobot.datasets.video_utils import decode_video_frames
    except ImportError as exc:  # pragma: no cover - exercised on the GPU server
        raise RuntimeError(
            "Decoding LeRobot videos requires the FastWAM video dependencies; "
            "install them or inject video_decoder for contract tests"
        ) from exc
    return decode_video_frames(
        video_path,
        list(timestamps),
        tolerance_s,
        backend=backend,
    )


def _resolve_inside(root: Path, relative: str | Path, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be relative to the dataset root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the dataset root") from exc
    return resolved


def _sha256_file(path: Path, *, label: str) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FileNotFoundError(f"cannot hash {label}: {path}") from exc
    return digest.hexdigest()


def _column_to_array(column: Any, *, name: str) -> np.ndarray:
    try:
        if hasattr(column, "combine_chunks"):
            column = column.combine_chunks()
        if hasattr(column, "to_pylist"):
            column = column.to_pylist()
        array = np.asarray(column)
    except Exception as exc:
        raise ValueError(f"cannot convert parquet column {name!r} to NumPy") from exc
    if array.ndim == 0:
        raise ValueError(f"parquet column {name!r} must have one value per row")
    if array.dtype.kind == "O":
        try:
            array = np.asarray(list(column))
        except Exception as exc:
            raise ValueError(f"parquet column {name!r} is ragged or non-numeric") from exc
    return array


def _scalar_column(column: Any, *, name: str, rows: int) -> np.ndarray:
    array = _column_to_array(column, name=name)
    if array.shape == (rows, 1):
        array = array[:, 0]
    if array.shape != (rows,):
        raise ValueError(f"parquet column {name!r} must have shape [N]")
    return array


def _integer_column(column: Any, *, name: str, rows: int) -> np.ndarray:
    array = _scalar_column(column, name=name, rows=rows)
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"parquet column {name!r} must contain integers")
    return np.asarray(array, dtype=np.int64)


def _numeric_matrix(column: Any, *, name: str, rows: int) -> np.ndarray:
    array = _column_to_array(column, name=name)
    if array.shape[0] != rows:
        raise ValueError(f"parquet column {name!r} has {array.shape[0]} rows, expected {rows}")
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] <= 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"parquet column {name!r} must be a dense numeric [N, D] matrix")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"parquet column {name!r} contains non-finite values")
    return _immutable_array(array, dtype=np.dtype(np.float32))


def _decoder_output_to_numpy(value: Any, *, camera: str, rows: int) -> np.ndarray:
    # Avoid importing torch: this duck-typed path handles a CPU torch.Tensor.
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        frames = np.asarray(value, dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"decoder output for {camera!r} is not a numeric array") from exc
    if frames.ndim != 4 or frames.shape[0] != rows or frames.shape[1] != 3:
        raise ValueError(
            f"decoder output for {camera!r} must have shape [N, 3, H, W], "
            f"got {frames.shape}"
        )
    if frames.shape[2] <= 0 or frames.shape[3] <= 0:
        raise ValueError(f"decoder output for {camera!r} has an empty spatial dimension")
    if not np.isfinite(frames).all():
        raise ValueError(f"decoder output for {camera!r} contains non-finite values")
    if np.any(frames < 0.0) or np.any(frames > 1.0):
        raise ValueError(
            f"decoder output for {camera!r} must contain RGB pixels in [0,1]"
        )
    return _immutable_array(frames, dtype=np.dtype(np.float32))


def read_full_lerobot_episode(
    record: EpisodeRecord,
    *,
    dataset_root: str | Path,
    audit_proof: EpisodeAuditProof,
    camera_keys: Sequence[str],
    action_key: str = "action",
    state_key: str = "observation.state",
    timestamp_tolerance_s: float = DEFAULT_TIMESTAMP_TOLERANCE_S,
    video_backend: str = "pyav",
    table_reader: TableReader | None = None,
    video_decoder: VideoDecoder | None = None,
) -> FullLerobotEpisode:
    """Read exactly one complete, audited episode without sampling or retries.

    ``actions`` deliberately contains all ``N`` parquet rows.  Event-feature
    preprocessing owns the later, explicit ``N -> N-1`` alignment decision.
    """

    if not isinstance(record, EpisodeRecord):
        raise TypeError("record must be an EpisodeRecord")
    if not isinstance(audit_proof, EpisodeAuditProof):
        raise TypeError("audit_proof must be an EpisodeAuditProof")
    expected_key = (record.dataset_id, record.dataset_index, record.episode_index)
    if audit_proof.episode_key != expected_key:
        raise ValueError("audit proof identity does not match the catalog record")
    if audit_proof.split != record.split:
        raise ValueError("audit proof split does not match the catalog record")
    if (
        not isinstance(timestamp_tolerance_s, (int, float))
        or isinstance(timestamp_tolerance_s, bool)
        or not np.isfinite(timestamp_tolerance_s)
        or timestamp_tolerance_s < 0
    ):
        raise ValueError("timestamp_tolerance_s must be a finite non-negative number")
    if not isinstance(action_key, str) or not action_key.strip() or action_key.strip() != action_key:
        raise ValueError("action_key must be a non-empty normalized string")
    if not isinstance(state_key, str) or not state_key.strip() or state_key.strip() != state_key:
        raise ValueError("state_key must be a non-empty normalized string")
    if (
        not isinstance(video_backend, str)
        or not video_backend.strip()
        or video_backend.strip() != video_backend
    ):
        raise ValueError("video_backend must be a non-empty normalized string")

    cameras = tuple(camera_keys)
    if not cameras:
        raise ValueError("at least one camera key is required")
    if any(not isinstance(key, str) or not key or key.strip() != key for key in cameras):
        raise ValueError("camera keys must be non-empty normalized strings")
    if len(set(cameras)) != len(cameras):
        raise ValueError("camera keys must be unique")
    if cameras != audit_proof.camera_keys:
        raise ValueError(
            "requested camera keys must exactly match the audit proof order: "
            f"{cameras!r} != {audit_proof.camera_keys!r}"
        )

    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    data_path = _resolve_inside(root, record.data_relpath, label="episode data path")
    try:
        parquet_snapshot = data_path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"cannot read episode parquet: {data_path}") from exc
    actual_sha256 = sha256(parquet_snapshot).hexdigest()
    if actual_sha256 != audit_proof.table_sha256:
        raise ValueError("episode parquet does not match its audited SHA-256 snapshot")

    reader = table_reader or _default_table_reader
    try:
        table = reader(parquet_snapshot)
    except Exception as exc:
        if isinstance(exc, (RuntimeError, ValueError)):
            raise
        raise ValueError(f"cannot parse audited episode parquet {data_path}") from exc
    if not isinstance(table, Mapping):
        raise TypeError("table_reader must return a mapping of parquet columns")

    required = {
        action_key,
        state_key,
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
    }
    missing = sorted(required.difference(table))
    if missing:
        raise ValueError(f"episode parquet is missing required columns: {missing}")
    rows = record.length
    for name, column in table.items():
        array = _column_to_array(column, name=str(name))
        if array.shape[0] != rows:
            raise ValueError(
                f"episode parquet column {name!r} has {array.shape[0]} rows, expected {rows}"
            )

    episode_indices = _integer_column(table["episode_index"], name="episode_index", rows=rows)
    if not np.all(episode_indices == record.episode_index):
        raise ValueError("episode_index column does not match the catalog episode")
    frame_indices = _integer_column(table["frame_index"], name="frame_index", rows=rows)
    if not np.array_equal(frame_indices, np.arange(rows, dtype=np.int64)):
        raise ValueError("frame_index must be exactly contiguous [0, N)")
    task_indices = _integer_column(table["task_index"], name="task_index", rows=rows)
    if np.any(task_indices < 0):
        raise ValueError("task_index values must be non-negative")
    if np.unique(task_indices).size != 1:
        raise ValueError(
            "task_index must remain constant within one WARM episode"
        )

    timestamps_raw = _scalar_column(table["timestamp"], name="timestamp", rows=rows)
    if timestamps_raw.dtype.kind not in "iuf":
        raise ValueError("timestamp column must be numeric")
    timestamps = np.asarray(timestamps_raw, dtype=np.float64)
    if not np.isfinite(timestamps).all():
        raise ValueError("timestamp column contains non-finite values")
    expected_timestamps = frame_indices.astype(np.float64) / record.fps
    if not np.allclose(
        timestamps,
        expected_timestamps,
        rtol=0.0,
        atol=float(timestamp_tolerance_s),
    ):
        raise ValueError("timestamps must equal frame_index / fps within tolerance")

    actions = _numeric_matrix(table[action_key], name=action_key, rows=rows)
    states = _numeric_matrix(table[state_key], name=state_key, rows=rows)
    immutable_timestamps = _immutable_array(timestamps, dtype=np.dtype(np.float64))
    immutable_task_indices = _immutable_array(task_indices, dtype=np.dtype(np.int64))

    camera_paths = resolve_episode_video_paths(
        root,
        episode_index=record.episode_index,
        camera_keys=cameras,
    )
    audited_camera_hashes = audit_proof.ordered_camera_sha256
    before_camera_hashes: list[CameraVideoAuditProof] = []
    for (camera, video_path), expected_camera in zip(
        camera_paths,
        audited_camera_hashes,
        strict=True,
    ):
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        before_digest = _sha256_file(
            video_path,
            label=f"camera {camera!r} before decode",
        )
        if before_digest != expected_camera.sha256:
            raise ValueError(
                f"camera MP4 {camera!r} does not match its audited SHA-256 before decode"
            )
        before_camera_hashes.append(CameraVideoAuditProof(camera, before_digest))
    before_bundle = compute_source_bundle_sha256(
        actual_sha256,
        before_camera_hashes,
    )
    if before_bundle != audit_proof.source_episode_sha256:
        raise ValueError("episode source bundle does not match its audit proof before decode")

    if video_decoder is None:
        def decoder(
            path: Path,
            requested_timestamps: Sequence[float],
            tolerance: float,
        ) -> Any:
            return _default_video_decoder(
                path,
                requested_timestamps,
                tolerance,
                backend=video_backend,
            )
    else:
        decoder = video_decoder
    decoded: dict[str, np.ndarray] = {}
    after_camera_hashes: list[CameraVideoAuditProof] = []
    for (camera, video_path), expected_camera in zip(
        camera_paths,
        audited_camera_hashes,
        strict=True,
    ):
        # Re-hash immediately before this decoder invocation.  The initial
        # all-camera pass validates the complete bundle; this second check
        # also catches another decoder (or process) replacing a later stream
        # between bundle validation and its own decode.
        immediate_digest = _sha256_file(
            video_path,
            label=f"camera {camera!r} immediately before decode",
        )
        if immediate_digest != expected_camera.sha256:
            raise ValueError(
                f"camera MP4 {camera!r} changed before it could be decoded"
            )
        try:
            frames = decoder(video_path, immutable_timestamps, float(timestamp_tolerance_s))
        except Exception as exc:
            after_digest = _sha256_file(
                video_path,
                label=f"camera {camera!r} after failed decode",
            )
            if after_digest != expected_camera.sha256:
                raise ValueError(
                    f"camera MP4 {camera!r} changed while it was being decoded"
                ) from exc
            if isinstance(exc, (RuntimeError, ValueError, FileNotFoundError)):
                raise
            raise ValueError(f"cannot decode camera {camera!r} from {video_path}") from exc
        after_digest = _sha256_file(
            video_path,
            label=f"camera {camera!r} after decode",
        )
        if after_digest != expected_camera.sha256:
            raise ValueError(
                f"camera MP4 {camera!r} changed while it was being decoded"
            )
        after_camera_hashes.append(CameraVideoAuditProof(camera, after_digest))
        decoded[camera] = _decoder_output_to_numpy(frames, camera=camera, rows=rows)

    after_bundle = compute_source_bundle_sha256(actual_sha256, after_camera_hashes)
    if after_bundle != audit_proof.source_episode_sha256:
        raise ValueError("episode source bundle changed while videos were decoded")

    return FullLerobotEpisode(
        record=record,
        source_episode_sha256=after_bundle,
        data_path=data_path,
        actions=actions,
        states=states,
        images=MappingProxyType(decoded),
        timestamps=immutable_timestamps,
        task_indices=immutable_task_indices,
    )


__all__ = [
    "DEFAULT_TIMESTAMP_TOLERANCE_S",
    "FullLerobotEpisode",
    "TableReader",
    "VideoDecoder",
    "read_full_lerobot_episode",
]
