"""Atomic official RMBench -> FastWAM LeRobot v2.1 conversion.

The source HDF5 stores ``N`` factual robot observations but no separately
recorded command stream.  For qpos control, the only factual transition label
is the next observed qpos.  Consequently each published LeRobot episode has
``N-1`` rows:

``observation.state[t] = qpos[t]`` and ``action[t] = qpos[t+1]``.

The terminal observation is source-validated and source-hashed but is not
invented into a row with a synthetic action.  This convention is recorded in
the immutable conversion manifest.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from fastwam.datasets.lerobot.episode_catalog import (
    EpisodeCatalog,
    scan_lerobot_datasets,
)

from .constants import (
    ACTION_DIM,
    CATALOG_FILENAME,
    CONVERSION_SCHEMA,
    CONVERSION_SCHEMA_VERSION,
    DATA_PATH_TEMPLATE,
    DEFAULT_DEV_PER_TASK,
    DEFAULT_FPS,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_SPLIT_SEED,
    LEROBOT_CHUNK_SIZE,
    LEROBOT_CODEBASE_VERSION,
    MANIFEST_FILENAME,
    MOTOR_NAMES,
    OFFICIAL_CAMERA_KEYS,
    OFFICIAL_EPISODES_PER_TASK,
    OFFICIAL_RMBENCH_TASKS,
    OFFICIAL_TASK_CONFIG,
    SOURCE_CAMERA_PATHS,
    VIDEO_PATH_TEMPLATE,
)
from .source import (
    RMBenchSourceError,
    SourceEpisodeSpec,
    canonical_json_bytes,
    decode_official_jpeg,
    file_sha256,
    inspect_source_episode,
    read_qpos,
)
from .split import deterministic_task_split


_EPISODE_FILE_RE = re.compile(r"episode(?P<index>[0-9]+)\.hdf5\Z")


def _normalized_text(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value or value.strip() != value or any(char in value for char in "\r\n\0"):
        raise ValueError(f"{label} must be a non-empty normalized single-line string")
    return value


@dataclass(frozen=True)
class RMBenchConversionConfig:
    source_root: Path
    output_root: Path
    source_revision: str
    data_revision: str
    rmbench_code_revision: str
    source_dataset: str = "TianxingChen/RMBench"
    dataset_id: str = "rmbench_demo_clean_v1"
    fps: int = DEFAULT_FPS
    dev_per_task: int = DEFAULT_DEV_PER_TASK
    split_seed: int = DEFAULT_SPLIT_SEED
    workers: int = 4
    tasks: tuple[str, ...] = OFFICIAL_RMBENCH_TASKS
    episodes_per_task: int = OFFICIAL_EPISODES_PER_TASK
    expected_image_height: int = DEFAULT_IMAGE_HEIGHT
    expected_image_width: int = DEFAULT_IMAGE_WIDTH
    strict_official_contract: bool = True
    data_profile: str = "official50-dev45"
    progress: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(
            self, "source_revision", _normalized_text(self.source_revision, label="source_revision")
        )
        object.__setattr__(
            self, "data_revision", _normalized_text(self.data_revision, label="data_revision")
        )
        object.__setattr__(
            self,
            "rmbench_code_revision",
            _normalized_text(self.rmbench_code_revision, label="rmbench_code_revision"),
        )
        object.__setattr__(
            self,
            "source_dataset",
            _normalized_text(self.source_dataset, label="source_dataset"),
        )
        object.__setattr__(self, "dataset_id", _normalized_text(self.dataset_id, label="dataset_id"))
        object.__setattr__(
            self,
            "data_profile",
            _normalized_text(self.data_profile, label="data_profile"),
        )
        if "/" in self.dataset_id or "\\" in self.dataset_id:
            raise ValueError("dataset_id must not contain path separators")
        if isinstance(self.fps, bool) or self.fps <= 0:
            raise ValueError("fps must be a positive integer")
        if isinstance(self.workers, bool) or self.workers <= 0:
            raise ValueError("workers must be a positive integer")
        if isinstance(self.episodes_per_task, bool) or self.episodes_per_task < 2:
            raise ValueError("episodes_per_task must be at least two")
        if self.dev_per_task < 1 or self.dev_per_task >= self.episodes_per_task:
            raise ValueError("dev_per_task must leave non-empty train and dev splits")
        if self.expected_image_height <= 0 or self.expected_image_width <= 0:
            raise ValueError("expected image dimensions must be positive")
        if not self.tasks or len(self.tasks) != len(set(self.tasks)):
            raise ValueError("tasks must be a non-empty sequence without duplicates")
        if self.strict_official_contract:
            if self.source_dataset != "TianxingChen/RMBench":
                raise ValueError(
                    "official conversion requires source_dataset='TianxingChen/RMBench'"
                )
            if self.tasks != OFFICIAL_RMBENCH_TASKS:
                raise ValueError("production conversion requires the exact nine-task allow-list")
            if self.episodes_per_task != OFFICIAL_EPISODES_PER_TASK:
                raise ValueError("production conversion requires exactly 50 demos per task")
            if (self.expected_image_height, self.expected_image_width) != (
                DEFAULT_IMAGE_HEIGHT,
                DEFAULT_IMAGE_WIDTH,
            ):
                raise ValueError("production conversion requires official 240x320 cameras")


@dataclass(frozen=True)
class _EpisodePlan:
    source: SourceEpisodeSpec
    split: str
    output_episode_index: int
    global_start_index: int
    task_index: int
    source_hdf5_relpath: str
    source_instruction_relpath: str


@dataclass(frozen=True)
class _EpisodeResult:
    output_episode_index: int
    split: str
    episode_row: Mapping[str, Any]
    episode_stats: Mapping[str, Any]
    manifest_row: Mapping[str, Any]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _atomic_publish_directory(staging: Path, output: Path) -> None:
    """Atomically rename ``staging`` without ever replacing ``output``.

    Windows ``MoveFile`` already refuses an existing destination.  Linux
    requires ``renameat2(RENAME_NOREPLACE)``; a check followed by plain
    ``rename`` would contain a clobber race and is deliberately not used.
    """

    if os.name == "nt":
        os.rename(staging, output)
        return
    if sys.platform.startswith("linux"):
        import ctypes
        import errno

        at_fdcwd = -100
        rename_noreplace = 1
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:  # pragma: no cover - old/non-glibc Linux
            raise RuntimeError(
                "atomic no-clobber publication requires Linux renameat2"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        status = renameat2(
            at_fdcwd,
            os.fsencode(staging),
            at_fdcwd,
            os.fsencode(output),
            rename_noreplace,
        )
        if status == 0:
            return
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, os.strerror(error), str(output))
        raise OSError(error, os.strerror(error), str(output))
    raise RuntimeError(
        "this platform lacks a verified atomic no-clobber directory publish primitive"
    )


def _resolve_source_data_root(config: RMBenchConversionConfig) -> Path:
    root = config.source_root.resolve(strict=True)
    direct = root / config.tasks[0] / OFFICIAL_TASK_CONFIG
    nested = root / "data" / config.tasks[0] / OFFICIAL_TASK_CONFIG
    matches = [candidate.parent.parent for candidate in (direct, nested) if candidate.is_dir()]
    unique = tuple(dict.fromkeys(item.resolve() for item in matches))
    if len(unique) != 1:
        raise FileNotFoundError(
            "source_root must be either the official RMBench data directory or "
            "the repository root containing data/"
        )
    return unique[0]


def _discover_specs(
    config: RMBenchConversionConfig,
    data_root: Path,
) -> tuple[SourceEpisodeSpec, ...]:
    specs: list[SourceEpisodeSpec] = []
    resolved_root = data_root.resolve(strict=True)
    expected_indices = set(range(config.episodes_per_task))
    for task in config.tasks:
        if task not in OFFICIAL_RMBENCH_TASKS:
            raise ValueError(f"task {task!r} is outside the official RMBench allow-list")
        task_root = data_root / task / OFFICIAL_TASK_CONFIG
        data_dir = task_root / "data"
        instruction_dir = task_root / "instructions"
        if not data_dir.is_dir() or not instruction_dir.is_dir():
            raise FileNotFoundError(
                f"missing official demo_clean data/instructions for task {task!r}"
            )
        found: dict[int, Path] = {}
        for path in data_dir.iterdir():
            if not path.is_file():
                continue
            match = _EPISODE_FILE_RE.fullmatch(path.name)
            if match:
                found[int(match.group("index"))] = path
        if set(found) != expected_indices:
            missing = sorted(expected_indices - set(found))
            extra = sorted(set(found) - expected_indices)
            raise RMBenchSourceError(
                f"task {task!r} must contain exactly episode0.."
                f"episode{config.episodes_per_task - 1}; missing={missing}, extra={extra}"
            )
        for source_index in range(config.episodes_per_task):
            hdf5_path = found[source_index].resolve(strict=True)
            instruction_path = (
                instruction_dir / f"episode{source_index}.json"
            ).resolve(strict=True)
            if not _is_relative_to(hdf5_path, resolved_root) or not _is_relative_to(
                instruction_path, resolved_root
            ):
                raise RMBenchSourceError("source symlink escapes the selected RMBench data root")
            specs.append(
                inspect_source_episode(
                    task_name=task,
                    source_episode_index=source_index,
                    hdf5_path=hdf5_path,
                    instruction_path=instruction_path,
                )
            )
    return tuple(specs)


def _ordered_plans(
    config: RMBenchConversionConfig,
    data_root: Path,
    specs: Sequence[SourceEpisodeSpec],
) -> tuple[_EpisodePlan, ...]:
    assignments = deterministic_task_split(
        {
            task: range(config.episodes_per_task)
            for task in config.tasks
        },
        dev_per_task=config.dev_per_task,
        seed=config.split_seed,
    )
    spec_by_id = {(item.task_name, item.source_episode_index): item for item in specs}
    instructions = sorted({item.instruction for item in specs})
    task_indices = {instruction: index for index, instruction in enumerate(instructions)}

    ordered_ids: list[tuple[str, int]] = []
    for split in ("train", "dev"):
        for task in config.tasks:
            for source_index in range(config.episodes_per_task):
                identity = (task, source_index)
                if assignments[identity] == split:
                    ordered_ids.append(identity)

    output: list[_EpisodePlan] = []
    global_index = 0
    for output_index, identity in enumerate(ordered_ids):
        source = spec_by_id[identity]
        output.append(
            _EpisodePlan(
                source=source,
                split=assignments[identity],
                output_episode_index=output_index,
                global_start_index=global_index,
                task_index=task_indices[source.instruction],
                source_hdf5_relpath=source.hdf5_path.relative_to(data_root).as_posix(),
                source_instruction_relpath=source.instruction_path.relative_to(
                    data_root
                ).as_posix(),
            )
        )
        global_index += source.transition_count
    return tuple(output)


def _parquet_path(root: Path, episode_index: int) -> Path:
    return root / DATA_PATH_TEMPLATE.format(
        episode_chunk=episode_index // LEROBOT_CHUNK_SIZE,
        episode_index=episode_index,
    )


def _video_path(root: Path, episode_index: int, camera_key: str) -> Path:
    return root / VIDEO_PATH_TEMPLATE.format(
        episode_chunk=episode_index // LEROBOT_CHUNK_SIZE,
        episode_index=episode_index,
        video_key=f"observation.images.{camera_key}",
    )


def _write_parquet(
    path: Path,
    *,
    plan: _EpisodePlan,
    qpos: np.ndarray,
    fps: int,
) -> dict[str, np.ndarray]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pyarrow is required for RMBench conversion") from exc

    rows = plan.source.transition_count
    states = np.ascontiguousarray(qpos[:-1], dtype=np.float32)
    actions = np.ascontiguousarray(qpos[1:], dtype=np.float32)
    frame_index = np.arange(rows, dtype=np.int64)
    timestamp = np.asarray(frame_index / float(fps), dtype=np.float32)
    episode_index = np.full(rows, plan.output_episode_index, dtype=np.int64)
    global_index = np.arange(
        plan.global_start_index,
        plan.global_start_index + rows,
        dtype=np.int64,
    )
    task_index = np.full(rows, plan.task_index, dtype=np.int64)
    next_done = np.zeros(rows, dtype=np.bool_)
    next_done[-1] = True
    next_reward = np.zeros(rows, dtype=np.float32)
    next_reward[-1] = np.float32(1.0)

    schema = pa.schema(
        [
            pa.field("timestamp", pa.float32(), nullable=False),
            pa.field("frame_index", pa.int64(), nullable=False),
            pa.field("episode_index", pa.int64(), nullable=False),
            pa.field("index", pa.int64(), nullable=False),
            pa.field("task_index", pa.int64(), nullable=False),
            pa.field(
                "observation.state",
                pa.list_(pa.float32(), ACTION_DIM),
                nullable=False,
            ),
            pa.field("action", pa.list_(pa.float32(), ACTION_DIM), nullable=False),
            pa.field("next.done", pa.bool_(), nullable=False),
            pa.field("next.reward", pa.float32(), nullable=False),
        ],
        metadata={
            b"warm.action_semantics": b"qpos[t+1]",
        },
    )
    arrays = [
        pa.array(timestamp, type=pa.float32()),
        pa.array(frame_index, type=pa.int64()),
        pa.array(episode_index, type=pa.int64()),
        pa.array(global_index, type=pa.int64()),
        pa.array(task_index, type=pa.int64()),
        pa.array(states.tolist(), type=pa.list_(pa.float32(), ACTION_DIM)),
        pa.array(actions.tolist(), type=pa.list_(pa.float32(), ACTION_DIM)),
        pa.array(next_done, type=pa.bool_()),
        pa.array(next_reward, type=pa.float32()),
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        version="2.6",
        data_page_version="2.0",
        use_dictionary=False,
        write_statistics=True,
        row_group_size=rows,
    )
    return {
        "timestamp": timestamp,
        "frame_index": frame_index,
        "episode_index": episode_index,
        "index": global_index,
        "task_index": task_index,
        "observation.state": states,
        "action": actions,
        "next.done": next_done,
        "next.reward": next_reward,
    }


def _write_camera_video(
    path: Path,
    *,
    encoded_cells: Any,
    camera_key: str,
    source_label: str,
    observation_count: int,
    fps: int,
    height: int,
    width: int,
) -> str:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("PyAV with libx264 support is required for RMBench conversion") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    terminal_jpeg_sha256: str | None = None
    try:
        container = av.open(
            str(path),
            mode="w",
            format="mp4",
            options={"movflags": "+faststart", "brand": "mp42"},
        )
        container.metadata.clear()
        stream = container.add_stream(
            "libx264",
            rate=Fraction(fps, 1),
            options={
                "crf": "18",
                "preset": "medium",
                "threads": "1",
                "x264-params": (
                    "threads=1:lookahead_threads=1:sync-lookahead=0:"
                    "scenecut=0:open-gop=0"
                ),
            },
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.gop_size = max(1, fps)
        stream.codec_context.max_b_frames = 0
        stream.codec_context.time_base = Fraction(1, fps)
        stream.time_base = Fraction(1, fps)
        stream.metadata.clear()
        for source_frame in range(observation_count):
            image, jpeg = decode_official_jpeg(
                encoded_cells[source_frame],
                label=f"{source_label}:{camera_key}[{source_frame}]",
                expected_height=height,
                expected_width=width,
            )
            if source_frame == observation_count - 1:
                terminal_jpeg_sha256 = sha256(jpeg).hexdigest()
                continue
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = source_frame
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()
    except Exception:
        path.unlink(missing_ok=True)
        raise

    decoded_count = 0
    with av.open(str(path), mode="r") as check:
        for frame in check.decode(video=0):
            if frame.width != width or frame.height != height:
                raise RuntimeError(f"encoded video geometry changed for {path}")
            decoded_count += 1
    if decoded_count != observation_count - 1:
        raise RuntimeError(
            f"encoded video {path} has {decoded_count} frames; expected "
            f"{observation_count - 1}"
        )
    if terminal_jpeg_sha256 is None:  # pragma: no cover - defensive
        raise RuntimeError("terminal JPEG was not validated")
    return terminal_jpeg_sha256


def _feature_stats(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    if value.ndim == 1:
        minimum = np.min(value, keepdims=True)
        maximum = np.max(value, keepdims=True)
        mean = np.mean(value, keepdims=True)
        std = np.std(value, keepdims=True)
    else:
        minimum = np.min(value, axis=0)
        maximum = np.max(value, axis=0)
        mean = np.mean(value, axis=0)
        std = np.std(value, axis=0)
    return {
        "min": np.asarray(minimum).tolist(),
        "max": np.asarray(maximum).tolist(),
        "mean": np.asarray(mean).tolist(),
        "std": np.asarray(std).tolist(),
        "count": [int(value.shape[0])],
    }


def _convert_episode(
    config: RMBenchConversionConfig,
    plan: _EpisodePlan,
    staging_root: Path,
) -> _EpisodeResult:
    if file_sha256(plan.source.instruction_path) != plan.source.instruction_sha256:
        raise RMBenchSourceError(
            f"instruction JSON changed after preflight: {plan.source.instruction_path}"
        )
    hdf5_before = file_sha256(plan.source.hdf5_path)
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("h5py is required for RMBench conversion") from exc

    parquet = _parquet_path(staging_root, plan.output_episode_index)
    terminal_jpegs: dict[str, str] = {}
    with h5py.File(plan.source.hdf5_path, "r") as episode:
        qpos = read_qpos(episode, source_label=str(plan.source.hdf5_path))
        if qpos.shape[0] != plan.source.observation_count:
            raise RMBenchSourceError("source observation count changed after preflight")
        table_arrays = _write_parquet(
            parquet,
            plan=plan,
            qpos=qpos,
            fps=config.fps,
        )
        for camera_key in OFFICIAL_CAMERA_KEYS:
            video = _video_path(staging_root, plan.output_episode_index, camera_key)
            terminal_jpegs[camera_key] = _write_camera_video(
                video,
                encoded_cells=episode[SOURCE_CAMERA_PATHS[camera_key]],
                camera_key=camera_key,
                source_label=plan.source_hdf5_relpath,
                observation_count=plan.source.observation_count,
                fps=config.fps,
                height=config.expected_image_height,
                width=config.expected_image_width,
            )
    hdf5_after = file_sha256(plan.source.hdf5_path)
    if hdf5_after != hdf5_before:
        raise RMBenchSourceError(
            f"source HDF5 changed during conversion: {plan.source.hdf5_path}"
        )
    if file_sha256(plan.source.instruction_path) != plan.source.instruction_sha256:
        raise RMBenchSourceError(
            f"instruction JSON changed during conversion: {plan.source.instruction_path}"
        )

    output_videos = {
        camera_key: {
            "relpath": _video_path(
                Path("."), plan.output_episode_index, camera_key
            ).as_posix(),
            "sha256": file_sha256(
                _video_path(staging_root, plan.output_episode_index, camera_key)
            ),
        }
        for camera_key in OFFICIAL_CAMERA_KEYS
    }
    episode_stats = {
        "observation.state": _feature_stats(table_arrays["observation.state"]),
        "action": _feature_stats(table_arrays["action"]),
    }
    episode_row = {
        "episode_index": plan.output_episode_index,
        "tasks": [plan.source.instruction],
        "warm_task_identity": plan.source.task_name,
        "length": plan.source.transition_count,
        "raw_file_name": plan.source_hdf5_relpath,
    }
    terminal_state_sha256 = sha256(
        np.asarray(qpos[-1], dtype="<f4").tobytes()
    ).hexdigest()
    terminal_observation_digest = sha256(
        canonical_json_bytes(
            {
                "state_sha256": terminal_state_sha256,
                "jpeg_sha256": terminal_jpegs,
            }
        )
    ).hexdigest()
    manifest_row = {
        "output_episode_index": plan.output_episode_index,
        "source_task": plan.source.task_name,
        "retrieval_task_identity": plan.source.task_name,
        "source_episode_index": plan.source.source_episode_index,
        "split": plan.split,
        "instruction": plan.source.instruction,
        # Keep the complete official wording inventory bound to this episode.
        # Runtime training is publication-fair by default and samples only
        # ``seen``.  ``unseen`` is retained for auditability and for an
        # explicitly labelled transductive ablation; it is never silently
        # admitted by the dataset loader.
        "instruction_variants": {
            key: list(value)
            for key, value in sorted(plan.source.instruction_variants.items())
        },
        "instruction_variant_counts": {
            key: len(value)
            for key, value in sorted(plan.source.instruction_variants.items())
        },
        "source_hdf5_relpath": plan.source_hdf5_relpath,
        "source_hdf5_sha256": hdf5_before,
        "source_instruction_relpath": plan.source_instruction_relpath,
        "source_instruction_sha256": plan.source.instruction_sha256,
        "source_observation_count": plan.source.observation_count,
        "factual_transition_count": plan.source.transition_count,
        "terminal_observation_count": 1,
        "terminal_visual_excluded_from_training": True,
        "terminal_observation_published_as_training_row": False,
        "terminal_observation_digest": terminal_observation_digest,
        "terminal_state_sha256": terminal_state_sha256,
        "terminal_jpeg_sha256": terminal_jpegs,
        "output_parquet": {
            "relpath": _parquet_path(
                Path("."), plan.output_episode_index
            ).as_posix(),
            "sha256": file_sha256(parquet),
        },
        "output_videos": output_videos,
    }
    return _EpisodeResult(
        output_episode_index=plan.output_episode_index,
        split=plan.split,
        episode_row=episode_row,
        episode_stats=episode_stats,
        manifest_row=manifest_row,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())


def _info_document(
    config: RMBenchConversionConfig,
    results: Sequence[_EpisodeResult],
) -> dict[str, Any]:
    train_count = sum(item.split == "train" for item in results)
    total_frames = sum(int(item.episode_row["length"]) for item in results)
    instructions = sorted(
        {str(item.episode_row["tasks"][0]) for item in results}
    )
    vector_feature = {
        "dtype": "float32",
        "shape": [ACTION_DIM],
        "names": [list(MOTOR_NAMES)],
    }
    scalar_i64 = {"dtype": "int64", "shape": [1], "names": None}
    features: dict[str, Any] = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": scalar_i64,
        "episode_index": scalar_i64,
        "index": scalar_i64,
        "task_index": scalar_i64,
        "observation.state": vector_feature,
        "action": vector_feature,
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
        "next.reward": {"dtype": "float32", "shape": [1], "names": None},
    }
    for camera_key in OFFICIAL_CAMERA_KEYS:
        features[f"observation.images.{camera_key}"] = {
            "dtype": "video",
            "shape": [
                3,
                config.expected_image_height,
                config.expected_image_width,
            ],
            "names": ["channels", "height", "width"],
        }
    return {
        "codebase_version": LEROBOT_CODEBASE_VERSION,
        "robot_type": "aloha",
        "total_episodes": len(results),
        "total_frames": total_frames,
        "total_tasks": len(instructions),
        "total_videos": len(results) * len(OFFICIAL_CAMERA_KEYS),
        "total_chunks": (len(results) + LEROBOT_CHUNK_SIZE - 1)
        // LEROBOT_CHUNK_SIZE,
        "chunks_size": LEROBOT_CHUNK_SIZE,
        "fps": config.fps,
        "splits": {
            "train": f"0:{train_count}",
            "dev": f"{train_count}:{len(results)}",
        },
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
        "features": features,
    }


def _artifact_rows(root: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relpath = path.relative_to(root).as_posix()
        if relpath in exclude:
            continue
        rows.append(
            {
                "relpath": relpath,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def _publish_metadata(
    config: RMBenchConversionConfig,
    staging_root: Path,
    data_root: Path,
    results: Sequence[_EpisodeResult],
) -> dict[str, Any]:
    ordered = tuple(sorted(results, key=lambda item: item.output_episode_index))
    expected = list(range(len(ordered)))
    if [item.output_episode_index for item in ordered] != expected:
        raise RuntimeError("worker results are not a complete contiguous episode set")
    info = _info_document(config, ordered)
    meta = staging_root / "meta"
    _write_json(meta / "info.json", info)
    _write_jsonl(meta / "episodes.jsonl", (item.episode_row for item in ordered))
    _write_jsonl(
        meta / "warm_instruction_variants.jsonl",
        (
            {
                "episode_index": item.output_episode_index,
                "primary": str(item.manifest_row["instruction"]),
                "seen": list(item.manifest_row["instruction_variants"]["seen"]),
                "unseen": list(item.manifest_row["instruction_variants"]["unseen"]),
            }
            for item in ordered
        ),
    )
    instructions = sorted(
        {str(item.episode_row["tasks"][0]) for item in ordered}
    )
    _write_jsonl(
        meta / "tasks.jsonl",
        (
            {"task_index": index, "task": instruction}
            for index, instruction in enumerate(instructions)
        ),
    )
    _write_jsonl(
        meta / "episodes_stats.jsonl",
        (
            {
                "episode_index": item.output_episode_index,
                "stats": item.episode_stats,
            }
            for item in ordered
        ),
    )

    scanned = scan_lerobot_datasets(
        [staging_root],
        dataset_ids=[config.dataset_id],
        require_episode_data=True,
    )
    split_by_episode = {
        item.output_episode_index: item.split for item in ordered
    }
    catalog = EpisodeCatalog(
        scanned.datasets,
        tuple(
            replace(record, split=split_by_episode[record.episode_index])
            for record in scanned.episodes
        ),
    )
    catalog_path = meta / CATALOG_FILENAME
    catalog.save(catalog_path)

    episode_manifest = [dict(item.manifest_row) for item in ordered]
    source_hash_rows = [
        {
            "hdf5_relpath": item["source_hdf5_relpath"],
            "hdf5_sha256": item["source_hdf5_sha256"],
            "instruction_relpath": item["source_instruction_relpath"],
            "instruction_sha256": item["source_instruction_sha256"],
        }
        for item in episode_manifest
    ]
    artifacts = _artifact_rows(
        staging_root,
        exclude={f"meta/{MANIFEST_FILENAME}"},
    )
    source_tree_sha256 = sha256(canonical_json_bytes(source_hash_rows)).hexdigest()
    artifact_tree_sha256 = sha256(canonical_json_bytes(artifacts)).hexdigest()
    manifest: dict[str, Any] = {
        "schema": CONVERSION_SCHEMA,
        "schema_version": CONVERSION_SCHEMA_VERSION,
        "source": {
            "dataset": config.source_dataset,
            "revision": config.source_revision,
            "rmbench_code_revision": config.rmbench_code_revision,
            "task_config": OFFICIAL_TASK_CONFIG,
            "source_tree_sha256": source_tree_sha256,
        },
        "output": {
            "dataset_id": config.dataset_id,
            "data_revision": config.data_revision,
            "lerobot_codebase_version": LEROBOT_CODEBASE_VERSION,
            "catalog_relpath": f"meta/{CATALOG_FILENAME}",
            "catalog_sha256": file_sha256(catalog_path),
            "artifact_tree_sha256": artifact_tree_sha256,
            "artifacts": artifacts,
        },
        "protocol": {
            "data_profile": config.data_profile,
            "official_task_allow_list": list(config.tasks),
            "episodes_per_task": config.episodes_per_task,
            "fps": config.fps,
            "camera_order": list(OFFICIAL_CAMERA_KEYS),
            "action_dim": ACTION_DIM,
            "language_prompt_contract": (
                "parquet.task_index -> meta/tasks.jsonl primary natural seen "
                "instruction; meta/warm_instruction_variants.jsonl binds all "
                "official seen/unseen variants; training defaults to deterministic "
                "seen-only sampling and requires an explicit transductive flag to "
                "admit unseen wording"
            ),
            "retrieval_task_contract": (
                "meta/episodes.jsonl.warm_task_identity -> exact official task name"
            ),
            "source_observation_semantics": "qpos[0:N]",
            "published_row_semantics": (
                "N-1 factual transitions: state[t]=qpos[t], action[t]=qpos[t+1]"
            ),
            "terminal_observation_policy": (
                "strictly validate and source-hash; do not publish a synthetic-action row"
            ),
            "terminal_visual_excluded_from_training": True,
            "terminal_observation_count": len(ordered),
            "terminal_observation_digest_contract": (
                "per-episode SHA-256 over canonical JSON containing terminal float32 "
                "state SHA-256 and ordered camera JPEG SHA-256 values"
            ),
            "split": {
                "unit": "source_episode",
                "seed": config.split_seed,
                "dev_per_task": config.dev_per_task,
                "train_episodes": sum(item.split == "train" for item in ordered),
                "dev_episodes": sum(item.split == "dev" for item in ordered),
                "train_then_dev_contiguous_output_order": True,
            },
        },
        "episodes": episode_manifest,
    }
    manifest["manifest_sha256"] = sha256(canonical_json_bytes(manifest)).hexdigest()
    _write_json(meta / MANIFEST_FILENAME, manifest)
    return manifest


def convert_rmbench_dataset(config: RMBenchConversionConfig) -> dict[str, Any]:
    """Convert and atomically publish one immutable LeRobot dataset tree."""

    output = config.output_root.expanduser().resolve(strict=False)
    if os.path.lexists(output):
        raise FileExistsError(f"output_root already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    data_root = _resolve_source_data_root(config)
    if _is_relative_to(output, data_root) or _is_relative_to(data_root, output):
        raise ValueError("source and output roots must not overlap")

    specs = _discover_specs(config, data_root)
    plans = _ordered_plans(config, data_root, specs)
    staging = output.parent / f".{output.name}.tmp-{uuid4().hex}"
    if os.path.lexists(staging):  # pragma: no cover - UUID defensive check
        raise FileExistsError(f"temporary path already exists: {staging}")
    staging.mkdir()
    try:
        results: list[_EpisodeResult] = []
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            futures: dict[Future[_EpisodeResult], _EpisodePlan] = {
                executor.submit(_convert_episode, config, plan, staging): plan
                for plan in plans
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if config.progress:
                    plan = futures[future]
                    print(
                        f"[{completed}/{len(plans)}] {plan.source.task_name}/"
                        f"episode{plan.source.source_episode_index} -> "
                        f"episode_{plan.output_episode_index:06d}"
                    )
        manifest = _publish_metadata(config, staging, data_root, results)
        if os.path.lexists(output):
            raise FileExistsError(f"output_root appeared during conversion: {output}")
        _atomic_publish_directory(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "RMBenchConversionConfig",
    "convert_rmbench_dataset",
]
