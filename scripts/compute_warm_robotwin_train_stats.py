#!/usr/bin/env python3
"""Compute immutable train-only 14D z-score statistics for RMBench/RoboTwin."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from fastwam.datasets.lerobot.audit import load_audit_report
from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog
from fastwam.memory.processor_contract import extract_m1_robotwin_processor_recipe
from fastwam.memory.robotwin_train_stats import (
    FastWAMRobotwinTrainStats,
    RobotwinTrainStatsManifest,
    encode_robotwin_manifest,
    encode_robotwin_stats,
    load_robotwin_train_stats_artifact,
)
from fastwam.memory.train_stats import (
    TrainEpisodeSource,
    compute_train_episode_set_sha256,
)
from fastwam.utils.artifact_claim import artifact_claim


_REVISION = re.compile(r"[0-9a-f]{40}\Z")
EXPECTED_CAMERAS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


class RobotwinStatsComputationError(ValueError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, action="append", type=Path)
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _sha(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_identity() -> tuple[str, bool]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RobotwinStatsComputationError("cannot inspect WARM Git identity") from exc
    return commit, dirty


def _load_config(path: Path) -> tuple[bytes, Mapping[str, Any]]:
    raw = path.read_bytes()
    try:
        from omegaconf import OmegaConf

        value = OmegaConf.to_container(
            OmegaConf.create({"data": OmegaConf.load(path)}), resolve=True
        )
    except Exception as exc:  # pragma: no cover - server dependency
        raise RobotwinStatsComputationError("cannot resolve data config") from exc
    if not isinstance(value, Mapping):
        raise RobotwinStatsComputationError("data config must resolve to a mapping")
    extract_m1_robotwin_processor_recipe(value)
    return raw, value


def _table(payload: bytes) -> Mapping[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - server dependency
        raise RuntimeError("pyarrow is required on the data server") from exc
    table = pq.read_table(
        pa.BufferReader(payload),
        columns=[
            "action",
            "observation.state",
            "episode_index",
            "frame_index",
            "timestamp",
            "task_index",
        ],
    )
    return {name: table[name] for name in table.column_names}


def _array(value: Any, *, rows: int, field: str, vector: bool) -> np.ndarray:
    if hasattr(value, "combine_chunks"):
        value = value.combine_chunks()
    if hasattr(value, "to_pylist"):
        value = value.to_pylist()
    result = np.asarray(value)
    if vector and result.shape != (rows, 14):
        raise RobotwinStatsComputationError(
            f"{field} must have native shape {(rows, 14)}, got {result.shape}"
        )
    if not vector and result.shape not in {(rows,), (rows, 1)}:
        raise RobotwinStatsComputationError(f"{field} must contain {rows} scalars")
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise RobotwinStatsComputationError(f"{field} must be finite numeric data")
    return np.asarray(result, dtype=np.float64)


def _roots(catalog: EpisodeCatalog, values: Sequence[Path]) -> tuple[Path, ...]:
    descriptors = tuple(sorted(catalog.datasets, key=lambda item: item.dataset_index))
    if len(values) != len(descriptors):
        raise RobotwinStatsComputationError("dataset-root count differs from catalog")
    result: list[Path] = []
    for descriptor, raw in zip(descriptors, values, strict=True):
        root = raw.expanduser().resolve()
        if _sha(root / "meta" / "info.json") != descriptor.info_sha256 or _sha(
            root / "meta" / "episodes.jsonl"
        ) != descriptor.episodes_sha256:
            raise RobotwinStatsComputationError("dataset metadata changed after catalog")
        result.append(root)
    return tuple(result)


def _moments(sum_: np.ndarray, square: np.ndarray, count: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mean = sum_ / count
    variance = np.maximum(square / count - mean * mean, 0.0)
    return tuple(mean.tolist()), tuple(np.sqrt(variance).tolist())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _REVISION.fullmatch(args.dataset_revision) is None:
        raise RobotwinStatsComputationError(
            "dataset-revision must be a lowercase 40-character commit"
        )
    catalog = EpisodeCatalog.load(args.catalog.expanduser().resolve())
    audit = load_audit_report(args.audit_report.expanduser().resolve())
    if (
        not audit.episode_tables_hashed
        or audit.catalog_sha256 != catalog.content_sha256
        or audit.cross_split_duplicate_count
        or audit.audited_camera_keys != EXPECTED_CAMERAS
    ):
        raise RobotwinStatsComputationError(
            "a complete duplicate-free three-camera audit is required"
        )
    roots = _roots(catalog, args.dataset_root)
    config_path = args.data_config.expanduser().resolve()
    config_raw, _ = _load_config(config_path)
    config_sha = sha256(config_raw).hexdigest()
    records = tuple(
        sorted(
            (item for item in catalog.episodes if item.split == "train"),
            key=lambda item: (item.dataset_index, item.episode_index),
        )
    )
    if not records:
        raise RobotwinStatsComputationError("catalog has no train episodes")

    action_sum = np.zeros(14, dtype=np.float64)
    action_square = np.zeros(14, dtype=np.float64)
    state_sum = np.zeros(14, dtype=np.float64)
    state_square = np.zeros(14, dtype=np.float64)
    total = 0
    sources: list[TrainEpisodeSource] = []
    for record in records:
        proof = audit.proof_index[
            (record.dataset_id, record.dataset_index, record.episode_index)
        ]
        path = (roots[record.dataset_index] / record.data_relpath).resolve()
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != proof.table_sha256:
            raise RobotwinStatsComputationError(f"episode table changed: {path}")
        data = _table(raw)
        action = _array(data["action"], rows=record.length, field="action", vector=True)
        state = _array(
            data["observation.state"],
            rows=record.length,
            field="observation.state",
            vector=True,
        )
        episode = _array(
            data["episode_index"], rows=record.length, field="episode_index", vector=False
        ).reshape(-1)
        frames = _array(
            data["frame_index"], rows=record.length, field="frame_index", vector=False
        ).reshape(-1)
        timestamps = _array(
            data["timestamp"], rows=record.length, field="timestamp", vector=False
        ).reshape(-1)
        if not np.all(episode == record.episode_index):
            raise RobotwinStatsComputationError("episode_index column mismatch")
        if not np.array_equal(frames, np.arange(record.length)):
            raise RobotwinStatsComputationError("frame_index must be contiguous")
        if not np.allclose(timestamps, frames / record.fps, rtol=0.0, atol=1e-4):
            raise RobotwinStatsComputationError("timestamps disagree with FPS")
        action_sum += action.sum(axis=0)
        action_square += np.square(action).sum(axis=0)
        state_sum += state.sum(axis=0)
        state_square += np.square(state).sum(axis=0)
        total += record.length
        sources.append(
            TrainEpisodeSource(
                record.dataset_id,
                record.dataset_index,
                record.episode_index,
                proof.source_episode_sha256,
            )
        )

    action_mean, action_std = _moments(action_sum, action_square, total)
    state_mean, state_std = _moments(state_sum, state_square, total)
    stats = FastWAMRobotwinTrainStats(
        action_mean,
        action_std,
        state_mean,
        state_std,
        len(records),
        total,
    )
    stats_raw = encode_robotwin_stats(stats)
    commit, dirty = _git_identity()
    if dirty:
        raise RobotwinStatsComputationError(
            "official stats require a committed, clean WARM checkout"
        )
    manifest = RobotwinTrainStatsManifest(
        stats_file_sha256=sha256(stats_raw).hexdigest(),
        catalog_sha256=catalog.content_sha256,
        audit_report_sha256=audit.report_sha256,
        data_config_sha256=config_sha,
        train_episode_set_sha256=compute_train_episode_set_sha256(sources),
        train_episode_count=len(records),
        train_frame_count=total,
        git_commit=commit,
        dataset_revision=args.dataset_revision,
    )
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for root in roots:
        if output == root or root in output.parents:
            raise RobotwinStatsComputationError("output must stay outside dataset roots")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.parent / f".{output.name}.warm-artifact.lock"
    with artifact_claim(lock, purpose=f"publish RoboTwin train stats: {output}"):
        staging = output.parent / f".{output.name}.{uuid4().hex}.staging"
        staging.mkdir()
        try:
            (staging / "dataset_stats.json").write_bytes(stats_raw)
            (staging / "train_stats_manifest.json").write_bytes(
                encode_robotwin_manifest(manifest)
            )
            load_robotwin_train_stats_artifact(
                staging,
                expected_catalog_sha256=catalog.content_sha256,
                expected_audit_report_sha256=audit.report_sha256,
                expected_data_config_sha256=config_sha,
                expected_train_episodes=sources,
            )
            os.rename(staging, output)
        except BaseException:
            # Preserve staging for forensic inspection; never publish partial output.
            raise
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest_sha256": manifest.content_sha256,
                "train_episode_count": len(records),
                "train_frame_count": total,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
