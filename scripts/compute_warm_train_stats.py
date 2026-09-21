#!/usr/bin/env python3
"""Compute immutable train-only FastWAM LIBERO normalization statistics.

Only audited parquet byte snapshots from catalog ``train`` episodes are read.
No video, GPU model, training window, padding, or terminal-action removal is
involved: global min/max is computed over every one of the N raw action and
state rows, matching FastWAM's non-stepwise min/max recipe.

Torch, PyArrow, and Hydra/OmegaConf are deliberately absent at module import.
The default adapters import PyArrow and OmegaConf only inside the execution
path; tests may inject a table reader and config mapping loader.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any
from uuid import uuid4

import numpy as np

from fastwam.datasets.lerobot.audit import (
    EpisodeAuditProof,
    LerobotAuditReport,
    load_audit_report,
)
from fastwam.datasets.lerobot.episode_catalog import (
    EpisodeCatalog,
    EpisodeRecord,
)
from fastwam.memory.train_stats import (
    TRAIN_STATS_FILENAME,
    TRAIN_STATS_MANIFEST_FILENAME,
    FastWAMLiberoTrainStats,
    TrainEpisodeSource,
    TrainStatsContractError,
    TrainStatsManifest,
    compute_train_episode_set_sha256,
    encode_train_stats_json,
    encode_train_stats_manifest_json,
    load_train_stats_artifact,
)
from fastwam.utils.artifact_claim import artifact_claim


ACTION_KEY = "action"
STATE_KEY = "observation.state"
EXPECTED_ACTION_DIM = 7
EXPECTED_STATE_DIM = 8
DEFAULT_TIMESTAMP_TOLERANCE_S = 1e-4
EXPECTED_AUDITED_CAMERAS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
_REQUIRED_COLUMNS = (
    ACTION_KEY,
    STATE_KEY,
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
)

TableReader = Callable[[bytes], Mapping[str, Any]]
ConfigLoader = Callable[[bytes], Mapping[str, Any]]


class TrainStatsComputationError(ValueError):
    """Raised when inputs cannot define one exact train-only artifact."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute FastWAM LIBERO global min/max from all raw rows of the "
            "catalog-authorized train episodes."
        )
    )
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        required=True,
        action="append",
        type=Path,
        help="Repeat in exact catalog dataset_index order.",
    )
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help=(
            "New immutable directory containing dataset_stats.json and "
            "train_stats_manifest.json."
        ),
    )
    parser.add_argument(
        "--timestamp-tolerance-s",
        type=float,
        default=DEFAULT_TIMESTAMP_TOLERANCE_S,
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise FileNotFoundError(path) from exc
    return digest.hexdigest()


def _read_opaque_snapshot(path: Path, *, label: str) -> tuple[bytes, str]:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"cannot read {label}: {resolved}") from exc
    if not raw.strip():
        raise TrainStatsComputationError(f"{label} must not be empty")
    return raw, sha256(raw).hexdigest()


def _resolve_inside(root: Path, relative: str | Path, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise TrainStatsComputationError(f"{label} must be relative")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TrainStatsComputationError(f"{label} escapes dataset root") from exc
    return resolved


def _default_config_loader(payload: bytes) -> Mapping[str, Any]:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - server dependency path
        raise RuntimeError(
            "Reading the FastWAM YAML requires omegaconf on the server; "
            "install project dependencies or inject config_loader in tests"
        ) from exc
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise TrainStatsComputationError("data config must be valid UTF-8") from exc
    raw = OmegaConf.create(text)
    wrapped = OmegaConf.create({"data": raw})
    OmegaConf.resolve(wrapped)
    value = OmegaConf.to_container(wrapped.data, resolve=True)
    if not isinstance(value, Mapping):
        raise TrainStatsComputationError("data config must resolve to a mapping")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainStatsComputationError(f"{label} must be a mapping")
    return value


def _one_meta_dimension(
    shape_meta: Mapping[str, Any],
    *,
    kind: str,
    expected: int,
) -> int:
    value = shape_meta.get(kind)
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], Mapping)
    ):
        raise TrainStatsComputationError(
            f"data config shape_meta.{kind} must contain exactly one entry"
        )
    entry = value[0]
    if entry.get("key") != "default":
        raise TrainStatsComputationError(
            f"data config shape_meta.{kind}[0].key must be 'default'"
        )
    for field in ("raw_shape", "shape"):
        value = entry.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != expected:
            raise TrainStatsComputationError(
                f"data config shape_meta.{kind}[0].{field} must be {expected}"
            )
    return expected


def _validate_libero_config(
    path: Path,
    *,
    config_loader: ConfigLoader | None = None,
) -> tuple[str, int, int]:
    """Bind exact config bytes and validate the raw-statistics semantics."""

    resolved = path.expanduser().resolve()
    config_snapshot, config_sha256 = _read_opaque_snapshot(
        resolved, label="data config"
    )
    loader = config_loader or _default_config_loader
    config = _mapping(loader(config_snapshot), label="data config")
    train = _mapping(config.get("train"), label="data config train")
    processor = _mapping(
        train.get("processor"), label="data config train.processor"
    )
    expected_target = (
        "fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor"
    )
    if processor.get("_target_") != expected_target:
        raise TrainStatsComputationError(
            "data config must use the FastWAMProcessor LIBERO processor"
        )
    if "action_state_transforms" not in processor:
        raise TrainStatsComputationError(
            "data config must explicitly declare action_state_transforms: null"
        )
    if processor["action_state_transforms"] is not None:
        raise TrainStatsComputationError(
            "LIBERO train statistics require action_state_transforms: null"
        )
    if processor.get("use_stepwise_action_norm") is not False:
        raise TrainStatsComputationError(
            "LIBERO train statistics require use_stepwise_action_norm: false"
        )
    if processor.get("norm_default_mode") != "min/max":
        raise TrainStatsComputationError(
            "LIBERO train statistics require norm_default_mode: min/max"
        )
    if processor.get("norm_exception_mode") is not None:
        raise TrainStatsComputationError(
            "LIBERO train statistics require norm_exception_mode: null"
        )
    shape_meta = _mapping(train.get("shape_meta"), label="data config shape_meta")
    action_dim = _one_meta_dimension(
        shape_meta, kind="action", expected=EXPECTED_ACTION_DIM
    )
    state_dim = _one_meta_dimension(
        shape_meta, kind="state", expected=EXPECTED_STATE_DIM
    )
    if processor.get("action_output_dim") != action_dim:
        raise TrainStatsComputationError(
            "processor.action_output_dim disagrees with LIBERO action shape"
        )
    if processor.get("proprio_output_dim") != state_dim:
        raise TrainStatsComputationError(
            "processor.proprio_output_dim disagrees with LIBERO state shape"
        )
    return config_sha256, action_dim, state_dim


def _validate_dataset_roots(
    catalog: EpisodeCatalog,
    dataset_roots: Sequence[Path],
) -> tuple[Path, ...]:
    descriptors = tuple(sorted(catalog.datasets, key=lambda item: item.dataset_index))
    if tuple(item.dataset_index for item in descriptors) != tuple(range(len(descriptors))):
        raise TrainStatsComputationError(
            "catalog dataset indices must be contiguous from zero"
        )
    if len(dataset_roots) != len(descriptors):
        raise TrainStatsComputationError(
            "--dataset-root count must equal catalog dataset count"
        )
    roots: list[Path] = []
    for descriptor, root_value in zip(descriptors, dataset_roots, strict=True):
        root = root_value.expanduser().resolve()
        if root in roots:
            raise TrainStatsComputationError("dataset roots must be unique")
        info_path = root / "meta" / "info.json"
        episodes_path = root / "meta" / "episodes.jsonl"
        if not root.is_dir() or not info_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(
                f"dataset root lacks LeRobot metadata: {root}"
            )
        if _sha256_file(info_path) != descriptor.info_sha256:
            raise TrainStatsComputationError(
                f"dataset metadata changed after cataloging: {descriptor.dataset_id}"
            )
        if _sha256_file(episodes_path) != descriptor.episodes_sha256:
            raise TrainStatsComputationError(
                f"episode metadata changed after cataloging: {descriptor.dataset_id}"
            )
        roots.append(root)
    return tuple(roots)


@dataclass(frozen=True, slots=True)
class TrainStatsPlan:
    catalog: EpisodeCatalog
    audit: LerobotAuditReport
    roots: tuple[Path, ...]
    records: tuple[EpisodeRecord, ...]
    proofs: tuple[EpisodeAuditProof, ...]
    data_config_sha256: str
    action_dim: int
    state_dim: int
    output: Path
    timestamp_tolerance_s: float

    @property
    def episode_sources(self) -> tuple[TrainEpisodeSource, ...]:
        return tuple(
            TrainEpisodeSource(
                dataset_id=record.dataset_id,
                dataset_index=record.dataset_index,
                episode_index=record.episode_index,
                source_episode_sha256=proof.source_episode_sha256,
            )
            for record, proof in zip(self.records, self.proofs, strict=True)
        )


def prepare_plan(
    *,
    catalog_path: Path,
    audit_report_path: Path,
    dataset_roots: Sequence[Path],
    data_config_path: Path,
    output: Path,
    timestamp_tolerance_s: float = DEFAULT_TIMESTAMP_TOLERANCE_S,
    config_loader: ConfigLoader | None = None,
) -> TrainStatsPlan:
    if (
        isinstance(timestamp_tolerance_s, bool)
        or not isinstance(timestamp_tolerance_s, (int, float))
        or not math.isfinite(timestamp_tolerance_s)
        or timestamp_tolerance_s < 0
    ):
        raise TrainStatsComputationError(
            "timestamp tolerance must be finite and non-negative"
        )
    catalog = EpisodeCatalog.load(catalog_path.expanduser().resolve())
    audit = load_audit_report(audit_report_path.expanduser().resolve())
    if not audit.episode_tables_hashed:
        raise TrainStatsComputationError(
            "train statistics require a complete source-hashed audit"
        )
    if audit.catalog_sha256 != catalog.content_sha256:
        raise TrainStatsComputationError("audit does not bind the supplied catalog")
    if audit.cross_split_duplicate_count:
        raise TrainStatsComputationError(
            "audit reports raw-source duplicates across data splits"
        )
    if audit.audited_camera_keys != EXPECTED_AUDITED_CAMERAS:
        raise TrainStatsComputationError(
            "train statistics require the complete FastWAM LIBERO two-camera "
            f"source audit in order {EXPECTED_AUDITED_CAMERAS!r}"
        )
    catalog_keys = {
        (record.dataset_id, record.dataset_index, record.episode_index)
        for record in catalog.episodes
    }
    if set(audit.proof_index) != catalog_keys:
        raise TrainStatsComputationError(
            "audit proofs do not exactly cover the episode catalog"
        )
    for record in catalog.episodes:
        key = (record.dataset_id, record.dataset_index, record.episode_index)
        proof = audit.proof_index[key]
        if proof.split != record.split or proof.camera_keys != EXPECTED_AUDITED_CAMERAS:
            raise TrainStatsComputationError(
                f"audit proof does not exactly bind catalog split/cameras for {key}"
            )
    roots = _validate_dataset_roots(catalog, dataset_roots)
    config_sha256, action_dim, state_dim = _validate_libero_config(
        data_config_path, config_loader=config_loader
    )
    records = tuple(
        sorted(
            (record for record in catalog.episodes if record.split == "train"),
            key=lambda item: (item.dataset_index, item.episode_index, item.dataset_id),
        )
    )
    if not records:
        raise TrainStatsComputationError("catalog contains no train episodes")
    proofs: list[EpisodeAuditProof] = []
    for record in records:
        key = (record.dataset_id, record.dataset_index, record.episode_index)
        proof = audit.proof_index[key]
        if proof.split != "train" or proof.split != record.split:
            raise TrainStatsComputationError(
                f"audit split disagrees for train episode {key}"
            )
        proofs.append(proof)

    resolved_output = output.expanduser().resolve()
    if resolved_output.exists() or resolved_output.is_symlink():
        raise FileExistsError(f"train-stats output already exists: {resolved_output}")
    for root in roots:
        if resolved_output == root or root in resolved_output.parents:
            raise TrainStatsComputationError(
                f"train-stats output must stay outside dataset root {root}"
            )
    return TrainStatsPlan(
        catalog=catalog,
        audit=audit,
        roots=roots,
        records=records,
        proofs=tuple(proofs),
        data_config_sha256=config_sha256,
        action_dim=action_dim,
        state_dim=state_dim,
        output=resolved_output,
        timestamp_tolerance_s=float(timestamp_tolerance_s),
    )


def _default_table_reader(payload: bytes) -> Mapping[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - server dependency path
        raise RuntimeError(
            "Computing train stats requires pyarrow on the server; install "
            "project dependencies or inject table_reader in CPU tests"
        ) from exc
    table = pq.read_table(pa.BufferReader(payload), columns=list(_REQUIRED_COLUMNS))
    return {name: table[name] for name in table.column_names}


def _column_to_array(value: Any, *, name: str) -> np.ndarray:
    try:
        if hasattr(value, "combine_chunks"):
            value = value.combine_chunks()
        if hasattr(value, "to_pylist"):
            value = value.to_pylist()
        array = np.asarray(value)
    except Exception as exc:
        raise TrainStatsComputationError(
            f"cannot convert parquet column {name!r} to NumPy"
        ) from exc
    if array.ndim == 0:
        raise TrainStatsComputationError(
            f"parquet column {name!r} must contain one value per row"
        )
    if array.dtype.kind == "O":
        try:
            array = np.asarray(list(value))
        except Exception as exc:
            raise TrainStatsComputationError(
                f"parquet column {name!r} is ragged"
            ) from exc
    return array


def _integer_vector(value: Any, *, name: str, rows: int) -> np.ndarray:
    array = _column_to_array(value, name=name)
    if array.shape == (rows, 1):
        array = array[:, 0]
    if array.shape != (rows,) or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise TrainStatsComputationError(
            f"parquet column {name!r} must be an integer [N] vector"
        )
    return np.asarray(array, dtype=np.int64)


def _numeric_vector(value: Any, *, name: str, rows: int) -> np.ndarray:
    array = _column_to_array(value, name=name)
    if array.shape == (rows, 1):
        array = array[:, 0]
    if array.shape != (rows,) or array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise TrainStatsComputationError(
            f"parquet column {name!r} must be a numeric [N] vector"
        )
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise TrainStatsComputationError(f"parquet column {name!r} is non-finite")
    return result


def _numeric_matrix(
    value: Any,
    *,
    name: str,
    rows: int,
    dimensions: int,
) -> np.ndarray:
    array = _column_to_array(value, name=name)
    if (
        array.shape != (rows, dimensions)
        or array.dtype.kind not in "iuf"
        or array.dtype.kind == "b"
    ):
        raise TrainStatsComputationError(
            f"parquet column {name!r} must have numeric shape [{rows},{dimensions}]"
        )
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise TrainStatsComputationError(f"parquet column {name!r} is non-finite")
    return result


@dataclass(frozen=True, slots=True)
class RawEpisodeStatsRows:
    actions: np.ndarray
    states: np.ndarray


def read_audited_stats_rows(
    record: EpisodeRecord,
    *,
    proof: EpisodeAuditProof,
    dataset_root: Path,
    action_dim: int,
    state_dim: int,
    timestamp_tolerance_s: float,
    table_reader: TableReader | None = None,
) -> RawEpisodeStatsRows:
    key = (record.dataset_id, record.dataset_index, record.episode_index)
    if proof.episode_key != key or proof.split != record.split or record.split != "train":
        raise TrainStatsComputationError(
            "only a matching audited train proof may be read for statistics"
        )
    data_path = _resolve_inside(
        dataset_root, record.data_relpath, label="episode parquet path"
    )
    try:
        # One byte snapshot is both hashed and parsed.  The source path is not
        # reopened by any downstream parser.
        snapshot = data_path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(data_path) from exc
    if sha256(snapshot).hexdigest() != proof.table_sha256:
        raise TrainStatsComputationError(
            f"episode parquet does not match audit proof: {key}"
        )
    reader = table_reader or _default_table_reader
    try:
        table = reader(snapshot)
    except (RuntimeError, TrainStatsComputationError):
        raise
    except Exception as exc:
        raise TrainStatsComputationError(
            f"cannot parse audited parquet snapshot for {key}"
        ) from exc
    if not isinstance(table, Mapping):
        raise TypeError("table_reader must return a column mapping")
    missing = sorted(set(_REQUIRED_COLUMNS).difference(table))
    if missing:
        raise TrainStatsComputationError(
            f"episode parquet is missing required columns: {missing}"
        )
    rows = record.length
    episode_indices = _integer_vector(
        table["episode_index"], name="episode_index", rows=rows
    )
    if not np.all(episode_indices == record.episode_index):
        raise TrainStatsComputationError(
            "episode_index column disagrees with catalog identity"
        )
    frame_indices = _integer_vector(
        table["frame_index"], name="frame_index", rows=rows
    )
    if not np.array_equal(frame_indices, np.arange(rows, dtype=np.int64)):
        raise TrainStatsComputationError("frame_index must be contiguous [0,N)")
    task_indices = _integer_vector(table["task_index"], name="task_index", rows=rows)
    if np.any(task_indices < 0) or np.unique(task_indices).size != 1:
        raise TrainStatsComputationError(
            "task_index must be one constant non-negative value per episode"
        )
    timestamps = _numeric_vector(table["timestamp"], name="timestamp", rows=rows)
    expected = frame_indices.astype(np.float64) / record.fps
    if not np.allclose(
        timestamps,
        expected,
        rtol=0.0,
        atol=timestamp_tolerance_s,
    ):
        raise TrainStatsComputationError(
            "timestamps must equal frame_index/fps within tolerance"
        )
    actions = _numeric_matrix(
        table[ACTION_KEY],
        name=ACTION_KEY,
        rows=rows,
        dimensions=action_dim,
    )
    states = _numeric_matrix(
        table[STATE_KEY],
        name=STATE_KEY,
        rows=rows,
        dimensions=state_dim,
    )
    return RawEpisodeStatsRows(actions=actions, states=states)


def compute_train_stats(
    plan: TrainStatsPlan,
    *,
    table_reader: TableReader | None = None,
) -> FastWAMLiberoTrainStats:
    """Reduce all N raw rows of every authorized train episode."""

    if not isinstance(plan, TrainStatsPlan):
        raise TypeError("plan must be TrainStatsPlan")
    action_min: np.ndarray | None = None
    action_max: np.ndarray | None = None
    state_min: np.ndarray | None = None
    state_max: np.ndarray | None = None
    frames = 0
    for record, proof in zip(plan.records, plan.proofs, strict=True):
        rows = read_audited_stats_rows(
            record,
            proof=proof,
            dataset_root=plan.roots[record.dataset_index],
            action_dim=plan.action_dim,
            state_dim=plan.state_dim,
            timestamp_tolerance_s=plan.timestamp_tolerance_s,
            table_reader=table_reader,
        )
        cur_action_min = rows.actions.min(axis=0)
        cur_action_max = rows.actions.max(axis=0)
        cur_state_min = rows.states.min(axis=0)
        cur_state_max = rows.states.max(axis=0)
        action_min = (
            cur_action_min
            if action_min is None
            else np.minimum(action_min, cur_action_min)
        )
        action_max = (
            cur_action_max
            if action_max is None
            else np.maximum(action_max, cur_action_max)
        )
        state_min = (
            cur_state_min if state_min is None else np.minimum(state_min, cur_state_min)
        )
        state_max = (
            cur_state_max if state_max is None else np.maximum(state_max, cur_state_max)
        )
        frames += record.length
    if any(value is None for value in (action_min, action_max, state_min, state_max)):
        raise TrainStatsComputationError("no train rows were reduced")
    return FastWAMLiberoTrainStats(
        action_min=tuple(float(value) for value in action_min),
        action_max=tuple(float(value) for value in action_max),
        state_min=tuple(float(value) for value in state_min),
        state_max=tuple(float(value) for value in state_max),
        num_episodes=len(plan.records),
        num_transition=frames,
    )


def _git_clean_commit(repository: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "0" * 40
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        return "0" * 40
    return commit


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def publish_train_stats(
    plan: TrainStatsPlan,
    stats: FastWAMLiberoTrainStats,
    *,
    git_commit: str,
) -> TrainStatsManifest:
    """Atomically publish and self-verify the two-file directory artifact."""

    stats_payload = encode_train_stats_json(stats)
    manifest = TrainStatsManifest(
        stats_file_sha256=sha256(stats_payload).hexdigest(),
        catalog_sha256=plan.catalog.content_sha256,
        audit_report_sha256=plan.audit.report_sha256,
        data_config_sha256=plan.data_config_sha256,
        train_episode_set_sha256=compute_train_episode_set_sha256(
            plan.episode_sources
        ),
        train_episode_count=len(plan.records),
        train_frame_count=sum(record.length for record in plan.records),
        action_dim=plan.action_dim,
        state_dim=plan.state_dim,
        git_commit=git_commit,
    )
    manifest_payload = encode_train_stats_manifest_json(manifest)
    output = plan.output
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.warm-train-stats.lock"
    staging = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    with artifact_claim(lock_path, purpose="publish WARM train-only stats"):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"train-stats output already exists: {output}")
        staging.mkdir()
        try:
            _write_exclusive(staging / TRAIN_STATS_FILENAME, stats_payload)
            _write_exclusive(
                staging / TRAIN_STATS_MANIFEST_FILENAME, manifest_payload
            )
            load_train_stats_artifact(
                staging,
                expected_catalog_sha256=plan.catalog.content_sha256,
                expected_audit_report_sha256=plan.audit.report_sha256,
                expected_data_config_sha256=plan.data_config_sha256,
                expected_train_episodes=plan.episode_sources,
            )
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"train-stats output already exists: {output}")
            os.rename(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    git_commit = _git_clean_commit(repository)
    plan = prepare_plan(
        catalog_path=args.catalog,
        audit_report_path=args.audit_report,
        dataset_roots=args.dataset_root,
        data_config_path=args.data_config,
        output=args.output,
        timestamp_tolerance_s=args.timestamp_tolerance_s,
    )
    if plan.output == repository or repository in plan.output.parents:
        raise TrainStatsComputationError(
            "formal train statistics must be written outside the Git worktree"
        )
    stats = compute_train_stats(plan)
    manifest = publish_train_stats(
        plan,
        stats,
        git_commit=git_commit,
    )
    print(f"output={plan.output}")
    print(f"stats_sha256={manifest.stats_file_sha256}")
    print(f"manifest_sha256={manifest.content_sha256}")
    print(
        f"train_episodes={manifest.train_episode_count} "
        f"train_frames={manifest.train_frame_count} "
        f"action_dim={manifest.action_dim} state_dim={manifest.state_dim}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
