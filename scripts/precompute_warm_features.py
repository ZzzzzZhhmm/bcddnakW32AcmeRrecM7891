#!/usr/bin/env python3
"""Precompute immutable WARM M1 episode features on a CUDA server.

The module is importable without Torch, PyArrow, Transformers, or Hydra.  The
``--plan-only`` validates static contracts and file snapshots, then prints the
exact work plan without creating an artifact.  CUDA availability, resolved
Hydra processor semantics, and every parquet/MP4 are revalidated on the server
execution path.  Heavy imports occur only after the static plan has passed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

from fastwam.datasets.lerobot.audit import LerobotAuditReport, load_audit_report
from fastwam.datasets.lerobot.episode_catalog import (
    DatasetDescriptor,
    EpisodeCatalog,
    EpisodeRecord,
)
from fastwam.memory.action_contract import ActionSpaceContract
from fastwam.memory.manifest import ManifestError, sha256_file, sha256_path_tree
from fastwam.memory.train_stats import (
    TRAIN_STATS_FILENAME,
    TRAIN_STATS_MANIFEST_FILENAME,
    TrainEpisodeSource,
    TrainStatsContractError,
    load_train_stats_artifact,
)
from fastwam.utils.artifact_claim import artifact_claim


PLAN_SCHEMA = "warm.feature-precompute-plan"
PLAN_VERSION = 1
ENCODER_CONTRACT_SCHEMA = "warm.feature-encoder"
ENCODER_CONTRACT_VERSION = 2
CAMERA_CONTRACT_SCHEMA = "warm.camera-layout"
CAMERA_CONTRACT_VERSION = 1
SUMMARY_SCHEMA = "warm.feature-precompute-summary"
SUMMARY_VERSION = 1
_PINNED_HF_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class WarmFeaturePrecomputeError(ValueError):
    """Raised when server inputs cannot define one reproducible feature set."""


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read complete audited LeRobot episodes and precompute factual "
            "DINO/Wan-VAE features without temporal-window padding."
        )
    )
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument(
        "--benchmark-profile",
        choices=("libero", "robotwin"),
        default="libero",
        help="Closed processor/camera/action profile; never inferred from shapes.",
    )
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        required=True,
        action="append",
        type=Path,
        help="Repeat in the exact catalog dataset_index order.",
    )
    parser.add_argument("--dataset-stats", required=True, type=Path)
    parser.add_argument("--dataset-stats-manifest", required=True, type=Path)
    parser.add_argument("--dino-checkpoint", required=True, type=Path)
    parser.add_argument("--dino-model-id", default="facebook/dinov2-base")
    parser.add_argument("--dino-revision", required=True)
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--include-vae", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "dev"),
        default=None,
        help="Defaults to both train and dev; test is intentionally unsupported.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help=(
            "LeRobot video key; defaults to observation.images.image and "
            "observation.images.wrist_image."
        ),
    )
    parser.add_argument("--semantic-camera", default=None)
    parser.add_argument(
        "--concat-mode",
        choices=("horizontal", "vertical", "robotwin"),
        default="horizontal",
    )
    parser.add_argument(
        "--context-mode",
        choices=("task-conditioned", "visual-only"),
        default="task-conditioned",
    )
    parser.add_argument("--video-backend", choices=("pyav",), default="pyav")
    parser.add_argument("--timestamp-tolerance-s", type=float, default=1e-4)
    parser.add_argument("--dino-batch-size", type=_positive_int, default=64)
    parser.add_argument("--vae-batch-size", type=_positive_int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--max-episodes-per-split",
        type=_positive_int,
        default=None,
        help="Smoke-only truncation; requires --allow-incomplete-smoke.",
    )
    parser.add_argument("--allow-incomplete-smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate and print the plan; do not import GPU dependencies or write output.",
    )
    return parser


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _tree_sha256(path: Path) -> tuple[str, int]:
    """Hash a checkpoint file/tree by relative name, size, and file digest."""

    try:
        return sha256_path_tree(path)
    except ManifestError as exc:
        raise WarmFeaturePrecomputeError(str(exc)) from exc


def _validate_dataset_roots(
    catalog: EpisodeCatalog,
    roots: Sequence[Path],
) -> tuple[Path, ...]:
    descriptors = tuple(sorted(catalog.datasets, key=lambda item: item.dataset_index))
    if [item.dataset_index for item in descriptors] != list(range(len(descriptors))):
        raise WarmFeaturePrecomputeError(
            "catalog dataset indices must be contiguous and start at zero"
        )
    if len(roots) != len(descriptors):
        raise WarmFeaturePrecomputeError(
            "--dataset-root count must match the catalog dataset count"
        )
    resolved_roots: list[Path] = []
    for descriptor, root_value in zip(descriptors, roots, strict=True):
        root = root_value.expanduser().resolve()
        info_path = root / "meta" / "info.json"
        episodes_path = root / "meta" / "episodes.jsonl"
        if not root.is_dir() or not info_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(
                f"dataset root {root} is missing meta/info.json or meta/episodes.jsonl"
            )
        if sha256_file(info_path) != descriptor.info_sha256:
            raise WarmFeaturePrecomputeError(
                f"dataset metadata changed after cataloging: {descriptor.dataset_id}"
            )
        if sha256_file(episodes_path) != descriptor.episodes_sha256:
            raise WarmFeaturePrecomputeError(
                f"episode metadata changed after cataloging: {descriptor.dataset_id}"
            )
        resolved_roots.append(root)
    return tuple(resolved_roots)


def _task_vocabulary(
    catalog: EpisodeCatalog,
) -> tuple[str, ...]:
    """Return the catalog-bound train/dev retrieval vocabulary.

    Test instructions must not affect the dimensionality or ordering of an M1
    retrieval key.  Every train-only, dev-only, combined, and smoke precompute
    uses the same complete train+dev vocabulary so independently produced
    official feature collections remain encoder-compatible.
    """

    split_set = frozenset(("train", "dev"))
    tasks = tuple(
        sorted(
            {
                episode.primary_task
                for episode in catalog.episodes
                if episode.split in split_set
            }
        )
    )
    if not tasks or any(not task.strip() or task.strip() != task for task in tasks):
        raise WarmFeaturePrecomputeError("catalog contains an invalid task vocabulary")
    return tasks


def _selected_records(
    catalog: EpisodeCatalog,
    splits: Sequence[str],
    *,
    max_per_split: int | None,
) -> tuple[EpisodeRecord, ...]:
    selected: list[EpisodeRecord] = []
    for split in splits:
        rows = sorted(
            (record for record in catalog.episodes if record.split == split),
            key=lambda item: (item.dataset_index, item.episode_index),
        )
        if not rows:
            raise WarmFeaturePrecomputeError(f"catalog has no episodes in split {split!r}")
        selected.extend(rows if max_per_split is None else rows[:max_per_split])
    return tuple(selected)


def _expected_train_episode_sources(
    catalog: EpisodeCatalog,
    audit: LerobotAuditReport,
) -> tuple[TrainEpisodeSource, ...]:
    proof_index = audit.proof_index
    return tuple(
        TrainEpisodeSource(
            dataset_id=record.dataset_id,
            dataset_index=record.dataset_index,
            episode_index=record.episode_index,
            source_episode_sha256=proof_index[
                (record.dataset_id, record.dataset_index, record.episode_index)
            ].source_episode_sha256,
        )
        for record in sorted(
            (item for item in catalog.episodes if item.split == "train"),
            key=lambda item: (item.dataset_index, item.episode_index, item.dataset_id),
        )
    )


@dataclass(frozen=True, slots=True)
class PrecomputePlan:
    catalog: EpisodeCatalog
    audit: LerobotAuditReport
    roots: tuple[Path, ...]
    records: tuple[EpisodeRecord, ...]
    splits: tuple[str, ...]
    task_vocabulary: tuple[str, ...]
    stats_raw: bytes
    stats_sha256: str
    stats_manifest_raw: bytes
    stats_manifest_file_sha256: str
    stats_manifest_content_sha256: str
    data_config_raw: bytes
    data_config_sha256: str
    dino_checkpoint_sha256: str
    dino_checkpoint_files: int
    vae_checkpoint_sha256: str | None
    output: Path
    camera_keys: tuple[str, ...]
    semantic_camera: str
    incomplete_smoke: bool

    def to_dict(self, args: argparse.Namespace) -> dict[str, object]:
        split_counts = {
            split: sum(record.split == split for record in self.records)
            for split in self.splits
        }
        return {
            "schema": PLAN_SCHEMA,
            "version": PLAN_VERSION,
            "benchmark_profile": args.benchmark_profile,
            "catalog_sha256": self.catalog.content_sha256,
            "audit_report_sha256": self.audit.report_sha256,
            "dataset_roots": [str(path) for path in self.roots],
            "splits": list(self.splits),
            "split_episode_counts": split_counts,
            "episode_count": len(self.records),
            "task_count": len(self.task_vocabulary),
            "task_vocabulary_sha256": hashlib.sha256(
                b"warm.task-vocabulary.v1\0" + _canonical_json(self.task_vocabulary)
            ).hexdigest(),
            "data_config_sha256": self.data_config_sha256,
            "dataset_stats_sha256": self.stats_sha256,
            "dataset_stats_manifest_file_sha256": self.stats_manifest_file_sha256,
            "dataset_stats_manifest_content_sha256": (
                self.stats_manifest_content_sha256
            ),
            "dino": {
                "model_id": args.dino_model_id,
                "revision": args.dino_revision,
                "checkpoint_sha256": self.dino_checkpoint_sha256,
                "checkpoint_file_count": self.dino_checkpoint_files,
            },
            "vae": {
                "enabled": bool(args.include_vae),
                "checkpoint_sha256": self.vae_checkpoint_sha256,
            },
            "camera_keys": list(self.camera_keys),
            "semantic_camera": self.semantic_camera,
            "context_mode": args.context_mode,
            "concat_mode": args.concat_mode,
            "video_backend": args.video_backend,
            "timestamp_tolerance_s": args.timestamp_tolerance_s,
            "device": args.device,
            "dtype": args.dtype,
            "dino_batch_size": args.dino_batch_size,
            "vae_batch_size": args.vae_batch_size,
            "incomplete_smoke": self.incomplete_smoke,
            "output": str(self.output),
        }


def _build_plan(args: argparse.Namespace) -> PrecomputePlan:
    if args.max_episodes_per_split is not None and not args.allow_incomplete_smoke:
        raise WarmFeaturePrecomputeError(
            "--max-episodes-per-split requires --allow-incomplete-smoke"
        )
    if args.allow_incomplete_smoke and args.max_episodes_per_split is None:
        raise WarmFeaturePrecomputeError(
            "--allow-incomplete-smoke requires --max-episodes-per-split"
        )
    if not math.isfinite(args.timestamp_tolerance_s) or args.timestamp_tolerance_s < 0:
        raise WarmFeaturePrecomputeError(
            "--timestamp-tolerance-s must be finite and non-negative"
        )
    required_concat = "horizontal" if args.benchmark_profile == "libero" else "robotwin"
    if args.concat_mode != required_concat:
        raise WarmFeaturePrecomputeError(
            f"{args.benchmark_profile} profile requires concat-mode={required_concat}"
        )
    if (
        not isinstance(args.dino_revision, str)
        or _PINNED_HF_COMMIT.fullmatch(args.dino_revision) is None
    ):
        raise WarmFeaturePrecomputeError(
            "--dino-revision must be a lowercase 40-character commit SHA"
        )
    if not isinstance(args.dino_model_id, str) or not args.dino_model_id.strip():
        raise WarmFeaturePrecomputeError("--dino-model-id must be non-empty")
    if args.include_vae and args.vae_checkpoint is None:
        raise WarmFeaturePrecomputeError(
            "--include-vae requires an explicit local --vae-checkpoint"
        )
    if not args.include_vae and args.vae_checkpoint is not None:
        raise WarmFeaturePrecomputeError(
            "--vae-checkpoint is accepted only together with --include-vae"
        )
    if args.allow_dirty and not args.allow_incomplete_smoke:
        raise WarmFeaturePrecomputeError(
            "--allow-dirty is restricted to explicitly incomplete smoke artifacts"
        )

    catalog = EpisodeCatalog.load(args.catalog.expanduser().resolve())
    audit = load_audit_report(args.audit_report.expanduser().resolve())
    if not audit.episode_tables_hashed:
        raise WarmFeaturePrecomputeError(
            "feature precompute requires an audit with --hash-episode-tables"
        )
    if audit.catalog_sha256 != catalog.content_sha256:
        raise WarmFeaturePrecomputeError("audit report does not bind the supplied catalog")
    if audit.cross_split_duplicate_count:
        raise WarmFeaturePrecomputeError(
            "audited raw episode content is duplicated across splits"
        )
    if set(audit.proof_index) != {
        (record.dataset_id, record.dataset_index, record.episode_index)
        for record in catalog.episodes
    }:
        raise WarmFeaturePrecomputeError(
            "audit episode proofs do not exactly cover the catalog"
        )
    roots = _validate_dataset_roots(catalog, args.dataset_root)

    config_raw, config_value, config_sha256 = _read_json_snapshot_or_yaml_bytes(
        args.data_config
    )
    del config_value
    stats_path = args.dataset_stats.expanduser().resolve()
    stats_manifest_path = args.dataset_stats_manifest.expanduser().resolve()
    if stats_path.name != TRAIN_STATS_FILENAME:
        raise WarmFeaturePrecomputeError(
            f"--dataset-stats must be named {TRAIN_STATS_FILENAME!r}"
        )
    if stats_manifest_path.name != TRAIN_STATS_MANIFEST_FILENAME:
        raise WarmFeaturePrecomputeError(
            "--dataset-stats-manifest must be named "
            f"{TRAIN_STATS_MANIFEST_FILENAME!r}"
        )
    if stats_path.parent != stats_manifest_path.parent:
        raise WarmFeaturePrecomputeError(
            "dataset stats and manifest must belong to the same immutable artifact"
        )
    try:
        stats_raw = stats_path.read_bytes()
        stats_manifest_raw = stats_manifest_path.read_bytes()
        train_sources = _expected_train_episode_sources(catalog, audit)
        if args.benchmark_profile == "libero":
            stats_artifact = load_train_stats_artifact(
                stats_path.parent,
                expected_catalog_sha256=catalog.content_sha256,
                expected_audit_report_sha256=audit.report_sha256,
                expected_data_config_sha256=config_sha256,
                expected_train_episodes=train_sources,
            )
        else:
            from fastwam.memory.robotwin_train_stats import (
                load_robotwin_train_stats_artifact,
            )

            stats_artifact = load_robotwin_train_stats_artifact(
                stats_path.parent,
                expected_catalog_sha256=catalog.content_sha256,
                expected_audit_report_sha256=audit.report_sha256,
                expected_data_config_sha256=config_sha256,
                expected_train_episodes=train_sources,
            )
        if stats_raw != stats_path.read_bytes() or stats_manifest_raw != (
            stats_manifest_path.read_bytes()
        ):
            raise WarmFeaturePrecomputeError(
                "train-statistics artifact changed while the plan was validated"
            )
    except TrainStatsContractError as exc:
        raise WarmFeaturePrecomputeError(
            f"invalid train-only normalization artifact: {exc}"
        ) from exc
    except OSError as exc:
        raise WarmFeaturePrecomputeError(
            "cannot snapshot train-only normalization artifact"
        ) from exc
    expected_dims = (7, 8) if args.benchmark_profile == "libero" else (14, 14)
    if (
        stats_artifact.stats.action_dim,
        stats_artifact.stats.state_dim,
    ) != expected_dims:
        raise WarmFeaturePrecomputeError(
            "train-only normalization dimensions disagree with benchmark profile"
        )
    expected_train_frames = sum(
        record.length for record in catalog.episodes if record.split == "train"
    )
    if stats_artifact.manifest.train_frame_count != expected_train_frames:
        raise WarmFeaturePrecomputeError(
            "train-only normalization frame count disagrees with the catalog"
        )
    stats_sha256 = stats_artifact.manifest.stats_file_sha256
    stats_manifest_file_sha256 = hashlib.sha256(stats_manifest_raw).hexdigest()
    stats_manifest_content_sha256 = stats_artifact.manifest.content_sha256
    dino_hash, dino_files = _tree_sha256(args.dino_checkpoint)
    vae_hash = (
        None
        if args.vae_checkpoint is None
        else sha256_file(args.vae_checkpoint.expanduser().resolve())
    )

    splits = tuple(dict.fromkeys(args.split or ("train", "dev")))
    records = _selected_records(
        catalog,
        splits,
        max_per_split=args.max_episodes_per_split,
    )
    default_cameras = (
        (
            "observation.images.image",
            "observation.images.wrist_image",
        )
        if args.benchmark_profile == "libero"
        else (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        )
    )
    cameras = tuple(
        dict.fromkeys(
            args.camera or default_cameras
        )
    )
    if not cameras or any(not camera.strip() for camera in cameras):
        raise WarmFeaturePrecomputeError("camera keys must be non-empty")
    expected_cameras = default_cameras
    if cameras != expected_cameras:
        raise WarmFeaturePrecomputeError(
            f"WARM M1 {args.benchmark_profile} requires the exact camera order "
            f"{expected_cameras!r}"
        )
    semantic_camera = args.semantic_camera or expected_cameras[0]
    if semantic_camera != expected_cameras[0]:
        raise WarmFeaturePrecomputeError(
            "WARM M1 effect semantics must use the profile's head/external camera"
        )
    if audit.audited_camera_keys != cameras:
        raise WarmFeaturePrecomputeError(
            "feature cameras must exactly match the audit's ordered camera contract: "
            f"{cameras!r} != {audit.audited_camera_keys!r}"
        )
    dino_checkpoint = args.dino_checkpoint.expanduser().resolve()
    if not dino_checkpoint.is_dir():
        raise WarmFeaturePrecomputeError(
            "--dino-checkpoint must be a complete local snapshot directory"
        )

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"feature output already exists at {output}")
    protected_directories = (
        *roots,
        dino_checkpoint,
        stats_path.parent,
        _repository_root(),
    )
    for protected in protected_directories:
        directory = protected if protected.is_dir() else protected.parent
        if output == directory or directory in output.parents:
            raise WarmFeaturePrecomputeError(
                f"feature output must stay outside protected input directory {directory}"
            )

    return PrecomputePlan(
        catalog=catalog,
        audit=audit,
        roots=roots,
        records=records,
        splits=splits,
        task_vocabulary=_task_vocabulary(catalog),
        stats_raw=stats_raw,
        stats_sha256=stats_sha256,
        stats_manifest_raw=stats_manifest_raw,
        stats_manifest_file_sha256=stats_manifest_file_sha256,
        stats_manifest_content_sha256=stats_manifest_content_sha256,
        data_config_raw=config_raw,
        data_config_sha256=config_sha256,
        dino_checkpoint_sha256=dino_hash,
        dino_checkpoint_files=dino_files,
        vae_checkpoint_sha256=vae_hash,
        output=output,
        camera_keys=cameras,
        semantic_camera=semantic_camera,
        incomplete_smoke=args.max_episodes_per_split is not None,
    )


def _read_json_snapshot_or_yaml_bytes(path: Path) -> tuple[bytes, None, str]:
    """Bind an opaque Hydra YAML config by exact bytes without parsing locally."""

    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
    except OSError as exc:
        raise WarmFeaturePrecomputeError(f"cannot read data config {resolved}") from exc
    if not raw.strip():
        raise WarmFeaturePrecomputeError("data config must not be empty")
    return raw, None, hashlib.sha256(raw).hexdigest()


def _software_provenance() -> dict[str, object]:
    repository = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WarmFeaturePrecomputeError("cannot inspect the WARM Git repository") from exc
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise WarmFeaturePrecomputeError("Git returned an invalid commit SHA")
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _write_exclusive(path, encoded)


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _load_processor(
    data_config: Path, stats_path: Path, *, benchmark_profile: str = "libero"
) -> Any:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.datasets.lerobot.utils.normalizer import (
        load_dataset_stats_from_json,
    )

    raw = OmegaConf.load(data_config)
    wrapped = OmegaConf.create({"data": raw})
    OmegaConf.resolve(wrapped)
    train = wrapped.data.train
    if benchmark_profile == "robotwin":
        from fastwam.memory.processor_contract import (
            extract_m1_robotwin_processor_recipe,
            validate_processor_instance,
        )

        recipe_value = OmegaConf.to_container(wrapped, resolve=True)
        if not isinstance(recipe_value, Mapping):
            raise WarmFeaturePrecomputeError("resolved RoboTwin data config is invalid")
        recipe = extract_m1_robotwin_processor_recipe(recipe_value)
        processor = instantiate(wrapped.data.train.processor)
        processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(str(stats_path))
        )
        processor.eval()
        validate_processor_instance(processor, recipe)
        return processor
    if benchmark_profile != "libero":
        raise WarmFeaturePrecomputeError("unknown benchmark profile")
    image_meta = tuple(train.shape_meta.images)
    image_keys = tuple(str(item.key) for item in image_meta)
    image_shapes = tuple(tuple(int(value) for value in item.shape) for item in image_meta)
    if image_keys != ("image", "wrist_image") or image_shapes != (
        (3, 224, 224),
        (3, 224, 224),
    ):
        raise WarmFeaturePrecomputeError(
            "data config does not define FastWAM LIBERO's exact two-camera shape_meta"
        )
    if int(train.processor.num_output_cameras) != 2:
        raise WarmFeaturePrecomputeError(
            "data config must set processor.num_output_cameras=2"
        )
    if str(train.concat_multi_camera) != "horizontal":
        raise WarmFeaturePrecomputeError(
            "data config must use concat_multi_camera=horizontal"
        )
    if tuple(int(value) for value in train.video_size) != (224, 448):
        raise WarmFeaturePrecomputeError(
            "data config must use the FastWAM LIBERO video_size [224,448]"
        )
    action_meta = tuple(
        (str(item.key), int(item.raw_shape), int(item.shape))
        for item in train.shape_meta.action
    )
    state_meta = tuple(
        (str(item.key), int(item.raw_shape), int(item.shape))
        for item in train.shape_meta.state
    )
    if action_meta != (("default", 7, 7),):
        raise WarmFeaturePrecomputeError(
            "data config must define exactly one default 7D LIBERO action field"
        )
    if state_meta != (("default", 8, 8),):
        raise WarmFeaturePrecomputeError(
            "data config must define exactly one default 8D LIBERO state field"
        )
    processor_config = train.processor
    if processor_config.action_state_transforms is not None:
        raise WarmFeaturePrecomputeError(
            "WARM M1 LIBERO requires action_state_transforms=null"
        )
    if bool(processor_config.use_stepwise_action_norm):
        raise WarmFeaturePrecomputeError(
            "WARM M1 LIBERO requires global rather than stepwise normalization"
        )
    if str(processor_config.norm_default_mode) != "min/max":
        raise WarmFeaturePrecomputeError(
            "WARM M1 LIBERO requires norm_default_mode=min/max"
        )
    if processor_config.norm_exception_mode is not None:
        raise WarmFeaturePrecomputeError(
            "WARM M1 LIBERO requires norm_exception_mode=null"
        )
    merger_config = processor_config.action_state_merger
    if str(merger_config.get("_target_", "")) != (
        "fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign"
    ):
        raise WarmFeaturePrecomputeError(
            "WARM M1 LIBERO requires the exact ConcatLeftAlign action/state merger"
        )
    if merger_config.get("action_target_dim") is not None or merger_config.get(
        "state_target_dim"
    ) is not None:
        raise WarmFeaturePrecomputeError(
            "WARM M1 LIBERO does not permit action/state merger padding"
        )
    transform_configs = tuple(train.processor.val_transforms)
    transform_targets = tuple(str(item.get("_target_", "")) for item in transform_configs)
    if transform_targets != (
        "fastwam.datasets.lerobot.transforms.image.ToTensor",
        "torchvision.transforms.Resize",
    ):
        raise WarmFeaturePrecomputeError(
            "data config must use only FastWAM ToTensor then torchvision Resize "
            "for factual LIBERO frames"
        )
    if set(transform_configs[0].keys()) != {"_target_"}:
        raise WarmFeaturePrecomputeError(
            "FastWAM LIBERO ToTensor config must not contain extra parameters"
        )
    if set(transform_configs[1].keys()) != {"_target_", "size"}:
        raise WarmFeaturePrecomputeError(
            "FastWAM LIBERO Resize config must contain only _target_ and size"
        )
    resize_size = tuple(int(value) for value in transform_configs[1].size)
    if resize_size != (224, 224):
        raise WarmFeaturePrecomputeError(
            "FastWAM LIBERO validation Resize must be [224,224]"
        )
    processor = instantiate(wrapped.data.train.processor)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    processor.eval()
    return processor


def _runtime_provenance(device: str) -> dict[str, object]:
    """Capture numerical-runtime versions in the encoder cache contract."""

    from fastwam.memory.runtime_fingerprint import (
        RuntimeFingerprintError,
        current_encoder_runtime,
    )

    try:
        return current_encoder_runtime(device)
    except RuntimeFingerprintError as exc:
        raise WarmFeaturePrecomputeError(str(exc)) from exc


def _validate_libero_processor(processor: Any) -> str:
    if int(processor.action_output_dim) != 7 or int(processor.proprio_output_dim) != 8:
        raise WarmFeaturePrecomputeError(
            "WARM M1 LIBERO requires action_dim=7 and proprio_dim=8"
        )
    action_meta = tuple(
        (str(item["key"]), int(item["raw_shape"]), int(item["shape"]))
        for item in processor.shape_meta["action"]
    )
    state_meta = tuple(
        (str(item["key"]), int(item["raw_shape"]), int(item["shape"]))
        for item in processor.shape_meta["state"]
    )
    if action_meta != (("default", 7, 7),) or state_meta != (("default", 8, 8),):
        raise WarmFeaturePrecomputeError(
            "instantiated processor does not preserve exact LIBERO action/state metadata"
        )
    if processor.action_state_transforms is not None:
        raise WarmFeaturePrecomputeError(
            "instantiated processor unexpectedly applies action/state transforms"
        )
    if bool(processor.use_stepwise_action_norm):
        raise WarmFeaturePrecomputeError(
            "instantiated processor unexpectedly uses stepwise normalization"
        )
    if str(processor.norm_default_mode) != "min/max":
        raise WarmFeaturePrecomputeError(
            "instantiated processor does not use min/max normalization"
        )
    normalization_mode = "global:min/max"
    if processor.norm_exception_mode not in (None, {}):
        raise WarmFeaturePrecomputeError(
            "WARM M1 LIBERO does not support per-field normalization exceptions"
        )
    merger = processor.action_state_merger
    merger_type = f"{type(merger).__module__}.{type(merger).__name__}"
    if merger_type != (
        "fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign"
    ):
        raise WarmFeaturePrecomputeError(
            "instantiated processor does not use the exact ConcatLeftAlign merger"
        )
    if merger.action_target_dim is not None or merger.state_target_dim is not None:
        raise WarmFeaturePrecomputeError(
            "instantiated processor unexpectedly pads action or state dimensions"
        )
    delta_mask = processor.delta_action_dim_mask
    if delta_mask is None or set(delta_mask) != {"default"}:
        raise WarmFeaturePrecomputeError("unexpected LIBERO delta-action contract")
    mask = tuple(bool(value) for value in delta_mask["default"].tolist())
    if mask != (True, True, True, True, True, True, False):
        raise WarmFeaturePrecomputeError(
            f"unexpected LIBERO delta-action mask: {mask!r}"
        )
    return normalization_mode


def _validate_robotwin_processor(processor: Any) -> str:
    if int(processor.action_output_dim) != 14 or int(processor.proprio_output_dim) != 14:
        raise WarmFeaturePrecomputeError(
            "RoboTwin M1 requires native action_dim=14 and proprio_dim=14"
        )
    action_meta = tuple(
        (str(item["key"]), int(item["raw_shape"]), int(item["shape"]))
        for item in processor.shape_meta["action"]
    )
    state_meta = tuple(
        (str(item["key"]), int(item["raw_shape"]), int(item["shape"]))
        for item in processor.shape_meta["state"]
    )
    if action_meta != (("default", 14, 14),) or state_meta != (("default", 14, 14),):
        raise WarmFeaturePrecomputeError("RoboTwin processor metadata must stay 14D")
    if processor.action_state_transforms is not None:
        raise WarmFeaturePrecomputeError("RoboTwin qpos preprocessing must be identity")
    if bool(processor.use_stepwise_action_norm) or str(processor.norm_default_mode) != "z-score":
        raise WarmFeaturePrecomputeError("RoboTwin M1 requires global z-score normalization")
    if processor.norm_exception_mode not in (None, {}):
        raise WarmFeaturePrecomputeError("RoboTwin M1 forbids normalization exceptions")
    if processor.delta_action_dim_mask not in (None, {}):
        raise WarmFeaturePrecomputeError("RoboTwin qpos must not use a delta-action mask")
    return "global:z-score"


def _processor_camera_key(source_key: str) -> str:
    prefix = "observation.images."
    if not source_key.startswith(prefix) or len(source_key) == len(prefix):
        raise WarmFeaturePrecomputeError(
            f"unsupported LeRobot camera key for LIBERO: {source_key!r}"
        )
    return source_key[len(prefix) :]


def _run_server_precompute(args: argparse.Namespace, plan: PrecomputePlan) -> None:
    """Execute the validated plan; all imports in this function are server-only."""

    # Server-only imports are deliberately below the plan-only return path.
    from fastwam.datasets.lerobot.full_episode_reader import (
        read_full_lerobot_episode,
    )
    from fastwam.memory.feature_cache import (
        FeatureCacheMetadata,
        load_episode_feature_cache,
        save_episode_feature_cache,
    )
    from fastwam.memory.feature_precompute import (
        FastWAMProcessorAdapter,
        assemble_episode_features,
    )
    from fastwam.memory.server_feature_encoders import (
        DinoV2FactualEncoder,
        FastWAMImageAdapter,
    )

    software = _software_provenance()
    if software["git_dirty"] and not args.allow_dirty:
        raise WarmFeaturePrecomputeError(
            "refusing to publish official features from a dirty Git tree; "
            "commit first or use --allow-dirty only for smoke/debug output"
        )
    if plan.incomplete_smoke and not args.allow_incomplete_smoke:
        raise AssertionError("incomplete smoke plan lost its explicit opt-in")
    runtime = _runtime_provenance(args.device)

    output = plan.output
    output.parent.mkdir(parents=True, exist_ok=True)
    claim_path = output.parent / f".{output.name}.warm-feature-precompute.lock"
    with artifact_claim(claim_path, purpose=f"publish WARM features: {output}"):
        if output.exists():
            raise FileExistsError(f"feature output already exists at {output}")
        staging = output.parent / f".{output.name}.{uuid4().hex}.staging"
        staging.mkdir()
        try:
            contracts = staging / "contracts"
            contracts.mkdir()
            stats_copy = contracts / "dataset_stats.source.json"
            _write_exclusive(stats_copy, plan.stats_raw)
            if sha256_file(stats_copy) != plan.stats_sha256:
                raise WarmFeaturePrecomputeError("dataset-statistics copy hash mismatch")
            stats_manifest_copy = contracts / "train_stats_manifest.source.json"
            _write_exclusive(stats_manifest_copy, plan.stats_manifest_raw)
            if sha256_file(stats_manifest_copy) != plan.stats_manifest_file_sha256:
                raise WarmFeaturePrecomputeError(
                    "train-statistics manifest copy hash mismatch"
                )
            config_copy = contracts / "data_config.source.yaml"
            _write_exclusive(config_copy, plan.data_config_raw)
            if sha256_file(config_copy) != plan.data_config_sha256:
                raise WarmFeaturePrecomputeError("data-config copy hash mismatch")

            processor = _load_processor(
                config_copy,
                stats_copy,
                benchmark_profile=args.benchmark_profile,
            )
            normalization_mode = (
                _validate_libero_processor(processor)
                if args.benchmark_profile == "libero"
                else _validate_robotwin_processor(processor)
            )
            processor_adapter = FastWAMProcessorAdapter(processor, tensor_backend="torch")
            camera_mapping = {
                source: _processor_camera_key(source) for source in plan.camera_keys
            }
            processor_camera_keys = tuple(
                camera_mapping[source] for source in plan.camera_keys
            )
            if len(set(processor_camera_keys)) != len(processor_camera_keys):
                raise WarmFeaturePrecomputeError(
                    "LeRobot camera keys collapse to duplicate FastWAM camera keys"
                )
            image_adapter = FastWAMImageAdapter(
                processor,
                processor_camera_keys,
                args.concat_mode,
                benchmark_profile=args.benchmark_profile,
            )
            before_dino_hash = _tree_sha256(
                args.dino_checkpoint.expanduser().resolve()
            )
            if before_dino_hash != (
                plan.dino_checkpoint_sha256,
                plan.dino_checkpoint_files,
            ):
                raise WarmFeaturePrecomputeError(
                    "DINO checkpoint changed after the precompute plan was validated"
                )
            dino = DinoV2FactualEncoder.from_pretrained(
                local_checkpoint=args.dino_checkpoint.expanduser().resolve(),
                model_id=args.dino_model_id,
                revision=args.dino_revision,
                device=args.device,
                torch_dtype=_torch_dtype(args.dtype),
            )
            after_dino_hash = _tree_sha256(
                args.dino_checkpoint.expanduser().resolve()
            )
            if after_dino_hash != before_dino_hash:
                raise WarmFeaturePrecomputeError(
                    "DINO checkpoint changed while it was being loaded"
                )

            loaded_vae = None
            if args.include_vae:
                from fastwam.models.wan22.helpers.loader import load_wan22_vae_only

                loaded_vae = load_wan22_vae_only(
                    device=args.device,
                    torch_dtype=_torch_dtype(args.dtype),
                    vae_path=args.vae_checkpoint.expanduser().resolve(),
                )
                if sha256_file(Path(loaded_vae.vae_path)) != plan.vae_checkpoint_sha256:
                    raise WarmFeaturePrecomputeError("loaded VAE checkpoint hash mismatch")

            task_to_index = {
                task: index for index, task in enumerate(plan.task_vocabulary)
            }
            action_dim = 7 if args.benchmark_profile == "libero" else 14
            arm_dims = (
                (0, 1, 2, 3, 4, 5)
                if args.benchmark_profile == "libero"
                else (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
            )
            gripper_dims = (6,) if args.benchmark_profile == "libero" else (6, 13)
            if args.benchmark_profile == "robotwin":
                from fastwam.memory.robotwin_artifacts import (
                    robotwin_qpos_action_contract,
                )

                action_contract = robotwin_qpos_action_contract(
                    plan.stats_sha256
                ).to_dict()
            else:
                action_contract = ActionSpaceContract(
                    action_dim=action_dim,
                    arm_dims=arm_dims,
                    gripper_dims=gripper_dims,
                    gripper_threshold=0.0,
                    normalization_mode=normalization_mode,
                    normalization_stats_sha256=plan.stats_sha256,
                    control_mode="libero_delta_eef_axis_angle_plus_gripper",
                    embodiment="libero_panda",
                ).to_dict()
            encoder_contract: dict[str, Any] = {
                "schema": ENCODER_CONTRACT_SCHEMA,
                "version": ENCODER_CONTRACT_VERSION,
                "official_complete": not plan.incomplete_smoke,
                "implementation": {
                    "git_commit": software["git_commit"],
                    "git_dirty": software["git_dirty"],
                },
                "runtime": runtime,
                "data_config_sha256": plan.data_config_sha256,
                "normalization": {
                    "stats_sha256": plan.stats_sha256,
                    "train_stats_manifest_file_sha256": (
                        plan.stats_manifest_file_sha256
                    ),
                    "train_stats_manifest_content_sha256": (
                        plan.stats_manifest_content_sha256
                    ),
                },
                "output_dtype": "float32",
                "compute": {
                    "device": args.device,
                    "dtype": args.dtype,
                    "dino_batch_size": args.dino_batch_size,
                    "vae_batch_size": (
                        args.vae_batch_size if args.include_vae else None
                    ),
                },
                "dino": {
                    "model_id": args.dino_model_id,
                    "revision": args.dino_revision,
                    "checkpoint_tree_sha256": plan.dino_checkpoint_sha256,
                    "checkpoint_file_count": plan.dino_checkpoint_files,
                    "hidden_size": dino.hidden_size,
                    "patch_size": list(dino.patch_size),
                    "patch_grid_size": list(dino.patch_grid_size),
                    "register_token_count": dino.register_token_count,
                    "image_size": list(dino.image_size),
                    "image_mean": list(dino.image_mean),
                    "image_std": list(dino.image_std),
                    "semantic_pool": "adaptive_avg_pool_2x2_row_major",
                    "resize_in_encoder": False,
                },
                "context": {
                    "mode": args.context_mode,
                    "task_vocabulary": list(plan.task_vocabulary),
                    "visual_task_energy": (
                        [1.0, 0.0]
                        if args.context_mode == "visual-only"
                        else [0.5, 0.5]
                    ),
                },
                "vae": {
                    "enabled": bool(args.include_vae),
                    "checkpoint_sha256": plan.vae_checkpoint_sha256,
                    "per_frame_singleton_time": bool(args.include_vae),
                    "spatial_pool": [4, 8] if args.include_vae else None,
                },
                "factual_gripper": (
                    "abs(raw_state_last2).sum"
                    if args.benchmark_profile == "libero"
                    else "abs(raw_state_indices_6_13).sum"
                ),
            }
            camera_contract: dict[str, Any] = {
                "schema": CAMERA_CONTRACT_SCHEMA,
                "version": CAMERA_CONTRACT_VERSION,
                "benchmark_profile": args.benchmark_profile,
                "source_camera_keys": list(plan.camera_keys),
                "processor_camera_mapping": camera_mapping,
                "semantic_camera": plan.semantic_camera,
                "concat_mode": args.concat_mode,
                "decoded_range": [0.0, 1.0],
                "baseline_quantization": "validated_0_1_times_255_to_uint8",
                "per_camera_size": (
                    [224, 224]
                    if args.benchmark_profile == "libero"
                    else [240, 320]
                ),
                **(
                    {}
                    if args.benchmark_profile == "libero"
                    else {
                        "dino_semantic_size": [224, 224],
                        "vae_composite_size": [384, 320],
                    }
                ),
                "vae_model_range": [-1.0, 1.0],
                "timestamp_source": "parquet.timestamp",
                "timestamp_tolerance_s": args.timestamp_tolerance_s,
                "video_backend": args.video_backend,
            }
            normalizer_path = contracts / "normalizer_contract.json"
            encoder_path = contracts / "encoder_contract.json"
            camera_path = contracts / "camera_contract.json"
            _write_json_exclusive(normalizer_path, action_contract)
            _write_json_exclusive(encoder_path, encoder_contract)
            _write_json_exclusive(camera_path, camera_contract)
            contract_hashes = {
                "catalog_hash": plan.catalog.content_sha256,
                "normalizer_hash": sha256_file(normalizer_path),
                "encoder_hash": sha256_file(encoder_path),
                "camera_hash": sha256_file(camera_path),
            }

            paths_by_split: dict[str, list[Path]] = {
                split: [] for split in plan.splits
            }
            first_shapes: dict[str, list[int]] | None = None
            for ordinal, record in enumerate(plan.records, start=1):
                proof = plan.audit.proof_index[
                    (record.dataset_id, record.dataset_index, record.episode_index)
                ]
                full = read_full_lerobot_episode(
                    record,
                    dataset_root=plan.roots[record.dataset_index],
                    audit_proof=proof,
                    camera_keys=plan.camera_keys,
                    timestamp_tolerance_s=args.timestamp_tolerance_s,
                    video_backend=args.video_backend,
                )
                processed = processor_adapter.process(
                    full.actions,
                    full.states,
                    gripper_indices=(
                        None if args.benchmark_profile == "libero" else (6, 13)
                    ),
                )
                processor_images = {
                    camera_mapping[source]: full.images[source]
                    for source in plan.camera_keys
                }
                camera_batch = image_adapter.prepare(processor_images)
                semantic_frames = camera_batch.camera_frames[
                    camera_mapping[plan.semantic_camera]
                ]
                dino_features = dino.encode(
                    semantic_frames,
                    batch_size=args.dino_batch_size,
                )
                vae_features = None
                if loaded_vae is not None:
                    from fastwam.memory.feature_precompute_vae import (
                        encode_wan22_factual_frames,
                    )

                    vae_features = encode_wan22_factual_frames(
                        loaded_vae.vae,
                        camera_batch.vae_frames,
                        device=args.device,
                        batch_size=args.vae_batch_size,
                        torch_dtype=_torch_dtype(args.dtype),
                    )
                task_index = task_to_index[record.primary_task]
                episode = assemble_episode_features(
                    dataset_id=record.dataset_id,
                    dataset_index=record.dataset_index,
                    episode_index=record.episode_index,
                    task_index=task_index,
                    catalog_task_count=len(plan.task_vocabulary),
                    source_episode_sha256=full.source_episode_sha256,
                    model_actions=processed.model_actions,
                    proprio=processed.proprio,
                    gripper=processed.gripper,
                    dino_cls=dino_features.cls,
                    semantic_features=dino_features.spatial,
                    vae_features=vae_features,
                    visual_only_context=args.context_mode == "visual-only",
                )
                payload = (
                    staging
                    / f"dataset_{record.dataset_index:03d}"
                    / f"episode_{record.episode_index:06d}.npz"
                )
                manifest = save_episode_feature_cache(
                    payload,
                    episode,
                    metadata=FeatureCacheMetadata.for_episode(
                        episode,
                        split=record.split,
                        **contract_hashes,
                    ),
                )
                restored = load_episode_feature_cache(payload)
                if restored.manifest != manifest:
                    raise WarmFeaturePrecomputeError(
                        f"feature cache did not round-trip: {payload}"
                    )
                paths_by_split[record.split].append(payload)
                if first_shapes is None:
                    first_shapes = {
                        "model_actions": list(episode.model_actions.shape[1:]),
                        "proprio": list(episode.proprio.shape[1:]),
                        "context_keys": list(episode.context_keys.shape[1:]),
                        "semantic_features": list(episode.semantic_features.shape[1:]),
                        "vae_features": (
                            []
                            if episode.vae_features is None
                            else list(episode.vae_features.shape[1:])
                        ),
                    }
                print(
                    f"encoded={ordinal}/{len(plan.records)} "
                    f"episode={record.dataset_index}:{record.episode_index}"
                )

            for split, paths in paths_by_split.items():
                list_path = staging / f"{split}_features.list"
                lines = [
                    path.relative_to(staging).as_posix() for path in paths
                ]
                _write_exclusive(
                    list_path,
                    ("\n".join(lines) + "\n").encode("utf-8"),
                )

            summary = {
                "schema": SUMMARY_SCHEMA,
                "version": SUMMARY_VERSION,
                "plan": plan.to_dict(args),
                "contracts": {
                    **contract_hashes,
                    "audit_report_sha256": plan.audit.report_sha256,
                    "train_stats_manifest_file_sha256": (
                        plan.stats_manifest_file_sha256
                    ),
                    "train_stats_manifest_content_sha256": (
                        plan.stats_manifest_content_sha256
                    ),
                },
                "feature_shapes_excluding_time": first_shapes,
                "split_counts": {
                    split: len(paths) for split, paths in paths_by_split.items()
                },
                "software": software,
                "official_complete": not plan.incomplete_smoke,
            }
            _write_json_exclusive(staging / "precompute_summary.json", summary)
            os.rename(staging, output)
        except BaseException as exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"Incomplete WARM feature staging directory retained at {staging}"
                )
            raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan = _build_plan(args)
        plan_payload = plan.to_dict(args)
        if args.plan_only:
            print(
                json.dumps(
                    plan_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            return 0
        _run_server_precompute(args, plan)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    print(f"output={plan.output}")
    print(f"episodes={len(plan.records)}")
    print(f"catalog_sha256={plan.catalog.content_sha256}")
    print(f"audit_report_sha256={plan.audit.report_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
