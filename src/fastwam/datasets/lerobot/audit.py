"""Reproducible metadata and source-bundle audits for WARM LeRobot data.

An episode's WARM source identity binds the parquet table and every requested
camera MP4 in an explicit order.  This prevents a feature cache from claiming
an audited parquet while silently decoding unaudited or replaced videos.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from statistics import mean
from string import Formatter
from typing import Any, Iterable, Sequence
from uuid import uuid4

from .episode_catalog import EpisodeCatalog


AUDIT_SCHEMA = "warm.lerobot-audit"
AUDIT_VERSION = 2
DEFAULT_AUDITED_CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VIDEO_TEMPLATE_FIELDS = {"episode_chunk", "video_key", "episode_index"}


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class CameraVideoAuditProof:
    """One ordered camera MP4 bound to its byte content."""

    camera_key: str
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.camera_key, str)
            or not self.camera_key
            or self.camera_key.strip() != self.camera_key
        ):
            raise ValueError("camera_key must be a non-empty normalized string")
        _require_sha256(self.sha256, label="camera video sha256")

    def as_dict(self) -> dict[str, str]:
        return {"camera_key": self.camera_key, "sha256": self.sha256}


def compute_camera_bundle_sha256(
    ordered_camera_sha256: Sequence[CameraVideoAuditProof],
) -> str | None:
    """Hash an ordered camera manifest, or return ``None`` for no cameras."""

    cameras = tuple(ordered_camera_sha256)
    if not cameras:
        return None
    if len({item.camera_key for item in cameras}) != len(cameras):
        raise ValueError("ordered camera hashes contain duplicate camera keys")
    return _canonical_hash(
        {
            "schema": "warm.camera-video-bundle",
            "version": 1,
            "cameras": [item.as_dict() for item in cameras],
        }
    )


def compute_source_bundle_sha256(
    table_sha256: str,
    ordered_camera_sha256: Sequence[CameraVideoAuditProof],
) -> str:
    """Return the immutable identity of one complete factual episode source.

    Camera-free synthetic fixtures deliberately retain the old identity
    ``bundle == table``.  Production proofs with cameras are domain-separated
    canonical hashes over the table digest and ordered camera digests.
    """

    table_digest = _require_sha256(table_sha256, label="table sha256")
    cameras = tuple(ordered_camera_sha256)
    if not cameras:
        return table_digest
    # Also validates unique keys.
    compute_camera_bundle_sha256(cameras)
    return _canonical_hash(
        {
            "schema": "warm.episode-source-bundle",
            "version": 1,
            "table_sha256": table_digest,
            "cameras": [item.as_dict() for item in cameras],
        }
    )


@dataclass(frozen=True, slots=True)
class EpisodeAuditProof:
    """One catalog episode bound to table bytes and ordered camera videos."""

    dataset_id: str
    dataset_index: int
    episode_index: int
    split: str
    source_episode_sha256: str
    table_sha256: str | None = None
    ordered_camera_sha256: tuple[CameraVideoAuditProof, ...] = ()
    camera_bundle_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("audit proof dataset_id must be non-empty")
        for field in ("dataset_index", "episode_index"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"audit proof {field} must be a non-negative integer")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("audit proof split must be train, dev, or test")

        source_digest = _require_sha256(
            self.source_episode_sha256,
            label="audit proof source bundle sha256",
        )
        table_digest = self.table_sha256
        if table_digest is None:
            # Compatibility for camera-free synthetic unit fixtures only.
            table_digest = source_digest
            object.__setattr__(self, "table_sha256", table_digest)
        table_digest = _require_sha256(table_digest, label="audit proof table sha256")

        cameras = tuple(self.ordered_camera_sha256)
        if any(not isinstance(item, CameraVideoAuditProof) for item in cameras):
            raise TypeError("ordered_camera_sha256 must contain CameraVideoAuditProof")
        if len({item.camera_key for item in cameras}) != len(cameras):
            raise ValueError("audit proof contains duplicate camera keys")
        object.__setattr__(self, "ordered_camera_sha256", cameras)

        expected_camera_bundle = compute_camera_bundle_sha256(cameras)
        if self.camera_bundle_sha256 is None:
            object.__setattr__(self, "camera_bundle_sha256", expected_camera_bundle)
        elif self.camera_bundle_sha256 != expected_camera_bundle:
            raise ValueError("audit proof camera bundle hash does not match camera hashes")
        if self.camera_bundle_sha256 is not None:
            _require_sha256(
                self.camera_bundle_sha256,
                label="audit proof camera bundle sha256",
            )

        expected_source = compute_source_bundle_sha256(table_digest, cameras)
        if source_digest != expected_source:
            raise ValueError("audit proof source bundle hash does not match its components")

    @property
    def episode_key(self) -> tuple[str, int, int]:
        return self.dataset_id, self.dataset_index, self.episode_index

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return tuple(item.camera_key for item in self.ordered_camera_sha256)


@dataclass(frozen=True, slots=True)
class LerobotAuditReport:
    """Verified immutable subset of a WARM LeRobot audit report."""

    catalog_sha256: str
    report_sha256: str
    episode_tables_hashed: bool
    audited_camera_keys: tuple[str, ...]
    cross_split_duplicate_count: int
    cross_split_table_duplicate_count: int
    cross_split_video_duplicate_count: int
    cross_split_camera_bundle_duplicate_count: int
    cross_split_source_bundle_duplicate_count: int
    episode_proofs: tuple[EpisodeAuditProof, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.catalog_sha256, label="audit catalog_sha256")
        _require_sha256(self.report_sha256, label="audit report_sha256")
        if not isinstance(self.episode_tables_hashed, bool):
            raise TypeError("episode_tables_hashed must be bool")
        cameras = tuple(self.audited_camera_keys)
        if any(not isinstance(key, str) or not key for key in cameras):
            raise ValueError("audited_camera_keys must contain non-empty strings")
        if len(set(cameras)) != len(cameras):
            raise ValueError("audited_camera_keys must be unique")
        object.__setattr__(self, "audited_camera_keys", cameras)
        count_fields = (
            "cross_split_duplicate_count",
            "cross_split_table_duplicate_count",
            "cross_split_video_duplicate_count",
            "cross_split_camera_bundle_duplicate_count",
            "cross_split_source_bundle_duplicate_count",
        )
        for field in count_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be non-negative")
        keys = [proof.episode_key for proof in self.episode_proofs]
        if len(keys) != len(set(keys)):
            raise ValueError("audit report contains duplicate episode proofs")
        if self.episode_tables_hashed:
            for proof in self.episode_proofs:
                if proof.camera_keys != cameras:
                    raise ValueError(
                        "every audit proof must use the report's exact ordered cameras"
                    )
        elif self.episode_proofs:
            raise ValueError("unhashed audit report must not contain episode proofs")

    @property
    def proof_index(self) -> dict[tuple[str, int, int], EpisodeAuditProof]:
        return {proof.episode_key: proof for proof in self.episode_proofs}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _validate_camera_keys(camera_keys: Sequence[str]) -> tuple[str, ...]:
    if isinstance(camera_keys, (str, bytes)):
        raise TypeError("camera_keys must be a sequence of camera feature names")
    cameras = tuple(camera_keys)
    if any(
        not isinstance(key, str) or not key or key.strip() != key for key in cameras
    ):
        raise ValueError("camera keys must be non-empty normalized strings")
    if len(set(cameras)) != len(cameras):
        raise ValueError("camera keys must be unique")
    return cameras


def _validate_video_template(template: object) -> str:
    if not isinstance(template, str) or not template or template.strip() != template:
        raise ValueError("LeRobot info.json must declare a normalized video_path")
    try:
        parsed = list(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError("invalid LeRobot video_path template") from exc
    fields = {field for _, field, _, _ in parsed if field is not None}
    if not fields.issubset(_VIDEO_TEMPLATE_FIELDS):
        raise ValueError(
            "LeRobot video_path template contains unsupported fields: "
            f"{sorted(fields.difference(_VIDEO_TEMPLATE_FIELDS))}"
        )
    for _, field, format_spec, conversion in parsed:
        if field is not None and conversion is not None:
            raise ValueError("LeRobot video_path template conversions are forbidden")
        if field is not None and format_spec not in {"", "03d", "06d"}:
            raise ValueError("LeRobot video_path template uses an unsupported format")
    if not {"video_key", "episode_index"}.issubset(fields):
        raise ValueError(
            "LeRobot video_path template must include video_key and episode_index"
        )
    return template


def resolve_episode_video_paths(
    dataset_root: str | Path,
    *,
    episode_index: int,
    camera_keys: Sequence[str],
    info: dict[str, Any] | None = None,
) -> tuple[tuple[str, Path], ...]:
    """Resolve an ordered camera list through the audited LeRobot layout."""

    root = Path(dataset_root).resolve()
    cameras = _validate_camera_keys(camera_keys)
    if info is None:
        info_path = _resolve_inside(root, "meta/info.json", label="metadata path")
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read LeRobot metadata at {info_path}") from exc
    if not isinstance(info, dict):
        raise ValueError("LeRobot info.json must contain an object")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("LeRobot info.features must be an object")
    missing = [key for key in cameras if key not in features]
    if missing:
        raise ValueError(f"audited camera keys are absent from info.features: {missing}")
    for key in cameras:
        spec = features[key]
        if not isinstance(spec, dict) or str(spec.get("dtype", "")) not in {
            "image",
            "video",
        }:
            raise ValueError(f"camera feature {key!r} must declare image/video dtype")
    chunks_size = info.get("chunks_size", 1000)
    if isinstance(chunks_size, bool) or not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError("LeRobot chunks_size must be a positive integer")
    template = _validate_video_template(info.get("video_path"))

    result: list[tuple[str, Path]] = []
    for camera in cameras:
        try:
            relative = template.format(
                episode_chunk=episode_index // chunks_size,
                video_key=camera,
                episode_index=episode_index,
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"unsupported LeRobot video_path template: {template}") from exc
        if not relative:
            raise ValueError("LeRobot video_path template rendered an empty path")
        result.append(
            (
                camera,
                _resolve_inside(root, relative, label=f"video path for camera {camera!r}"),
            )
        )
    return tuple(result)


def _feature_groups(features: dict[str, Any]) -> dict[str, list[str]]:
    names = sorted(str(name) for name in features)
    cameras: list[str] = []
    actions: list[str] = []
    states: list[str] = []
    outcome: list[str] = []
    for name in names:
        spec = features.get(name, {})
        dtype = str(spec.get("dtype", "")) if isinstance(spec, dict) else ""
        lower = name.lower()
        if dtype in {"image", "video"} or "observation.images" in lower:
            cameras.append(name)
        if lower == "action" or lower.startswith("action."):
            actions.append(name)
        if lower == "observation.state" or lower.startswith("observation.state."):
            states.append(name)
        if any(token in lower for token in ("success", "reward", "done", "terminal")):
            outcome.append(name)
    return {
        "camera_fields": cameras,
        "action_fields": actions,
        "state_fields": states,
        "outcome_fields": outcome,
    }


def _episode_ref(proof: EpisodeAuditProof, *, camera_key: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dataset_id": proof.dataset_id,
        "dataset_index": proof.dataset_index,
        "episode_index": proof.episode_index,
        "split": proof.split,
    }
    if camera_key is not None:
        result["camera_key"] = camera_key
    return result


def _duplicate_groups(
    rows: Iterable[tuple[str, EpisodeAuditProof, str | None]],
) -> list[dict[str, Any]]:
    by_hash: dict[str, list[tuple[EpisodeAuditProof, str | None]]] = defaultdict(list)
    for digest, proof, camera_key in rows:
        by_hash[digest].append((proof, camera_key))
    groups: list[dict[str, Any]] = []
    for digest, members in sorted(by_hash.items()):
        if len(members) <= 1 or len({proof.split for proof, _ in members}) <= 1:
            continue
        groups.append(
            {
                "sha256": digest,
                "episodes": [
                    _episode_ref(proof, camera_key=camera_key)
                    for proof, camera_key in sorted(
                        members,
                        key=lambda item: (
                            item[0].dataset_index,
                            item[0].episode_index,
                            item[1] or "",
                        ),
                    )
                ],
            }
        )
    return groups


def _recompute_duplicate_groups(
    proofs: Sequence[EpisodeAuditProof],
) -> dict[str, list[dict[str, Any]]]:
    tables = _duplicate_groups((proof.table_sha256, proof, None) for proof in proofs)
    videos = _duplicate_groups(
        (camera.sha256, proof, camera.camera_key)
        for proof in proofs
        for camera in proof.ordered_camera_sha256
    )
    camera_bundles = _duplicate_groups(
        (proof.camera_bundle_sha256, proof, None)
        for proof in proofs
        if proof.camera_bundle_sha256 is not None
    )
    source_bundles = _duplicate_groups(
        (proof.source_episode_sha256, proof, None) for proof in proofs
    )
    return {
        "cross_split_table_duplicates": tables,
        "cross_split_video_duplicates": videos,
        "cross_split_camera_bundle_duplicates": camera_bundles,
        "cross_split_source_bundle_duplicates": source_bundles,
    }


def _proof_as_row(proof: EpisodeAuditProof) -> dict[str, Any]:
    return {
        "dataset_id": proof.dataset_id,
        "dataset_index": proof.dataset_index,
        "episode_index": proof.episode_index,
        "split": proof.split,
        "table_sha256": proof.table_sha256,
        "ordered_camera_sha256": [
            item.as_dict() for item in proof.ordered_camera_sha256
        ],
        "camera_bundle_sha256": proof.camera_bundle_sha256,
        "source_episode_sha256": proof.source_episode_sha256,
    }


def audit_lerobot_catalog(
    catalog: EpisodeCatalog,
    dataset_roots: Sequence[str | Path],
    *,
    hash_episode_tables: bool = False,
    camera_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Audit metadata, exact splits, and complete factual episode sources."""

    cameras = _validate_camera_keys(camera_keys)
    if cameras and not hash_episode_tables:
        raise ValueError("camera source auditing requires hash_episode_tables=True")
    descriptors = sorted(catalog.datasets, key=lambda item: item.dataset_index)
    if len(dataset_roots) != len(descriptors):
        raise ValueError("dataset_roots must match catalog dataset count and order")

    dataset_reports: list[dict[str, Any]] = []
    proofs: list[EpisodeAuditProof] = []
    all_global_ids: set[tuple[int, int]] = set()

    for descriptor, root_value in zip(descriptors, dataset_roots, strict=True):
        root = Path(root_value).resolve()
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(info_path)
        info_snapshot = info_path.read_bytes()
        if sha256(info_snapshot).hexdigest() != descriptor.info_sha256:
            raise ValueError(f"Dataset metadata changed after cataloging: {descriptor.dataset_id}")
        try:
            info = json.loads(info_snapshot)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid LeRobot metadata at {info_path}") from exc
        features = info.get("features", {}) if isinstance(info, dict) else None
        if not isinstance(features, dict):
            raise ValueError(f"{descriptor.dataset_id}: info.features must be an object")
        if cameras:
            # Validate feature declarations and the template before hashing any episode.
            resolve_episode_video_paths(
                root,
                episode_index=0,
                camera_keys=cameras,
                info=info,
            )

        episodes = [
            item for item in catalog.episodes if item.dataset_index == descriptor.dataset_index
        ]
        split_counts = Counter(item.split for item in episodes)
        task_counts = Counter(item.primary_task for item in episodes)
        lengths = [item.length for item in episodes]
        ids = {item.global_episode_id for item in episodes}
        if all_global_ids.intersection(ids):
            raise ValueError("Global episode identities overlap across configured datasets")
        all_global_ids.update(ids)

        dataset_reports.append(
            {
                "dataset_id": descriptor.dataset_id,
                "dataset_index": descriptor.dataset_index,
                "fps": descriptor.fps,
                "total_episodes": descriptor.total_episodes,
                "total_frames_declared": info.get("total_frames"),
                "total_tasks_declared": info.get("total_tasks"),
                "episode_length": {
                    "min": min(lengths),
                    "max": max(lengths),
                    "mean": mean(lengths),
                },
                "split_counts": dict(sorted(split_counts.items())),
                "task_counts": dict(sorted(task_counts.items())),
                "metadata_hashes": {
                    "info_sha256": descriptor.info_sha256,
                    "episodes_sha256": descriptor.episodes_sha256,
                },
                **_feature_groups(features),
            }
        )

        if hash_episode_tables:
            for episode in episodes:
                table_path = _resolve_inside(
                    root,
                    episode.data_relpath,
                    label="episode data path",
                )
                if not table_path.is_file():
                    raise FileNotFoundError(table_path)
                table_digest = _sha256_file(table_path)
                camera_proofs: list[CameraVideoAuditProof] = []
                for camera, video_path in resolve_episode_video_paths(
                    root,
                    episode_index=episode.episode_index,
                    camera_keys=cameras,
                    info=info,
                ):
                    if not video_path.is_file():
                        raise FileNotFoundError(video_path)
                    camera_proofs.append(
                        CameraVideoAuditProof(camera, _sha256_file(video_path))
                    )
                ordered_cameras = tuple(camera_proofs)
                proofs.append(
                    EpisodeAuditProof(
                        dataset_id=episode.dataset_id,
                        dataset_index=episode.dataset_index,
                        episode_index=episode.episode_index,
                        split=episode.split,
                        table_sha256=table_digest,
                        ordered_camera_sha256=ordered_cameras,
                        camera_bundle_sha256=compute_camera_bundle_sha256(
                            ordered_cameras
                        ),
                        source_episode_sha256=compute_source_bundle_sha256(
                            table_digest,
                            ordered_cameras,
                        ),
                    )
                )

    proofs.sort(key=lambda proof: (proof.dataset_index, proof.episode_index))
    duplicate_groups = _recompute_duplicate_groups(proofs)
    duplicate_counts = {key: len(value) for key, value in duplicate_groups.items()}
    duplicate_total = sum(duplicate_counts.values())

    report: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "catalog_sha256": catalog.content_sha256,
        "audited_camera_keys": list(cameras),
        "datasets": dataset_reports,
        "summary": {
            "dataset_count": len(descriptors),
            "episode_count": len(catalog.episodes),
            "task_count": len(
                {(item.dataset_id, item.primary_task) for item in catalog.episodes}
            ),
            "split_counts": dict(
                sorted(Counter(item.split for item in catalog.episodes).items())
            ),
            "episode_tables_hashed": bool(hash_episode_tables),
            "cross_split_duplicate_count": duplicate_total,
            "cross_split_table_duplicate_count": duplicate_counts[
                "cross_split_table_duplicates"
            ],
            "cross_split_video_duplicate_count": duplicate_counts[
                "cross_split_video_duplicates"
            ],
            "cross_split_camera_bundle_duplicate_count": duplicate_counts[
                "cross_split_camera_bundle_duplicates"
            ],
            "cross_split_source_bundle_duplicate_count": duplicate_counts[
                "cross_split_source_bundle_duplicates"
            ],
        },
        **duplicate_groups,
        "episode_source_hashes": [_proof_as_row(proof) for proof in proofs],
    }
    report["report_sha256"] = _canonical_hash(report)
    return report


def write_audit_report(report: dict[str, Any], path: str | Path) -> None:
    expected = report.get("report_sha256")
    if not isinstance(expected, str):
        raise ValueError("audit report is missing report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256")
    if _canonical_hash(unhashed) != expected:
        raise ValueError("audit report content does not match report_sha256")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"audit report already exists at {target}")
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _require_summary_count(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"audit summary has invalid {key}")
    return value


def load_audit_report(path: str | Path) -> LerobotAuditReport:
    """Load, hash-check, and semantically re-derive a source audit report."""

    source = Path(path)
    try:
        snapshot = source.read_bytes()
        document = json.loads(snapshot)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read audit report {source}") from exc
    if not isinstance(document, dict):
        raise ValueError("audit report must be a JSON object")
    expected_hash = document.get("report_sha256")
    _require_sha256(expected_hash, label="audit report_sha256")
    unhashed = dict(document)
    unhashed.pop("report_sha256")
    if _canonical_hash(unhashed) != expected_hash:
        raise ValueError("audit report content does not match report_sha256")
    if document.get("schema") != AUDIT_SCHEMA or document.get("version") != AUDIT_VERSION:
        raise ValueError("unsupported WARM LeRobot audit schema")

    summary = document.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("audit report summary must be an object")
    episode_tables_hashed = summary.get("episode_tables_hashed")
    if not isinstance(episode_tables_hashed, bool):
        raise ValueError("audit summary must declare episode_tables_hashed")
    episode_count = _require_summary_count(summary, "episode_count")
    claimed_counts = {
        "cross_split_duplicate_count": _require_summary_count(
            summary, "cross_split_duplicate_count"
        ),
        "cross_split_table_duplicate_count": _require_summary_count(
            summary, "cross_split_table_duplicate_count"
        ),
        "cross_split_video_duplicate_count": _require_summary_count(
            summary, "cross_split_video_duplicate_count"
        ),
        "cross_split_camera_bundle_duplicate_count": _require_summary_count(
            summary, "cross_split_camera_bundle_duplicate_count"
        ),
        "cross_split_source_bundle_duplicate_count": _require_summary_count(
            summary, "cross_split_source_bundle_duplicate_count"
        ),
    }

    cameras_raw = document.get("audited_camera_keys")
    if not isinstance(cameras_raw, list):
        raise ValueError("audit audited_camera_keys must be a list")
    cameras = _validate_camera_keys(cameras_raw)
    rows = document.get("episode_source_hashes")
    if not isinstance(rows, list):
        raise ValueError("audit episode_source_hashes must be a list")
    expected_fields = {
        "dataset_id",
        "dataset_index",
        "episode_index",
        "split",
        "table_sha256",
        "ordered_camera_sha256",
        "camera_bundle_sha256",
        "source_episode_sha256",
    }
    proofs: list[EpisodeAuditProof] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError(f"audit episode_source_hashes[{position}] has invalid fields")
        camera_rows = row["ordered_camera_sha256"]
        if not isinstance(camera_rows, list):
            raise ValueError(
                f"audit episode_source_hashes[{position}] camera hashes must be a list"
            )
        ordered: list[CameraVideoAuditProof] = []
        for camera_position, camera_row in enumerate(camera_rows):
            if not isinstance(camera_row, dict) or set(camera_row) != {
                "camera_key",
                "sha256",
            }:
                raise ValueError(
                    "audit camera proof has invalid fields at "
                    f"episode {position}, camera {camera_position}"
                )
            ordered.append(CameraVideoAuditProof(**camera_row))
        proofs.append(
            EpisodeAuditProof(
                dataset_id=row["dataset_id"],
                dataset_index=row["dataset_index"],
                episode_index=row["episode_index"],
                split=row["split"],
                table_sha256=row["table_sha256"],
                ordered_camera_sha256=tuple(ordered),
                camera_bundle_sha256=row["camera_bundle_sha256"],
                source_episode_sha256=row["source_episode_sha256"],
            )
        )
    if episode_tables_hashed and len(proofs) != episode_count:
        raise ValueError("audit proof count does not match the catalog episode count")
    if not episode_tables_hashed and proofs:
        raise ValueError("unhashed audit report must not contain episode proofs")
    if episode_tables_hashed and any(proof.camera_keys != cameras for proof in proofs):
        raise ValueError("audit proof cameras do not exactly match audited_camera_keys")

    recomputed = _recompute_duplicate_groups(proofs)
    recomputed_counts = {
        "cross_split_table_duplicate_count": len(
            recomputed["cross_split_table_duplicates"]
        ),
        "cross_split_video_duplicate_count": len(
            recomputed["cross_split_video_duplicates"]
        ),
        "cross_split_camera_bundle_duplicate_count": len(
            recomputed["cross_split_camera_bundle_duplicates"]
        ),
        "cross_split_source_bundle_duplicate_count": len(
            recomputed["cross_split_source_bundle_duplicates"]
        ),
    }
    recomputed_total = sum(recomputed_counts.values())
    if claimed_counts["cross_split_duplicate_count"] != recomputed_total:
        raise ValueError("audit summary duplicate total disagrees with episode proofs")
    for key, value in recomputed_counts.items():
        if claimed_counts[key] != value:
            raise ValueError(f"audit summary {key} disagrees with episode proofs")
    for key, expected_groups in recomputed.items():
        if document.get(key) != expected_groups:
            raise ValueError(f"audit {key} disagrees with episode proofs")

    return LerobotAuditReport(
        catalog_sha256=document.get("catalog_sha256"),
        report_sha256=expected_hash,
        episode_tables_hashed=episode_tables_hashed,
        audited_camera_keys=cameras,
        cross_split_duplicate_count=recomputed_total,
        cross_split_table_duplicate_count=recomputed_counts[
            "cross_split_table_duplicate_count"
        ],
        cross_split_video_duplicate_count=recomputed_counts[
            "cross_split_video_duplicate_count"
        ],
        cross_split_camera_bundle_duplicate_count=recomputed_counts[
            "cross_split_camera_bundle_duplicate_count"
        ],
        cross_split_source_bundle_duplicate_count=recomputed_counts[
            "cross_split_source_bundle_duplicate_count"
        ],
        episode_proofs=tuple(proofs),
    )


__all__ = [
    "AUDIT_SCHEMA",
    "AUDIT_VERSION",
    "DEFAULT_AUDITED_CAMERA_KEYS",
    "CameraVideoAuditProof",
    "EpisodeAuditProof",
    "LerobotAuditReport",
    "audit_lerobot_catalog",
    "compute_camera_bundle_sha256",
    "compute_source_bundle_sha256",
    "load_audit_report",
    "resolve_episode_video_paths",
    "write_audit_report",
]
