"""Immutable train-only z-score statistics for native-14D RoboTwin data."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .train_stats import TrainEpisodeSource, compute_train_episode_set_sha256


ROBOTWIN_TRAIN_STATS_SCHEMA = "warm.fastwam-robotwin-train-stats"
ROBOTWIN_TRAIN_STATS_VERSION = 1
ROBOTWIN_TRAIN_STATS_RECIPE = "robotwin-global-zscore-all-raw-rows-v1"
ROBOTWIN_TRAIN_STATS_FILENAME = "dataset_stats.json"
ROBOTWIN_TRAIN_STATS_MANIFEST_FILENAME = "train_stats_manifest.json"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class RobotwinTrainStatsError(ValueError):
    pass


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RobotwinTrainStatsError("value is not finite canonical JSON") from exc


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RobotwinTrainStatsError(f"{field} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise RobotwinTrainStatsError(f"{field} must be a positive integer")
    return int(value)


def _vector(value: object, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise RobotwinTrainStatsError(f"{field} must be a non-empty vector")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise RobotwinTrainStatsError(f"{field} must contain numbers")
        number = float(item)
        if not math.isfinite(number):
            raise RobotwinTrainStatsError(f"{field} must contain finite numbers")
        result.append(number)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FastWAMRobotwinTrainStats:
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]
    state_mean: tuple[float, ...]
    state_std: tuple[float, ...]
    num_episodes: int
    num_transition: int

    def __post_init__(self) -> None:
        for field in ("action_mean", "action_std", "state_mean", "state_std"):
            object.__setattr__(self, field, _vector(getattr(self, field), field))
        if len(self.action_mean) != 14 or len(self.action_std) != 14:
            raise RobotwinTrainStatsError("RoboTwin action statistics must be 14D")
        if len(self.state_mean) != 14 or len(self.state_std) != 14:
            raise RobotwinTrainStatsError("RoboTwin state statistics must be 14D")
        if any(value < 0 for value in (*self.action_std, *self.state_std)):
            raise RobotwinTrainStatsError("standard deviations must be non-negative")
        object.__setattr__(
            self, "num_episodes", _positive_int(self.num_episodes, "num_episodes")
        )
        object.__setattr__(
            self,
            "num_transition",
            _positive_int(self.num_transition, "num_transition"),
        )
        if self.num_transition < self.num_episodes:
            raise RobotwinTrainStatsError("row count cannot be smaller than episodes")

    @property
    def action_dim(self) -> int:
        return 14

    @property
    def state_dim(self) -> int:
        return 14

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": {
                "default": {
                    "global_mean": list(self.action_mean),
                    "global_std": list(self.action_std),
                }
            },
            "state": {
                "default": {
                    "global_mean": list(self.state_mean),
                    "global_std": list(self.state_std),
                }
            },
            "num_episodes": self.num_episodes,
            "num_transition": self.num_transition,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FastWAMRobotwinTrainStats":
        if not isinstance(value, Mapping) or set(value) != {
            "action",
            "state",
            "num_episodes",
            "num_transition",
        }:
            raise RobotwinTrainStatsError("invalid RoboTwin stats fields")

        def pair(kind: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
            outer = value[kind]
            if not isinstance(outer, Mapping) or set(outer) != {"default"}:
                raise RobotwinTrainStatsError(f"invalid {kind} stats")
            row = outer["default"]
            if not isinstance(row, Mapping) or set(row) != {
                "global_mean",
                "global_std",
            }:
                raise RobotwinTrainStatsError(f"invalid {kind}.default stats")
            return _vector(row["global_mean"], f"{kind}_mean"), _vector(
                row["global_std"], f"{kind}_std"
            )

        action_mean, action_std = pair("action")
        state_mean, state_std = pair("state")
        return cls(
            action_mean,
            action_std,
            state_mean,
            state_std,
            value["num_episodes"],
            value["num_transition"],
        )


@dataclass(frozen=True, slots=True)
class RobotwinTrainStatsManifest:
    stats_file_sha256: str
    catalog_sha256: str
    audit_report_sha256: str
    data_config_sha256: str
    train_episode_set_sha256: str
    train_episode_count: int
    train_frame_count: int
    git_commit: str
    dataset_revision: str
    schema: str = ROBOTWIN_TRAIN_STATS_SCHEMA
    version: int = ROBOTWIN_TRAIN_STATS_VERSION
    recipe: str = ROBOTWIN_TRAIN_STATS_RECIPE
    action_dim: int = 14
    state_dim: int = 14
    stats_filename: str = ROBOTWIN_TRAIN_STATS_FILENAME

    def __post_init__(self) -> None:
        if self.schema != ROBOTWIN_TRAIN_STATS_SCHEMA or self.version != 1:
            raise RobotwinTrainStatsError("unsupported RoboTwin stats manifest")
        if self.recipe != ROBOTWIN_TRAIN_STATS_RECIPE:
            raise RobotwinTrainStatsError("unsupported RoboTwin stats recipe")
        for field in (
            "stats_file_sha256",
            "catalog_sha256",
            "audit_report_sha256",
            "data_config_sha256",
            "train_episode_set_sha256",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        for field in ("train_episode_count", "train_frame_count"):
            object.__setattr__(self, field, _positive_int(getattr(self, field), field))
        if self.action_dim != 14 or self.state_dim != 14:
            raise RobotwinTrainStatsError("RoboTwin manifest dimensions must be 14/14")
        if not isinstance(self.git_commit, str) or _GIT_COMMIT.fullmatch(
            self.git_commit
        ) is None:
            raise RobotwinTrainStatsError("git_commit must be a 40-character SHA")
        if (
            not isinstance(self.dataset_revision, str)
            or not self.dataset_revision
            or self.dataset_revision.strip() != self.dataset_revision
        ):
            raise RobotwinTrainStatsError("dataset_revision must be normalized")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "recipe": self.recipe,
            "stats_filename": self.stats_filename,
            "stats_file_sha256": self.stats_file_sha256,
            "catalog_sha256": self.catalog_sha256,
            "audit_report_sha256": self.audit_report_sha256,
            "data_config_sha256": self.data_config_sha256,
            "train_episode_set_sha256": self.train_episode_set_sha256,
            "train_episode_count": self.train_episode_count,
            "train_frame_count": self.train_frame_count,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "git_commit": self.git_commit,
            "dataset_revision": self.dataset_revision,
        }

    @property
    def content_sha256(self) -> str:
        return sha256(_canonical(self.to_payload())).hexdigest()

    def to_document(self) -> dict[str, Any]:
        return {**self.to_payload(), "content_sha256": self.content_sha256}

    @classmethod
    def from_document(
        cls, value: Mapping[str, Any]
    ) -> "RobotwinTrainStatsManifest":
        if not isinstance(value, Mapping):
            raise RobotwinTrainStatsError("manifest must be an object")
        expected = {
            "schema",
            "version",
            "recipe",
            "stats_filename",
            "stats_file_sha256",
            "catalog_sha256",
            "audit_report_sha256",
            "data_config_sha256",
            "train_episode_set_sha256",
            "train_episode_count",
            "train_frame_count",
            "action_dim",
            "state_dim",
            "git_commit",
            "dataset_revision",
            "content_sha256",
        }
        if set(value) != expected:
            raise RobotwinTrainStatsError("invalid RoboTwin manifest fields")
        expected_hash = _digest(value["content_sha256"], "content_sha256")
        payload = {key: value[key] for key in expected if key != "content_sha256"}
        if sha256(_canonical(payload)).hexdigest() != expected_hash:
            raise RobotwinTrainStatsError("manifest content hash mismatch")
        result = cls(**payload)
        if result.content_sha256 != expected_hash:
            raise RobotwinTrainStatsError("manifest did not round-trip")
        return result


@dataclass(frozen=True, slots=True)
class LoadedRobotwinTrainStatsArtifact:
    stats: FastWAMRobotwinTrainStats
    manifest: RobotwinTrainStatsManifest


def encode_robotwin_stats(value: FastWAMRobotwinTrainStats) -> bytes:
    return (json.dumps(value.to_dict(), sort_keys=True, indent=2) + "\n").encode()


def encode_robotwin_manifest(value: RobotwinTrainStatsManifest) -> bytes:
    return (json.dumps(value.to_document(), sort_keys=True, indent=2) + "\n").encode()


def load_robotwin_train_stats_artifact(
    directory: str | Path,
    *,
    expected_catalog_sha256: str | None = None,
    expected_audit_report_sha256: str | None = None,
    expected_data_config_sha256: str | None = None,
    expected_train_episodes: Sequence[TrainEpisodeSource] | None = None,
) -> LoadedRobotwinTrainStatsArtifact:
    root = Path(directory)
    expected_files = {
        ROBOTWIN_TRAIN_STATS_FILENAME,
        ROBOTWIN_TRAIN_STATS_MANIFEST_FILENAME,
    }
    if not root.is_dir() or {item.name for item in root.iterdir()} != expected_files:
        raise RobotwinTrainStatsError(
            "RoboTwin stats artifact must contain exactly stats and manifest"
        )
    stats_raw = (root / ROBOTWIN_TRAIN_STATS_FILENAME).read_bytes()
    stats = FastWAMRobotwinTrainStats.from_dict(json.loads(stats_raw))
    manifest = RobotwinTrainStatsManifest.from_document(
        json.loads((root / ROBOTWIN_TRAIN_STATS_MANIFEST_FILENAME).read_bytes())
    )
    if sha256(stats_raw).hexdigest() != manifest.stats_file_sha256:
        raise RobotwinTrainStatsError("stats bytes do not match manifest")
    if stats.num_episodes != manifest.train_episode_count:
        raise RobotwinTrainStatsError("episode count disagrees with manifest")
    if stats.num_transition != manifest.train_frame_count:
        raise RobotwinTrainStatsError("frame count disagrees with manifest")
    for field, expected in (
        ("catalog_sha256", expected_catalog_sha256),
        ("audit_report_sha256", expected_audit_report_sha256),
        ("data_config_sha256", expected_data_config_sha256),
    ):
        if expected is not None and getattr(manifest, field) != _digest(expected, field):
            raise RobotwinTrainStatsError(f"manifest {field} mismatch")
    if expected_train_episodes is not None:
        items = tuple(expected_train_episodes)
        if manifest.train_episode_set_sha256 != compute_train_episode_set_sha256(items):
            raise RobotwinTrainStatsError("train episode set mismatch")
        if manifest.train_episode_count != len(items):
            raise RobotwinTrainStatsError("train episode count mismatch")
    return LoadedRobotwinTrainStatsArtifact(stats, manifest)


__all__ = [
    "FastWAMRobotwinTrainStats",
    "LoadedRobotwinTrainStatsArtifact",
    "ROBOTWIN_TRAIN_STATS_FILENAME",
    "ROBOTWIN_TRAIN_STATS_MANIFEST_FILENAME",
    "ROBOTWIN_TRAIN_STATS_RECIPE",
    "ROBOTWIN_TRAIN_STATS_SCHEMA",
    "RobotwinTrainStatsError",
    "RobotwinTrainStatsManifest",
    "encode_robotwin_manifest",
    "encode_robotwin_stats",
    "load_robotwin_train_stats_artifact",
]
