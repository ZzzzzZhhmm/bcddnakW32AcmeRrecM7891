"""Strict train-only normalization-statistics contracts for WARM.

FastWAM's LIBERO processor consumes global per-dimension minima and maxima,
but the upstream JSON does not record which episodes produced those numbers.
This module adds an immutable manifest around the directly compatible stats
JSON.  It deliberately depends only on the Python standard library so data
provenance can be inspected without importing Torch, PyArrow, or Hydra.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


TRAIN_STATS_SCHEMA = "warm.fastwam-libero-train-stats"
TRAIN_STATS_VERSION = 1
TRAIN_STATS_RECIPE = "libero-global-min-max-all-raw-rows-v1"
TRAIN_STATS_FILENAME = "dataset_stats.json"
TRAIN_STATS_MANIFEST_FILENAME = "train_stats_manifest.json"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_STATS_TOP_FIELDS = frozenset(
    {"action", "state", "num_episodes", "num_transition"}
)
_STATS_KIND_FIELDS = frozenset({"default"})
_STATS_RANGE_FIELDS = frozenset({"global_min", "global_max"})
_MANIFEST_PAYLOAD_FIELDS = frozenset(
    {
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
    }
)
_MANIFEST_DOCUMENT_FIELDS = _MANIFEST_PAYLOAD_FIELDS | {"content_sha256"}


class TrainStatsContractError(ValueError):
    """Raised when train statistics or their provenance are ambiguous."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrainStatsContractError("value is not finite canonical JSON") from exc


def _pretty_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrainStatsContractError("value is not finite JSON") from exc


def _read_json_snapshot(path: str | Path, *, label: str) -> tuple[bytes, object]:
    source = Path(path)
    try:
        snapshot = source.read_bytes()
        def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
            output: dict[str, object] = {}
            for key, item in pairs:
                if key in output:
                    raise TrainStatsContractError(
                        f"{label} contains duplicate JSON key {key!r}"
                    )
                output[key] = item
            return output

        value = json.loads(
            snapshot.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                TrainStatsContractError(
                    f"{label} contains forbidden JSON constant {token}"
                )
            ),
            object_pairs_hook=reject_duplicate_keys,
        )
    except TrainStatsContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainStatsContractError(f"cannot read {label} at {source}") from exc
    return snapshot, value


def _require_exact_fields(
    value: object,
    expected: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainStatsContractError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise TrainStatsContractError(
            f"{label} has invalid fields; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TrainStatsContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TrainStatsContractError(f"{label} must be a positive integer")
    result = int(value)
    if result < 1:
        raise TrainStatsContractError(f"{label} must be a positive integer")
    return result


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TrainStatsContractError(f"{label} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise TrainStatsContractError(f"{label} must be a non-negative integer")
    return result


def _require_normalized_string(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or "\x00" in value
    ):
        raise TrainStatsContractError(f"{label} must be a non-empty normalized string")
    return value


def _require_finite_vector(value: object, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise TrainStatsContractError(f"{label} must be a non-empty numeric array")
    output: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TrainStatsContractError(f"{label}[{index}] must be a finite number")
        number = float(item)
        if not math.isfinite(number):
            raise TrainStatsContractError(f"{label}[{index}] must be a finite number")
        output.append(number)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class FastWAMLiberoTrainStats:
    """FastWAM-compatible global min/max statistics over raw train rows."""

    action_min: tuple[float, ...]
    action_max: tuple[float, ...]
    state_min: tuple[float, ...]
    state_max: tuple[float, ...]
    num_episodes: int
    num_transition: int

    def __post_init__(self) -> None:
        action_min = _require_finite_vector(self.action_min, label="action_min")
        action_max = _require_finite_vector(self.action_max, label="action_max")
        state_min = _require_finite_vector(self.state_min, label="state_min")
        state_max = _require_finite_vector(self.state_max, label="state_max")
        if len(action_min) != len(action_max):
            raise TrainStatsContractError("action min/max dimensions do not match")
        if len(state_min) != len(state_max):
            raise TrainStatsContractError("state min/max dimensions do not match")
        if any(low > high for low, high in zip(action_min, action_max, strict=True)):
            raise TrainStatsContractError("action minimum exceeds maximum")
        if any(low > high for low, high in zip(state_min, state_max, strict=True)):
            raise TrainStatsContractError("state minimum exceeds maximum")
        episodes = _require_positive_int(self.num_episodes, label="num_episodes")
        frames = _require_positive_int(self.num_transition, label="num_transition")
        if frames < episodes:
            raise TrainStatsContractError(
                "num_transition cannot be smaller than num_episodes"
            )
        object.__setattr__(self, "action_min", action_min)
        object.__setattr__(self, "action_max", action_max)
        object.__setattr__(self, "state_min", state_min)
        object.__setattr__(self, "state_max", state_max)
        object.__setattr__(self, "num_episodes", episodes)
        object.__setattr__(self, "num_transition", frames)

    @property
    def action_dim(self) -> int:
        return len(self.action_min)

    @property
    def state_dim(self) -> int:
        return len(self.state_min)

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON accepted by FastWAM's ``LinearNormalizer``."""

        return {
            "action": {
                "default": {
                    "global_min": list(self.action_min),
                    "global_max": list(self.action_max),
                }
            },
            "state": {
                "default": {
                    "global_min": list(self.state_min),
                    "global_max": list(self.state_max),
                }
            },
            "num_episodes": self.num_episodes,
            # This upstream field is named ``num_transition`` even though the
            # FastWAM implementation stores the total number of raw rows.
            "num_transition": self.num_transition,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FastWAMLiberoTrainStats":
        top = _require_exact_fields(value, _STATS_TOP_FIELDS, label="train stats")

        def parse_kind(kind: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
            outer = _require_exact_fields(
                top[kind], _STATS_KIND_FIELDS, label=f"train stats {kind}"
            )
            ranges = _require_exact_fields(
                outer["default"],
                _STATS_RANGE_FIELDS,
                label=f"train stats {kind}.default",
            )
            return (
                _require_finite_vector(
                    ranges["global_min"], label=f"{kind}.default.global_min"
                ),
                _require_finite_vector(
                    ranges["global_max"], label=f"{kind}.default.global_max"
                ),
            )

        action_min, action_max = parse_kind("action")
        state_min, state_max = parse_kind("state")
        return cls(
            action_min=action_min,
            action_max=action_max,
            state_min=state_min,
            state_max=state_max,
            num_episodes=top["num_episodes"],
            num_transition=top["num_transition"],
        )


@dataclass(frozen=True, slots=True)
class TrainEpisodeSource:
    """Identity and audited full-source digest of one train episode."""

    dataset_id: str
    dataset_index: int
    episode_index: int
    source_episode_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_id",
            _require_normalized_string(self.dataset_id, label="dataset_id"),
        )
        object.__setattr__(
            self,
            "dataset_index",
            _require_nonnegative_int(self.dataset_index, label="dataset_index"),
        )
        object.__setattr__(
            self,
            "episode_index",
            _require_nonnegative_int(self.episode_index, label="episode_index"),
        )
        object.__setattr__(
            self,
            "source_episode_sha256",
            _require_sha256(
                self.source_episode_sha256, label="source_episode_sha256"
            ),
        )

    @property
    def identity(self) -> tuple[str, int, int]:
        return self.dataset_id, self.dataset_index, self.episode_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_index": self.dataset_index,
            "episode_index": self.episode_index,
            "source_episode_sha256": self.source_episode_sha256,
        }


def compute_train_episode_set_sha256(
    episodes: Sequence[TrainEpisodeSource],
) -> str:
    """Hash the exact sorted train identity/full-source set.

    Input order is intentionally irrelevant; duplicate identities are rejected
    instead of being de-duplicated silently.
    """

    items = tuple(episodes)
    if not items:
        raise TrainStatsContractError("train episode source set must not be empty")
    if any(not isinstance(item, TrainEpisodeSource) for item in items):
        raise TypeError("episodes must contain TrainEpisodeSource values")
    identities = [item.identity for item in items]
    if len(identities) != len(set(identities)):
        raise TrainStatsContractError("train episode source set has duplicate identities")
    ordered = sorted(
        items,
        key=lambda item: (item.dataset_index, item.episode_index, item.dataset_id),
    )
    payload = {
        "schema": "warm.train-episode-source-set",
        "version": 1,
        "episodes": [item.to_dict() for item in ordered],
    }
    return sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainStatsManifest:
    """Closed-world provenance manifest for one train-only stats JSON."""

    stats_file_sha256: str
    catalog_sha256: str
    audit_report_sha256: str
    data_config_sha256: str
    train_episode_set_sha256: str
    train_episode_count: int
    train_frame_count: int
    action_dim: int
    state_dim: int
    git_commit: str
    schema: str = TRAIN_STATS_SCHEMA
    version: int = TRAIN_STATS_VERSION
    recipe: str = TRAIN_STATS_RECIPE
    stats_filename: str = TRAIN_STATS_FILENAME

    def __post_init__(self) -> None:
        if self.schema != TRAIN_STATS_SCHEMA:
            raise TrainStatsContractError(
                f"unsupported train-stats schema {self.schema!r}"
            )
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, Integral)
            or int(self.version) != TRAIN_STATS_VERSION
        ):
            raise TrainStatsContractError(
                f"unsupported train-stats version {self.version!r}"
            )
        if self.recipe != TRAIN_STATS_RECIPE:
            raise TrainStatsContractError(f"unsupported train-stats recipe {self.recipe!r}")
        if self.stats_filename != TRAIN_STATS_FILENAME:
            raise TrainStatsContractError(
                f"stats_filename must be {TRAIN_STATS_FILENAME!r}"
            )
        for field in (
            "stats_file_sha256",
            "catalog_sha256",
            "audit_report_sha256",
            "data_config_sha256",
            "train_episode_set_sha256",
        ):
            object.__setattr__(
                self,
                field,
                _require_sha256(getattr(self, field), label=field),
            )
        object.__setattr__(
            self,
            "train_episode_count",
            _require_positive_int(
                self.train_episode_count, label="train_episode_count"
            ),
        )
        object.__setattr__(
            self,
            "train_frame_count",
            _require_positive_int(self.train_frame_count, label="train_frame_count"),
        )
        if self.train_frame_count < self.train_episode_count:
            raise TrainStatsContractError(
                "train_frame_count cannot be smaller than train_episode_count"
            )
        object.__setattr__(
            self,
            "action_dim",
            _require_positive_int(self.action_dim, label="action_dim"),
        )
        object.__setattr__(
            self,
            "state_dim",
            _require_positive_int(self.state_dim, label="state_dim"),
        )
        if not isinstance(self.git_commit, str) or _GIT_COMMIT.fullmatch(
            self.git_commit
        ) is None:
            raise TrainStatsContractError(
                "git_commit must be a lowercase 40-character commit SHA"
            )
        object.__setattr__(self, "schema", TRAIN_STATS_SCHEMA)
        object.__setattr__(self, "version", TRAIN_STATS_VERSION)
        object.__setattr__(self, "recipe", TRAIN_STATS_RECIPE)
        object.__setattr__(self, "stats_filename", TRAIN_STATS_FILENAME)

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
        }

    @property
    def content_sha256(self) -> str:
        return sha256(_canonical_json(self.to_payload())).hexdigest()

    def to_document(self) -> dict[str, Any]:
        return {**self.to_payload(), "content_sha256": self.content_sha256}

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "TrainStatsManifest":
        document = _require_exact_fields(
            value, _MANIFEST_DOCUMENT_FIELDS, label="train-stats manifest"
        )
        expected_hash = _require_sha256(
            document["content_sha256"], label="manifest content_sha256"
        )
        payload = {key: document[key] for key in _MANIFEST_PAYLOAD_FIELDS}
        actual_hash = sha256(_canonical_json(payload)).hexdigest()
        if actual_hash != expected_hash:
            raise TrainStatsContractError("train-stats manifest content hash mismatch")
        manifest = cls(**payload)
        if manifest.content_sha256 != expected_hash:
            raise TrainStatsContractError(
                "train-stats manifest did not round-trip canonically"
            )
        return manifest


@dataclass(frozen=True, slots=True)
class LoadedTrainStatsArtifact:
    stats: FastWAMLiberoTrainStats
    manifest: TrainStatsManifest


def encode_train_stats_json(stats: FastWAMLiberoTrainStats) -> bytes:
    if not isinstance(stats, FastWAMLiberoTrainStats):
        raise TypeError("stats must be FastWAMLiberoTrainStats")
    return _pretty_json(stats.to_dict())


def encode_train_stats_manifest_json(manifest: TrainStatsManifest) -> bytes:
    if not isinstance(manifest, TrainStatsManifest):
        raise TypeError("manifest must be TrainStatsManifest")
    return _pretty_json(manifest.to_document())


def load_train_stats(path: str | Path) -> tuple[FastWAMLiberoTrainStats, str]:
    snapshot, value = _read_json_snapshot(path, label="train stats")
    stats = FastWAMLiberoTrainStats.from_dict(value)
    return stats, sha256(snapshot).hexdigest()


def load_train_stats_manifest(path: str | Path) -> TrainStatsManifest:
    _, value = _read_json_snapshot(path, label="train-stats manifest")
    return TrainStatsManifest.from_document(value)


def load_train_stats_artifact(
    directory: str | Path,
    *,
    expected_catalog_sha256: str | None = None,
    expected_audit_report_sha256: str | None = None,
    expected_data_config_sha256: str | None = None,
    expected_train_episodes: Sequence[TrainEpisodeSource] | None = None,
) -> LoadedTrainStatsArtifact:
    """Load and cross-validate both immutable files in one stats artifact."""

    root = Path(directory)
    if not root.is_dir():
        raise TrainStatsContractError(f"train-stats artifact is not a directory: {root}")
    expected_names = {TRAIN_STATS_FILENAME, TRAIN_STATS_MANIFEST_FILENAME}
    try:
        actual_names = {item.name for item in root.iterdir()}
    except OSError as exc:
        raise TrainStatsContractError(
            f"cannot enumerate train-stats artifact: {root}"
        ) from exc
    if actual_names != expected_names:
        raise TrainStatsContractError(
            "train-stats artifact must contain exactly its stats and manifest files; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    stats, stats_sha256 = load_train_stats(root / TRAIN_STATS_FILENAME)
    manifest = load_train_stats_manifest(root / TRAIN_STATS_MANIFEST_FILENAME)
    if stats_sha256 != manifest.stats_file_sha256:
        raise TrainStatsContractError("stats JSON does not match its manifest SHA-256")
    if stats.action_dim != manifest.action_dim:
        raise TrainStatsContractError("stats action dimension disagrees with manifest")
    if stats.state_dim != manifest.state_dim:
        raise TrainStatsContractError("stats state dimension disagrees with manifest")
    if stats.num_episodes != manifest.train_episode_count:
        raise TrainStatsContractError("stats episode count disagrees with manifest")
    if stats.num_transition != manifest.train_frame_count:
        raise TrainStatsContractError("stats frame count disagrees with manifest")

    expectations = (
        ("catalog_sha256", expected_catalog_sha256),
        ("audit_report_sha256", expected_audit_report_sha256),
        ("data_config_sha256", expected_data_config_sha256),
    )
    for field, expected in expectations:
        if expected is not None:
            expected_digest = _require_sha256(expected, label=f"expected {field}")
            if getattr(manifest, field) != expected_digest:
                raise TrainStatsContractError(f"manifest {field} does not match expected")
    if expected_train_episodes is not None:
        expected_items = tuple(expected_train_episodes)
        expected_hash = compute_train_episode_set_sha256(expected_items)
        if manifest.train_episode_set_sha256 != expected_hash:
            raise TrainStatsContractError(
                "manifest train episode source set does not match expected"
            )
        if manifest.train_episode_count != len(expected_items):
            raise TrainStatsContractError(
                "manifest train episode count does not match expected source set"
            )
    return LoadedTrainStatsArtifact(stats=stats, manifest=manifest)


__all__ = [
    "TRAIN_STATS_FILENAME",
    "TRAIN_STATS_MANIFEST_FILENAME",
    "TRAIN_STATS_RECIPE",
    "TRAIN_STATS_SCHEMA",
    "TRAIN_STATS_VERSION",
    "FastWAMLiberoTrainStats",
    "LoadedTrainStatsArtifact",
    "TrainEpisodeSource",
    "TrainStatsContractError",
    "TrainStatsManifest",
    "compute_train_episode_set_sha256",
    "encode_train_stats_json",
    "encode_train_stats_manifest_json",
    "load_train_stats",
    "load_train_stats_artifact",
    "load_train_stats_manifest",
]
