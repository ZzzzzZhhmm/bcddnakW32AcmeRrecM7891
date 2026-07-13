from __future__ import annotations

from hashlib import sha256
import json

import pytest

from fastwam.memory.robotwin_train_stats import (
    FastWAMRobotwinTrainStats,
    RobotwinTrainStatsError,
    RobotwinTrainStatsManifest,
    encode_robotwin_stats,
)


def _stats() -> FastWAMRobotwinTrainStats:
    return FastWAMRobotwinTrainStats(
        action_mean=tuple(range(14)),
        action_std=(1.0,) * 14,
        state_mean=tuple(range(14)),
        state_std=(2.0,) * 14,
        num_episodes=2,
        num_transition=20,
    )


def test_robotwin_stats_match_fastwam_zscore_json() -> None:
    value = _stats()
    restored = FastWAMRobotwinTrainStats.from_dict(value.to_dict())
    assert restored == value
    assert restored.to_dict()["action"]["default"]["global_std"] == [1.0] * 14


def test_robotwin_stats_manifest_round_trip() -> None:
    stats = _stats()
    raw = encode_robotwin_stats(stats)
    manifest = RobotwinTrainStatsManifest(
        stats_file_sha256=sha256(raw).hexdigest(),
        catalog_sha256="a" * 64,
        audit_report_sha256="b" * 64,
        data_config_sha256="c" * 64,
        train_episode_set_sha256="d" * 64,
        train_episode_count=2,
        train_frame_count=20,
        git_commit="e" * 40,
        dataset_revision="855e90e1213d150bf4889130e83398f107314681",
    )
    assert RobotwinTrainStatsManifest.from_document(manifest.to_document()) == manifest
    broken = json.loads(json.dumps(manifest.to_document()))
    broken["train_frame_count"] = 21
    with pytest.raises(RobotwinTrainStatsError, match="content hash"):
        RobotwinTrainStatsManifest.from_document(broken)
