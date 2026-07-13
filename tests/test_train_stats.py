from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from fastwam.memory.train_stats import (
    TRAIN_STATS_FILENAME,
    TRAIN_STATS_MANIFEST_FILENAME,
    TRAIN_STATS_RECIPE,
    FastWAMLiberoTrainStats,
    TrainEpisodeSource,
    TrainStatsContractError,
    TrainStatsManifest,
    compute_train_episode_set_sha256,
    encode_train_stats_json,
    encode_train_stats_manifest_json,
    load_train_stats_artifact,
)


HASHES = {
    "catalog": "a" * 64,
    "audit": "b" * 64,
    "config": "c" * 64,
}
COMMIT = "d" * 40


def _stats() -> FastWAMLiberoTrainStats:
    return FastWAMLiberoTrainStats(
        action_min=(-1.0, 0.0),
        action_max=(1.0, 2.0),
        state_min=(-3.0, -2.0, -1.0),
        state_max=(3.0, 2.0, 1.0),
        num_episodes=2,
        num_transition=7,
    )


def _sources() -> tuple[TrainEpisodeSource, ...]:
    return (
        TrainEpisodeSource("dataset-b", 1, 2, "2" * 64),
        TrainEpisodeSource("dataset-a", 0, 3, "1" * 64),
    )


def _manifest(stats_payload: bytes) -> TrainStatsManifest:
    return TrainStatsManifest(
        stats_file_sha256=sha256(stats_payload).hexdigest(),
        catalog_sha256=HASHES["catalog"],
        audit_report_sha256=HASHES["audit"],
        data_config_sha256=HASHES["config"],
        train_episode_set_sha256=compute_train_episode_set_sha256(_sources()),
        train_episode_count=2,
        train_frame_count=7,
        action_dim=2,
        state_dim=3,
        git_commit=COMMIT,
    )


def _write_artifact(tmp_path):
    artifact = tmp_path / "stats"
    artifact.mkdir(parents=True)
    stats_payload = encode_train_stats_json(_stats())
    manifest = _manifest(stats_payload)
    (artifact / TRAIN_STATS_FILENAME).write_bytes(stats_payload)
    (artifact / TRAIN_STATS_MANIFEST_FILENAME).write_bytes(
        encode_train_stats_manifest_json(manifest)
    )
    return artifact, manifest


def test_train_stats_round_trip_and_fastwam_shape(tmp_path) -> None:
    artifact, manifest = _write_artifact(tmp_path)
    loaded = load_train_stats_artifact(
        artifact,
        expected_catalog_sha256=HASHES["catalog"],
        expected_audit_report_sha256=HASHES["audit"],
        expected_data_config_sha256=HASHES["config"],
        expected_train_episodes=tuple(reversed(_sources())),
    )
    assert loaded.stats == _stats()
    assert loaded.manifest == manifest
    assert loaded.manifest.recipe == TRAIN_STATS_RECIPE
    raw = json.loads((artifact / TRAIN_STATS_FILENAME).read_text(encoding="utf-8"))
    assert raw == {
        "action": {
            "default": {"global_min": [-1.0, 0.0], "global_max": [1.0, 2.0]}
        },
        "state": {
            "default": {
                "global_min": [-3.0, -2.0, -1.0],
                "global_max": [3.0, 2.0, 1.0],
            }
        },
        "num_episodes": 2,
        "num_transition": 7,
    }


def test_episode_set_hash_is_order_independent_and_source_sensitive() -> None:
    sources = _sources()
    assert compute_train_episode_set_sha256(sources) == (
        compute_train_episode_set_sha256(tuple(reversed(sources)))
    )
    changed = replace(sources[0], source_episode_sha256="f" * 64)
    assert compute_train_episode_set_sha256((changed, sources[1])) != (
        compute_train_episode_set_sha256(sources)
    )
    with pytest.raises(TrainStatsContractError, match="duplicate identities"):
        compute_train_episode_set_sha256((sources[0], sources[0]))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: {**value, "extra": 1}, "invalid fields"),
        (
            lambda value: {
                **value,
                "action": {
                    "default": {
                        "global_min": [float("nan"), 0.0],
                        "global_max": [1.0, 2.0],
                    }
                },
            },
            "finite",
        ),
        (
            lambda value: {
                **value,
                "state": {
                    "default": {
                        "global_min": [4.0, -2.0, -1.0],
                        "global_max": [3.0, 2.0, 1.0],
                    }
                },
            },
            "minimum exceeds maximum",
        ),
    ],
)
def test_stats_reject_unknown_nonfinite_and_invalid_ranges(mutate, message) -> None:
    value = mutate(_stats().to_dict())
    with pytest.raises(TrainStatsContractError, match=message):
        FastWAMLiberoTrainStats.from_dict(value)


def test_manifest_rejects_extra_fields_and_hash_tampering() -> None:
    stats_payload = encode_train_stats_json(_stats())
    manifest = _manifest(stats_payload)
    document = manifest.to_document()
    with pytest.raises(TrainStatsContractError, match="invalid fields"):
        TrainStatsManifest.from_document({**document, "unknown": 1})
    with pytest.raises(TrainStatsContractError, match="content hash mismatch"):
        TrainStatsManifest.from_document({**document, "train_frame_count": 8})


def test_artifact_rejects_stats_mutation_and_extra_files(tmp_path) -> None:
    artifact, _ = _write_artifact(tmp_path)
    with (artifact / TRAIN_STATS_FILENAME).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(TrainStatsContractError, match="manifest SHA-256"):
        load_train_stats_artifact(artifact)

    artifact2, _ = _write_artifact(tmp_path / "second")
    (artifact2 / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(TrainStatsContractError, match="contain exactly"):
        load_train_stats_artifact(artifact2)


def test_json_loader_rejects_duplicate_keys_and_nonstandard_nan(tmp_path) -> None:
    artifact, _ = _write_artifact(tmp_path)
    (artifact / TRAIN_STATS_FILENAME).write_text(
        '{"action":{},"action":{},"state":{},"num_episodes":2,'
        '"num_transition":7}',
        encoding="utf-8",
    )
    with pytest.raises(TrainStatsContractError, match="duplicate JSON key"):
        load_train_stats_artifact(artifact)

    (artifact / TRAIN_STATS_FILENAME).write_text("NaN", encoding="utf-8")
    with pytest.raises(TrainStatsContractError, match="forbidden JSON constant"):
        load_train_stats_artifact(artifact)
