from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from fastwam.datasets.lerobot.audit import (
    audit_lerobot_catalog,
    write_audit_report,
)
from fastwam.datasets.lerobot.episode_catalog import (
    DatasetDescriptor,
    EpisodeCatalog,
    EpisodeRecord,
)
from fastwam.memory.train_stats import load_train_stats_artifact
import scripts.compute_warm_train_stats as stats_cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1" * 40
SOURCE_CAMERAS = (
    "observation.images.image",
    "observation.images.wrist_image",
)


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _valid_config() -> dict[str, Any]:
    return {
        "train": {
            "shape_meta": {
                "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
                "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
            },
            "processor": {
                "_target_": (
                    "fastwam.datasets.lerobot.processors.fastwam_processor."
                    "FastWAMProcessor"
                ),
                "action_state_transforms": None,
                "use_stepwise_action_norm": False,
                "norm_default_mode": "min/max",
                "norm_exception_mode": None,
                "action_output_dim": 7,
                "proprio_output_dim": 8,
            },
        }
    }


def _table(index: int, *, dev: bool = False) -> dict[str, np.ndarray]:
    if dev:
        actions = np.full((3, 7), -999.0, dtype=np.float32)
        states = np.full((3, 8), 999.0, dtype=np.float32)
    else:
        actions = (
            np.arange(21, dtype=np.float32).reshape(3, 7) + index * 100.0
        )
        states = (
            np.arange(24, dtype=np.float32).reshape(3, 8) - index * 100.0
        )
    return {
        "action": actions,
        "observation.state": states,
        "episode_index": np.full(3, index, dtype=np.int64),
        "frame_index": np.arange(3, dtype=np.int64),
        "timestamp": np.arange(3, dtype=np.float64) / 20.0,
        "task_index": np.full(3, 4, dtype=np.int64),
    }


def _fixture(tmp_path: Path):
    root = tmp_path / "dataset"
    meta = root / "meta"
    data = root / "data" / "chunk-000"
    meta.mkdir(parents=True)
    data.mkdir(parents=True)
    info = {
        "fps": 20,
        "total_episodes": 3,
        "chunks_size": 1000,
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "action": {"dtype": "float32"},
            "observation.state": {"dtype": "float32"},
            "observation.images.image": {"dtype": "video"},
            "observation.images.wrist_image": {"dtype": "video"},
        },
    }
    info_path = meta / "info.json"
    info_path.write_text(json.dumps(info, sort_keys=True) + "\n", encoding="utf-8")
    episode_rows = [
        {"episode_index": index, "length": 3, "tasks": ["pick object"]}
        for index in range(3)
    ]
    episodes_path = meta / "episodes.jsonl"
    episodes_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in episode_rows),
        encoding="utf-8",
    )
    payload_tables: dict[bytes, dict[str, np.ndarray]] = {}
    records: list[EpisodeRecord] = []
    for index in range(3):
        payload = f"fake-parquet-{index}".encode("utf-8")
        path = data / f"episode_{index:06d}.parquet"
        path.write_bytes(payload)
        for camera in SOURCE_CAMERAS:
            video = (
                root
                / "videos"
                / "chunk-000"
                / camera
                / f"episode_{index:06d}.mp4"
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"video:{camera}:{index}".encode("utf-8"))
        payload_tables[payload] = _table(index, dev=index == 2)
        records.append(
            EpisodeRecord(
                dataset_id="libero-test",
                dataset_index=0,
                episode_index=index,
                length=3,
                fps=20.0,
                tasks=("pick object",),
                data_relpath=f"data/chunk-000/episode_{index:06d}.parquet",
                split="dev" if index == 2 else "train",
            )
        )
    catalog = EpisodeCatalog(
        datasets=(
            DatasetDescriptor(
                dataset_id="libero-test",
                dataset_index=0,
                fps=20.0,
                total_episodes=3,
                chunks_size=1000,
                data_path_template=info["data_path"],
                info_sha256=_file_sha(info_path),
                episodes_sha256=_file_sha(episodes_path),
            ),
        ),
        episodes=tuple(records),
    )
    catalog_path = tmp_path / "catalog.json"
    catalog.save(catalog_path)
    report = audit_lerobot_catalog(
        catalog,
        [root],
        hash_episode_tables=True,
        camera_keys=SOURCE_CAMERAS,
    )
    audit_path = tmp_path / "audit.json"
    write_audit_report(report, audit_path)
    config_path = tmp_path / "libero.yaml"
    config_path.write_text("train:\n  processor: {}\n", encoding="utf-8")
    output = tmp_path / "artifacts" / "train-stats"
    args = [
        "--catalog",
        str(catalog_path),
        "--audit-report",
        str(audit_path),
        "--dataset-root",
        str(root),
        "--data-config",
        str(config_path),
        "--output",
        str(output),
    ]
    return {
        "root": root,
        "catalog": catalog,
        "catalog_path": catalog_path,
        "audit_path": audit_path,
        "config_path": config_path,
        "output": output,
        "args": args,
        "payload_tables": payload_tables,
    }


def test_cli_computes_only_train_rows_and_includes_terminal_action(
    tmp_path, monkeypatch
) -> None:
    fixture = _fixture(tmp_path)
    seen: list[bytes] = []

    def reader(payload: bytes):
        seen.append(payload)
        return fixture["payload_tables"][payload]

    monkeypatch.setattr(stats_cli, "_git_clean_commit", lambda _: COMMIT)
    monkeypatch.setattr(stats_cli, "_default_config_loader", lambda _: _valid_config())
    monkeypatch.setattr(stats_cli, "_default_table_reader", reader)
    assert stats_cli.main(fixture["args"]) == 0

    artifact = load_train_stats_artifact(fixture["output"])
    stats = artifact.stats
    # Episode 0's last raw row (14..20) is included. Episode 1 supplies the max.
    assert stats.action_min == tuple(float(value) for value in np.arange(7))
    assert stats.action_max == tuple(float(value) for value in np.arange(114, 121))
    assert stats.state_min == tuple(float(value) for value in np.arange(-100, -92))
    assert stats.state_max == tuple(float(value) for value in np.arange(16, 24))
    assert stats.num_episodes == 2
    assert stats.num_transition == 6
    assert artifact.manifest.git_commit == COMMIT
    assert artifact.manifest.catalog_sha256 == fixture["catalog"].content_sha256
    assert set(seen) == {b"fake-parquet-0", b"fake-parquet-1"}
    assert b"fake-parquet-2" not in seen


def test_plan_rejects_non_null_action_transform(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    config = _valid_config()
    config["train"]["processor"]["action_state_transforms"] = [
        {"_target_": "bad.Transform"}
    ]
    with pytest.raises(stats_cli.TrainStatsComputationError, match="transforms: null"):
        stats_cli.prepare_plan(
            catalog_path=fixture["catalog_path"],
            audit_report_path=fixture["audit_path"],
            dataset_roots=[fixture["root"]],
            data_config_path=fixture["config_path"],
            output=fixture["output"],
            config_loader=lambda _: config,
        )


def test_plan_requires_complete_two_camera_source_audit(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    table_only_report = audit_lerobot_catalog(
        fixture["catalog"],
        [fixture["root"]],
        hash_episode_tables=True,
        camera_keys=(),
    )
    table_only_path = tmp_path / "table-only-audit.json"
    write_audit_report(table_only_report, table_only_path)
    with pytest.raises(stats_cli.TrainStatsComputationError, match="two-camera"):
        stats_cli.prepare_plan(
            catalog_path=fixture["catalog_path"],
            audit_report_path=table_only_path,
            dataset_roots=[fixture["root"]],
            data_config_path=fixture["config_path"],
            output=fixture["output"],
            config_loader=lambda _: _valid_config(),
        )


@pytest.mark.parametrize(
    ("column", "mutate", "message"),
    [
        (
            "action",
            lambda value: np.full_like(value, np.nan),
            "non-finite",
        ),
        (
            "frame_index",
            lambda value: np.asarray([0, 2, 1], dtype=np.int64),
            "contiguous",
        ),
        (
            "task_index",
            lambda value: np.asarray([4, 5, 4], dtype=np.int64),
            "constant",
        ),
        (
            "episode_index",
            lambda value: np.asarray([9, 9, 9], dtype=np.int64),
            "disagrees",
        ),
    ],
)
def test_reader_rejects_invalid_table_contract(
    tmp_path, column, mutate, message
) -> None:
    fixture = _fixture(tmp_path)
    plan = stats_cli.prepare_plan(
        catalog_path=fixture["catalog_path"],
        audit_report_path=fixture["audit_path"],
        dataset_roots=[fixture["root"]],
        data_config_path=fixture["config_path"],
        output=fixture["output"],
        config_loader=lambda _: _valid_config(),
    )
    table = {key: np.array(value, copy=True) for key, value in _table(0).items()}
    table[column] = mutate(table[column])
    with pytest.raises(stats_cli.TrainStatsComputationError, match=message):
        stats_cli.read_audited_stats_rows(
            plan.records[0],
            proof=plan.proofs[0],
            dataset_root=plan.roots[0],
            action_dim=7,
            state_dim=8,
            timestamp_tolerance_s=1e-4,
            table_reader=lambda _: table,
        )


def test_formal_cli_allows_dirty_tree_before_writing(tmp_path, monkeypatch) -> None:
    fixture = _fixture(tmp_path)

    def reader(payload: bytes):
        return fixture["payload_tables"][payload]

    monkeypatch.setattr(stats_cli, "_git_clean_commit", lambda _: COMMIT)
    monkeypatch.setattr(stats_cli, "_default_config_loader", lambda _: _valid_config())
    monkeypatch.setattr(stats_cli, "_default_table_reader", reader)
    assert stats_cli.main(fixture["args"]) == 0
    assert fixture["output"].exists()


def test_output_is_immutable_and_no_overwrite(tmp_path) -> None:
    fixture = _fixture(tmp_path)
    fixture["output"].mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        stats_cli.prepare_plan(
            catalog_path=fixture["catalog_path"],
            audit_report_path=fixture["audit_path"],
            dataset_roots=[fixture["root"]],
            data_config_path=fixture["config_path"],
            output=fixture["output"],
            config_loader=lambda _: _valid_config(),
        )


def test_import_does_not_load_heavy_runtime_modules() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    command = (
        "import sys; import scripts.compute_warm_train_stats; "
        "forbidden={'torch','pyarrow','hydra','omegaconf'}; "
        "loaded=sorted(forbidden.intersection(sys.modules)); "
        "assert not loaded, loaded"
    )
    subprocess.run(
        [sys.executable, "-c", command],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
