from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from fastwam.datasets.lerobot.audit import (
    AUDIT_SCHEMA,
    AUDIT_VERSION,
    audit_lerobot_catalog,
    load_audit_report,
    write_audit_report,
)
from fastwam.datasets.lerobot.episode_catalog import (
    assign_task_stratified_dev_split,
    scan_lerobot_datasets,
)
from scripts.audit_warm_lerobot import main as audit_main


CAMERAS = ("observation.images.image", "observation.images.wrist_image")


def _canonical_hash(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _dataset(
    root: Path,
    *,
    duplicate_tables: bool = False,
    duplicate_camera: str | None = None,
) -> None:
    (root / "meta").mkdir(parents=True)
    info = {
        "fps": 20,
        "total_episodes": 3,
        "total_frames": 33,
        "total_tasks": 1,
        "chunks_size": 1000,
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            CAMERAS[0]: {"dtype": "video"},
            CAMERAS[1]: {"dtype": "video"},
            "observation.state": {"dtype": "float32"},
            "action": {"dtype": "float32"},
            "next.reward": {"dtype": "float32"},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    rows = []
    for index in range(3):
        rows.append({"episode_index": index, "length": 11, "tasks": ["pick"]})
        path = root / "data" / "chunk-000" / f"episode_{index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"same-table" if duplicate_tables else f"episode-{index}".encode())
        for camera in CAMERAS:
            video = (
                root / "videos" / "chunk-000" / camera / f"episode_{index:06d}.mp4"
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            if duplicate_camera == camera:
                video.write_bytes(f"same-video:{camera}".encode())
            else:
                video.write_bytes(f"video:{camera}:{index}".encode())
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _catalog(root: Path, *, seed: int = 9):
    return assign_task_stratified_dev_split(
        scan_lerobot_datasets([root]), dev_per_task=1, seed=seed
    )


def test_audit_binds_table_and_ordered_camera_bundle(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _dataset(root)
    catalog = _catalog(root)

    report = audit_lerobot_catalog(
        catalog,
        [root],
        hash_episode_tables=True,
        camera_keys=CAMERAS,
    )

    assert report["schema"] == AUDIT_SCHEMA
    assert report["version"] == AUDIT_VERSION
    assert report["audited_camera_keys"] == list(CAMERAS)
    assert report["summary"]["split_counts"] == {"dev": 1, "train": 2}
    assert report["summary"]["cross_split_duplicate_count"] == 0
    assert report["datasets"][0]["camera_fields"] == list(CAMERAS)
    assert report["datasets"][0]["outcome_fields"] == ["next.reward"]
    assert len(report["episode_source_hashes"]) == 3
    for row in report["episode_source_hashes"]:
        assert row["table_sha256"] != row["source_episode_sha256"]
        assert row["camera_bundle_sha256"] is not None
        assert [item["camera_key"] for item in row["ordered_camera_sha256"]] == list(
            CAMERAS
        )

    output = tmp_path / "audit.json"
    write_audit_report(report, output)
    with pytest.raises(FileExistsError):
        write_audit_report(report, output)

    restored = load_audit_report(output)
    assert restored.catalog_sha256 == catalog.content_sha256
    assert restored.audited_camera_keys == CAMERAS
    assert len(restored.episode_proofs) == 3
    assert all(proof.camera_keys == CAMERAS for proof in restored.episode_proofs)

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["episode_source_hashes"][0]["split"] = "test"
    output.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="report_sha256"):
        load_audit_report(output)


def test_audit_detects_table_and_individual_video_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _dataset(root, duplicate_tables=True, duplicate_camera=CAMERAS[0])
    catalog = _catalog(root, seed=3)

    report = audit_lerobot_catalog(
        catalog,
        [root],
        hash_episode_tables=True,
        camera_keys=CAMERAS,
    )

    summary = report["summary"]
    assert summary["cross_split_table_duplicate_count"] == 1
    assert summary["cross_split_video_duplicate_count"] == 1
    # The other camera stays episode-specific, so neither aggregate bundle repeats.
    assert summary["cross_split_camera_bundle_duplicate_count"] == 0
    assert summary["cross_split_source_bundle_duplicate_count"] == 0
    assert summary["cross_split_duplicate_count"] == 2
    assert {
        row["split"]
        for row in report["cross_split_video_duplicates"][0]["episodes"]
    } == {"train", "dev"}
    assert {
        row["camera_key"]
        for row in report["cross_split_video_duplicates"][0]["episodes"]
    } == {CAMERAS[0]}


def test_loaded_audit_recomputes_duplicate_counts_from_proofs(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _dataset(root, duplicate_tables=True)
    catalog = _catalog(root)
    report = audit_lerobot_catalog(
        catalog,
        [root],
        hash_episode_tables=True,
        camera_keys=CAMERAS,
    )
    # Forge a self-consistently content-hashed report whose summary hides a leak.
    report["summary"]["cross_split_table_duplicate_count"] = 0
    report["summary"]["cross_split_duplicate_count"] = 0
    report.pop("report_sha256")
    report["report_sha256"] = _canonical_hash(report)
    path = tmp_path / "forged-audit.json"
    write_audit_report(report, path)

    with pytest.raises(ValueError, match="duplicate"):
        load_audit_report(path)


def test_audit_rejects_missing_camera_feature_and_escaping_path(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _dataset(root)
    catalog = _catalog(root)
    with pytest.raises(ValueError, match="absent from info.features"):
        audit_lerobot_catalog(
            catalog,
            [root],
            hash_episode_tables=True,
            camera_keys=("observation.images.missing",),
        )

    # Re-scan after changing metadata so the catalog hash remains legitimate;
    # source path safety, not metadata drift, is under test.
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["video_path"] = "../../outside/{video_key}/episode_{episode_index:06d}.mp4"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    catalog = _catalog(root)
    with pytest.raises(ValueError, match="escapes the dataset root"):
        audit_lerobot_catalog(
            catalog,
            [root],
            hash_episode_tables=True,
            camera_keys=CAMERAS,
        )


def test_production_audit_cli_hashes_external_and_wrist_by_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _dataset(root)
    catalog = _catalog(root)
    catalog_path = tmp_path / "catalog.json"
    output = tmp_path / "audit.json"
    catalog.save(catalog_path)

    assert (
        audit_main(
            [
                "--catalog",
                str(catalog_path),
                "--dataset-root",
                str(root),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    restored = load_audit_report(output)
    assert restored.episode_tables_hashed is True
    assert restored.audited_camera_keys == CAMERAS
