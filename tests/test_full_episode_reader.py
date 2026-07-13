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
    CameraVideoAuditProof,
    EpisodeAuditProof,
    compute_camera_bundle_sha256,
    compute_source_bundle_sha256,
)
from fastwam.datasets.lerobot.episode_catalog import EpisodeRecord
from fastwam.datasets.lerobot.full_episode_reader import read_full_lerobot_episode


CAMERAS = ("observation.images.front", "observation.images.wrist")


def test_module_import_does_not_require_torch_or_pyarrow() -> None:
    script = """
import importlib.abc
import sys

class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'torch', 'pyarrow'}:
            raise ImportError(f'blocked dependency: {fullname}')
        return None

sys.meta_path.insert(0, BlockHeavyImports())
import fastwam.datasets.lerobot.full_episode_reader
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
    assert completed.returncode == 0, completed.stderr


def _episode_fixture(tmp_path: Path, *, rows: int = 4) -> tuple[Path, EpisodeRecord, EpisodeAuditProof, bytes]:
    root = tmp_path / "dataset"
    data_path = root / "data" / "chunk-000" / "episode_000007.parquet"
    data_path.parent.mkdir(parents=True)
    payload = b"audited-parquet-snapshot"
    data_path.write_bytes(payload)
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 20,
                "chunks_size": 1000,
                "video_path": (
                    "videos/chunk-{episode_chunk:03d}/{video_key}/"
                    "episode_{episode_index:06d}.mp4"
                ),
                "features": {
                    camera: {"dtype": "video"} for camera in CAMERAS
                },
            }
        ),
        encoding="utf-8",
    )
    for camera in CAMERAS:
        video_path = root / "videos" / "chunk-000" / camera / "episode_000007.mp4"
        video_path.parent.mkdir(parents=True)
        video_path.write_bytes(camera.encode("utf-8"))
    record = EpisodeRecord(
        dataset_id="fixture",
        dataset_index=0,
        episode_index=7,
        length=rows,
        fps=20.0,
        tasks=("pick",),
        data_relpath="data/chunk-000/episode_000007.parquet",
        split="train",
    )
    table_sha256 = sha256(payload).hexdigest()
    camera_hashes = tuple(
        CameraVideoAuditProof(camera, sha256(camera.encode("utf-8")).hexdigest())
        for camera in CAMERAS
    )
    proof = EpisodeAuditProof(
        dataset_id="fixture",
        dataset_index=0,
        episode_index=7,
        split="train",
        table_sha256=table_sha256,
        ordered_camera_sha256=camera_hashes,
        camera_bundle_sha256=compute_camera_bundle_sha256(camera_hashes),
        source_episode_sha256=compute_source_bundle_sha256(
            table_sha256,
            camera_hashes,
        ),
    )
    return root, record, proof, payload


def _table(rows: int = 4) -> dict[str, Any]:
    return {
        "episode_index": np.full(rows, 7, dtype=np.int64),
        "frame_index": np.arange(rows, dtype=np.int64),
        "timestamp": np.arange(rows, dtype=np.float64) / 20.0,
        "task_index": np.ones(rows, dtype=np.int64),
        "action": np.arange(rows * 3, dtype=np.float32).reshape(rows, 3),
        "observation.state": np.arange(rows * 5, dtype=np.float64).reshape(rows, 5),
        "unused": np.arange(rows),
    }


def _decoder(calls: list[tuple[Path, tuple[float, ...], float]]):
    def decode(path: Path, timestamps: Any, tolerance: float) -> np.ndarray:
        calls.append((path, tuple(float(value) for value in timestamps), tolerance))
        return np.ones((len(timestamps), 3, 2, 3), dtype=np.float64)

    return decode


def test_reads_exact_audited_snapshot_and_returns_immutable_full_episode(tmp_path: Path) -> None:
    root, record, proof, payload = _episode_fixture(tmp_path)
    parsed_payloads: list[bytes] = []
    decode_calls: list[tuple[Path, tuple[float, ...], float]] = []

    episode = read_full_lerobot_episode(
        record,
        dataset_root=root,
        audit_proof=proof,
        camera_keys=CAMERAS,
        table_reader=lambda value: parsed_payloads.append(value) or _table(),
        video_decoder=_decoder(decode_calls),
    )

    assert parsed_payloads == [payload]
    assert episode.source_episode_sha256 == proof.source_episode_sha256
    assert episode.actions.shape == (4, 3)  # Raw parquet N, not an N-1 window.
    assert episode.states.shape == (4, 5)
    assert episode.actions.dtype == np.float32
    assert episode.states.dtype == np.float32
    assert episode.timestamps.dtype == np.float64
    assert episode.task_indices.tolist() == [1, 1, 1, 1]
    assert set(episode.images) == set(CAMERAS)
    assert all(frames.shape == (4, 3, 2, 3) for frames in episode.images.values())
    assert all(frames.dtype == np.float32 for frames in episode.images.values())
    assert len(decode_calls) == 2
    assert decode_calls[0][1] == (0.0, 0.05, 0.1, 0.15)

    with pytest.raises(ValueError, match="read-only"):
        episode.actions[0, 0] = 99
    with pytest.raises(ValueError):
        episode.actions.setflags(write=True)
    with pytest.raises(TypeError):
        episode.images["new"] = np.zeros((4, 3, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        episode.images[CAMERAS[0]].setflags(write=True)


def test_parser_consumes_the_same_bytes_that_were_hashed(tmp_path: Path) -> None:
    root, record, proof, payload = _episode_fixture(tmp_path)
    parquet_path = root / record.data_relpath

    def mutate_after_snapshot(snapshot: bytes) -> dict[str, Any]:
        assert snapshot == payload
        parquet_path.write_bytes(b"changed-after-single-read")
        return _table()

    episode = read_full_lerobot_episode(
        record,
        dataset_root=root,
        audit_proof=proof,
        camera_keys=CAMERAS,
        table_reader=mutate_after_snapshot,
        video_decoder=lambda _path, timestamps, _tol: np.zeros(
            (len(timestamps), 3, 2, 2), dtype=np.float32
        ),
    )

    assert episode.source_episode_sha256 == proof.source_episode_sha256
    assert parquet_path.read_bytes() != payload


def test_hash_and_identity_are_verified_before_parsing(tmp_path: Path) -> None:
    root, record, proof, _ = _episode_fixture(tmp_path)
    calls = 0

    def reader(_payload: bytes) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _table()

    changed_table = "0" * 64
    changed = EpisodeAuditProof(
        dataset_id=proof.dataset_id,
        dataset_index=proof.dataset_index,
        episode_index=proof.episode_index,
        split=proof.split,
        table_sha256=changed_table,
        ordered_camera_sha256=proof.ordered_camera_sha256,
        camera_bundle_sha256=proof.camera_bundle_sha256,
        source_episode_sha256=compute_source_bundle_sha256(
            changed_table,
            proof.ordered_camera_sha256,
        ),
    )
    with pytest.raises(ValueError, match="audited SHA-256"):
        read_full_lerobot_episode(
            record,
            dataset_root=root,
            audit_proof=changed,
            camera_keys=CAMERAS,
            table_reader=reader,
            video_decoder=lambda *_args: None,
        )
    assert calls == 0

    wrong_identity = EpisodeAuditProof(
        dataset_id=proof.dataset_id,
        dataset_index=proof.dataset_index,
        episode_index=8,
        split=proof.split,
        table_sha256=proof.table_sha256,
        ordered_camera_sha256=proof.ordered_camera_sha256,
        camera_bundle_sha256=proof.camera_bundle_sha256,
        source_episode_sha256=proof.source_episode_sha256,
    )
    with pytest.raises(ValueError, match="identity"):
        read_full_lerobot_episode(
            record,
            dataset_root=root,
            audit_proof=wrong_identity,
            camera_keys=CAMERAS,
            table_reader=reader,
            video_decoder=lambda *_args: None,
        )
    assert calls == 0


def test_requested_cameras_must_exactly_match_audit_order(tmp_path: Path) -> None:
    root, record, proof, _ = _episode_fixture(tmp_path)
    calls = 0

    def reader(_payload: bytes) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _table()

    for requested in ((CAMERAS[0],), tuple(reversed(CAMERAS))):
        with pytest.raises(ValueError, match="exactly match the audit proof order"):
            read_full_lerobot_episode(
                record,
                dataset_root=root,
                audit_proof=proof,
                camera_keys=requested,
                table_reader=reader,
                video_decoder=lambda *_args: pytest.fail("decoder must not run"),
            )
    assert calls == 0


def test_rejects_camera_changed_since_audit_before_decode(tmp_path: Path) -> None:
    root, record, proof, _ = _episode_fixture(tmp_path)
    camera_path = (
        root / "videos" / "chunk-000" / CAMERAS[0] / "episode_000007.mp4"
    )
    camera_path.write_bytes(b"tampered-before-read")

    with pytest.raises(ValueError, match="before decode"):
        read_full_lerobot_episode(
            record,
            dataset_root=root,
            audit_proof=proof,
            camera_keys=CAMERAS,
            table_reader=lambda _payload: _table(),
            video_decoder=lambda *_args: pytest.fail("decoder must not run"),
        )


def test_rejects_camera_replaced_during_decode(tmp_path: Path) -> None:
    root, record, proof, _ = _episode_fixture(tmp_path)

    def mutating_decoder(path: Path, timestamps: Any, _tolerance: float) -> np.ndarray:
        path.write_bytes(b"replaced-during-decode")
        return np.zeros((len(timestamps), 3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="changed while it was being decoded"):
        read_full_lerobot_episode(
            record,
            dataset_root=root,
            audit_proof=proof,
            camera_keys=CAMERAS,
            table_reader=lambda _payload: _table(),
            video_decoder=mutating_decoder,
        )


def test_rehashes_each_camera_immediately_before_its_decode(tmp_path: Path) -> None:
    root, record, proof, _ = _episode_fixture(tmp_path)
    second_path = (
        root / "videos" / "chunk-000" / CAMERAS[1] / "episode_000007.mp4"
    )
    decoded: list[str] = []

    def mutate_later_camera(
        path: Path, timestamps: Any, _tolerance: float
    ) -> np.ndarray:
        decoded.append(path.parent.name)
        if path.parent.name == CAMERAS[0]:
            second_path.write_bytes(b"changed-by-first-camera-decoder")
        return np.zeros((len(timestamps), 3, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="before it could be decoded"):
        read_full_lerobot_episode(
            record,
            dataset_root=root,
            audit_proof=proof,
            camera_keys=CAMERAS,
            table_reader=lambda _payload: _table(),
            video_decoder=mutate_later_camera,
        )
    assert decoded == [CAMERAS[0]]

@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda table: table["episode_index"].__setitem__(1, 8), "episode_index"),
        (lambda table: table["frame_index"].__setitem__(1, 3), "frame_index"),
        (lambda table: table["timestamp"].__setitem__(2, 0.2), "timestamps"),
        (lambda table: table["timestamp"].__setitem__(2, np.nan), "non-finite"),
        (lambda table: table["task_index"].__setitem__(0, -1), "task_index"),
        (lambda table: table["task_index"].__setitem__(0, 2), "constant"),
        (lambda table: table["action"].__setitem__((0, 0), np.nan), "non-finite"),
        (lambda table: table["observation.state"].__setitem__((0, 0), np.inf), "non-finite"),
    ],
)
def test_strictly_validates_episode_columns(tmp_path: Path, mutation, message: str) -> None:
    root, record, proof, _ = _episode_fixture(tmp_path)
    table = _table()
    mutation(table)

    with pytest.raises(ValueError, match=message):
        read_full_lerobot_episode(
            record,
            dataset_root=root,
            audit_proof=proof,
            camera_keys=CAMERAS,
            table_reader=lambda _payload: table,
            video_decoder=lambda _path, timestamps, _tol: np.zeros(
                (len(timestamps), 3, 2, 2), dtype=np.float32
            ),
        )


def test_rejects_wrong_row_count_and_noninteger_task_index(tmp_path: Path) -> None:
    root, record, proof, _ = _episode_fixture(tmp_path)
    short = _table()
    short["unused"] = short["unused"][:-1]
    with pytest.raises(ValueError, match="3 rows, expected 4"):
        read_full_lerobot_episode(
            record,
            dataset_root=root,
            audit_proof=proof,
            camera_keys=CAMERAS,
            table_reader=lambda _payload: short,
            video_decoder=lambda *_args: None,
        )

    float_tasks = _table()
    float_tasks["task_index"] = float_tasks["task_index"].astype(np.float32)
    with pytest.raises(ValueError, match="must contain integers"):
        read_full_lerobot_episode(
            record,
            dataset_root=root,
            audit_proof=proof,
            camera_keys=CAMERAS,
            table_reader=lambda _payload: float_tasks,
            video_decoder=lambda *_args: None,
        )


@pytest.mark.parametrize(
    "frames",
    [
        np.zeros((4, 2, 2, 3), dtype=np.float32),
        np.zeros((3, 3, 2, 2), dtype=np.float32),
        np.full((4, 3, 2, 2), np.nan, dtype=np.float32),
        np.full((4, 3, 2, 2), 1.01, dtype=np.float32),
        np.full((4, 3, 2, 2), -0.01, dtype=np.float32),
    ],
)
def test_strictly_validates_decoded_frame_contract(tmp_path: Path, frames: np.ndarray) -> None:
    root, record, proof, _ = _episode_fixture(tmp_path)
    with pytest.raises(ValueError, match="decoder output"):
        read_full_lerobot_episode(
            record,
            dataset_root=root,
            audit_proof=proof,
            camera_keys=CAMERAS,
            table_reader=lambda _payload: _table(),
            video_decoder=lambda *_args: frames,
        )


def test_rejects_video_template_that_escapes_dataset_root(tmp_path: Path) -> None:
    root, record, proof, _ = _episode_fixture(tmp_path)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["video_path"] = (
        "../../outside/{video_key}/episode_{episode_index:06d}.mp4"
    )
    info_path.write_text(json.dumps(info), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes the dataset root"):
        read_full_lerobot_episode(
            record,
            dataset_root=root,
            audit_proof=proof,
            camera_keys=CAMERAS,
            table_reader=lambda _payload: _table(),
            video_decoder=lambda *_args: pytest.fail("decoder must not run"),
        )
