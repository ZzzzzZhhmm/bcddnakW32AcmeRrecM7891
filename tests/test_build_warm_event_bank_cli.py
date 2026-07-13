from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.build_warm_event_bank import (
    BuildWarmEventBankError,
    _load_contract_mapping,
    main,
)
from fastwam.memory.bank_builder import EpisodeFeatures
from fastwam.memory.event_bank import EventBank, MANIFEST_FILENAME
from fastwam.memory.feature_cache import FeatureCacheMetadata, save_episode_feature_cache
from fastwam.memory.manifest import sha256_file
from fastwam.memory.offline_pipeline import OfflinePipelineError
from fastwam.utils.artifact_claim import (
    ArtifactAlreadyClaimedError,
    artifact_claim,
)
from tests.warm_test_data import write_test_catalog_and_audit


def _write_contract(path: Path, kind: str) -> tuple[dict[str, object], str]:
    if kind == "normalizer":
        value: dict[str, object] = {
            "schema": "warm.action-space",
            "version": 1,
            "action_dim": 3,
            "arm_dims": [0, 1],
            "gripper_dims": [2],
            "gripper_threshold": 0.0,
            "normalization_mode": "synthetic-test-v1",
            "normalization_stats_sha256": "e" * 64,
            "control_mode": "delta-action",
            "embodiment": "synthetic-libero",
        }
    else:
        value = {
            "kind": kind,
            "revision": "test-v1",
            "parameters": {"alpha": 1, "enabled": True},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return value, sha256_file(path)


def _episode(index: int) -> EpisodeFeatures:
    steps = 8
    actions = np.zeros((steps, 3), dtype=np.float32)
    actions[:, 0] = float(index + 1)
    actions[4:, 1] = 0.5
    actions[4:, 2] = 1.0
    context = np.ones((steps + 1, 3), dtype=np.float32)
    context[:, 1] = float(index) * 0.1
    context[:, 2] = np.linspace(0.1, 0.2, steps + 1, dtype=np.float32)
    semantic = np.full((steps + 1, 2, 4), float(index), dtype=np.float32)
    semantic[:, :, 0] += np.arange(steps + 1, dtype=np.float32)[:, None]
    return EpisodeFeatures(
        dataset_id="libero",
        dataset_index=0,
        episode_index=index,
        task_index=index,
        source_episode_sha256=f"{index + 1:064x}",
        model_actions=actions,
        proprio=np.full((steps + 1, 8), float(index), dtype=np.float32),
        gripper=np.concatenate(
            [np.zeros(4, dtype=np.float32), np.ones(steps - 3, dtype=np.float32)]
        ),
        context_keys=context,
        semantic_features=semantic,
    )


def _prepare_inputs(
    tmp_path: Path,
    *,
    split: str = "train",
) -> dict[str, object]:
    normalizer, normalizer_hash = _write_contract(
        tmp_path / "contracts" / "normalizer.json", "normalizer"
    )
    encoder, encoder_hash = _write_contract(
        tmp_path / "contracts" / "encoder.json", "encoder"
    )
    camera, camera_hash = _write_contract(
        tmp_path / "contracts" / "camera.json", "camera"
    )
    episodes = [_episode(index) for index in range(2)]
    catalog_path, audit_path, catalog, _ = write_test_catalog_and_audit(
        tmp_path / "data-contract",
        [(episode, split) for episode in episodes],
    )
    hashes = {
        "catalog_hash": catalog.content_sha256,
        "normalizer_hash": normalizer_hash,
        "encoder_hash": encoder_hash,
        "camera_hash": camera_hash,
    }

    cache_paths: list[Path] = []
    for index, episode in enumerate(episodes):
        path = tmp_path / "features" / f"episode-{index}.npz"
        save_episode_feature_cache(
            path,
            episode,
            metadata=FeatureCacheMetadata.for_episode(
                episode,
                split=split,
                **hashes,
            ),
        )
        cache_paths.append(path)

    feature_list = tmp_path / "feature-lists" / "train.txt"
    feature_list.parent.mkdir(parents=True, exist_ok=True)
    relative = Path(os.path.relpath(cache_paths[1], feature_list.parent))
    feature_list.write_text(
        f"# one path comes from this UTF-8 list\n\n  {relative}  \n",
        encoding="utf-8",
    )
    return {
        "contracts": {
            "normalizer": normalizer,
            "encoder": encoder,
            "camera": camera,
        },
        "hashes": hashes,
        "cache_paths": cache_paths,
        "feature_list": feature_list,
        "normalizer_path": tmp_path / "contracts" / "normalizer.json",
        "encoder_path": tmp_path / "contracts" / "encoder.json",
        "camera_path": tmp_path / "contracts" / "camera.json",
        "catalog_path": catalog_path,
        "audit_path": audit_path,
    }


def _argv(tmp_path: Path, prepared: dict[str, object]) -> list[str]:
    cache_paths = prepared["cache_paths"]
    return [
        "--feature-cache",
        str(cache_paths[0]),
        "--feature-list",
        str(prepared["feature_list"]),
        "--output",
        str(tmp_path / "bank"),
        "--summary",
        str(tmp_path / "reports" / "summary.json"),
        "--catalog",
        str(prepared["catalog_path"]),
        "--audit-report",
        str(prepared["audit_path"]),
        "--start-mode",
        "uniform",
        "--action-horizon",
        "4",
        "--score-quantile",
        "0.8",
        "--local-max-radius",
        "1",
        "--nms-radius",
        "2",
        "--uniform-stride",
        "3",
        "--gripper-change-threshold",
        "0.01",
        "--mad-epsilon",
        "0.0001",
        "--robust-clip",
        "5.0",
        "--normalizer-contract",
        str(prepared["normalizer_path"]),
        "--encoder-contract",
        str(prepared["encoder_path"]),
        "--camera-contract",
        str(prepared["camera_path"]),
        "--allow-dirty",
    ]


def test_main_builds_bank_manifest_and_atomic_summary(tmp_path: Path) -> None:
    prepared = _prepare_inputs(tmp_path)
    argv = _argv(tmp_path, prepared)

    assert main(argv) == 0

    bank = EventBank.load(tmp_path / "bank")
    assert bank.manifest is not None
    contracts = prepared["contracts"]
    hashes = prepared["hashes"]
    assert dict(bank.manifest.action_normalizer) == {
        "file_sha256": hashes["normalizer_hash"],
        "contract": contracts["normalizer"],
    }
    assert dict(bank.manifest.encoder) == {
        "file_sha256": hashes["encoder_hash"],
        "contract": contracts["encoder"],
    }
    assert dict(bank.manifest.camera_layout) == {
        "file_sha256": hashes["camera_hash"],
        "contract": contracts["camera"],
    }
    assert bank.manifest.provenance["split"] == "train"
    assert bank.manifest.provenance["feature_contract"] == hashes
    assert len(bank.manifest.provenance["software"]["git_commit"]) == 40
    assert isinstance(bank.manifest.provenance["software"]["git_dirty"], bool)

    summary_path = tmp_path / "reports" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["bank_manifest_sha256"] == sha256_file(
        tmp_path / "bank" / MANIFEST_FILENAME
    )
    assert summary["feature_collection_sha256"] == bank.manifest.provenance[
        "feature_collection_sha256"
    ]
    assert summary["data_binding"] == bank.manifest.provenance["data_binding"]
    assert summary["events"] == {"count": 6, "source_episode_count": 2}
    assert summary["action"] == {"horizon": 4, "dimension": 3}
    assert summary["context"] == {"dimension": 3}
    assert summary["effect"] == {"shape": [2, 4]}
    assert summary["recipe"]["event_mining_config"] == {
        "action_horizon": 4,
        "score_quantile": 0.8,
        "local_max_radius": 1,
        "nms_radius": 2,
        "uniform_stride": 3,
        "gripper_change_threshold": 0.01,
        "mad_epsilon": 0.0001,
        "robust_clip": 5.0,
    }
    assert not list((tmp_path / "reports").glob(".*.tmp"))
    assert not (tmp_path / ".bank.warm-build.lock").exists()
    assert not (
        tmp_path / "reports" / ".summary.json.warm-build.lock"
    ).exists()

    before = {
        path: sha256_file(path)
        for path in (
            tmp_path / "bank" / MANIFEST_FILENAME,
            tmp_path / "bank" / "events.npz",
            summary_path,
        )
    }
    with pytest.raises(FileExistsError):
        main(argv)
    with pytest.raises(SystemExit) as exc_info:
        main([*argv, "--overwrite"])
    assert exc_info.value.code == 2
    assert {path: sha256_file(path) for path in before} == before


@pytest.mark.parametrize("claimed_target", ["bank", "summary"])
def test_main_rejects_preexisting_cross_process_claim(
    tmp_path: Path,
    claimed_target: str,
) -> None:
    prepared = _prepare_inputs(tmp_path)
    output = tmp_path / "bank"
    summary = tmp_path / "reports" / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    target = output if claimed_target == "bank" else summary
    claim_path = target.parent / f".{target.name}.warm-build.lock"

    with artifact_claim(claim_path, purpose="competing test publisher"):
        with pytest.raises(ArtifactAlreadyClaimedError, match="already exists"):
            main(_argv(tmp_path, prepared))
        assert claim_path.exists()

    assert not output.exists()
    assert not summary.exists()
    assert not (tmp_path / ".bank.warm-build.lock").exists()
    assert not (
        tmp_path / "reports" / ".summary.json.warm-build.lock"
    ).exists()


def test_main_rejects_contract_file_hash_mismatch(tmp_path: Path) -> None:
    prepared = _prepare_inputs(tmp_path)
    encoder_path = prepared["encoder_path"]
    encoder_path.write_text('{"kind":"changed"}\n', encoding="utf-8")

    with pytest.raises(BuildWarmEventBankError, match="encoder contract SHA-256 mismatch"):
        main(_argv(tmp_path, prepared))
    assert not (tmp_path / "bank" / MANIFEST_FILENAME).exists()


def test_contract_hash_and_json_use_one_byte_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "contract.json"
    original = b'{"revision":"original"}\n'
    path.write_bytes(original)
    real_read_bytes = Path.read_bytes
    reads = 0

    def read_then_replace(self: Path) -> bytes:
        nonlocal reads
        payload = real_read_bytes(self)
        if self.resolve() == path.resolve():
            reads += 1
            path.write_bytes(b'{"revision":"replacement"}\n')
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    mapping = _load_contract_mapping(
        "test", path, hashlib.sha256(original).hexdigest()
    )

    assert reads == 1
    assert mapping["contract"] == {"revision": "original"}
    assert mapping["file_sha256"] == hashlib.sha256(original).hexdigest()


def test_main_requires_at_least_one_feature_cache(tmp_path: Path) -> None:
    normalizer, _ = _write_contract(tmp_path / "normalizer.json", "normalizer")
    assert normalizer
    _write_contract(tmp_path / "encoder.json", "encoder")
    _write_contract(tmp_path / "camera.json", "camera")

    with pytest.raises(BuildWarmEventBankError, match="at least one"):
        main(
            [
                "--output",
                str(tmp_path / "bank"),
                "--summary",
                str(tmp_path / "summary.json"),
                "--catalog",
                str(tmp_path / "catalog.json"),
                "--audit-report",
                str(tmp_path / "audit.json"),
                "--action-horizon",
                "4",
                "--normalizer-contract",
                str(tmp_path / "normalizer.json"),
                "--encoder-contract",
                str(tmp_path / "encoder.json"),
                "--camera-contract",
                str(tmp_path / "camera.json"),
            ]
        )


def test_main_rejects_non_train_feature_collection(tmp_path: Path) -> None:
    prepared = _prepare_inputs(tmp_path, split="dev")

    with pytest.raises(OfflinePipelineError, match="must use split 'train'"):
        main(_argv(tmp_path, prepared))
    assert not (tmp_path / "bank" / MANIFEST_FILENAME).exists()
