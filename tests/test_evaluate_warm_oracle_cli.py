from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_warm_oracle import main
from fastwam.memory.bank_builder import EpisodeFeatures
from fastwam.memory.candidate_cache import canonical_event_bank_content_hash
from fastwam.memory.event_bank import EventBank
from fastwam.memory.event_mining import EventMiningConfig
from fastwam.memory.feature_cache import FeatureCacheMetadata, save_episode_feature_cache
from fastwam.memory.manifest import sha256_file
from fastwam.memory.offline_pipeline import (
    FeatureDataBinding,
    build_event_bank_from_collection,
    load_feature_cache_collection,
)
from fastwam.utils.artifact_claim import (
    ArtifactAlreadyClaimedError,
    artifact_claim,
)
from tests.warm_test_data import write_test_catalog_and_audit


HASHES = {
    "catalog_hash": "a" * 64,
    "normalizer_hash": "b" * 64,
    "encoder_hash": "c" * 64,
    "camera_hash": "d" * 64,
}

ACTION_SPACE_CONTRACT = {
    "schema": "warm.action-space",
    "version": 1,
    "action_dim": 3,
    "arm_dims": [0, 1],
    "gripper_dims": [2],
    "gripper_threshold": 0.1,
    "normalization_mode": "synthetic-test-v1",
    "normalization_stats_sha256": "e" * 64,
    "control_mode": "delta-action",
    "embodiment": "synthetic-libero",
}


def _episode(index: int) -> EpisodeFeatures:
    steps = 8
    actions = np.zeros((steps, 3), dtype=np.float32)
    actions[:, :2] = float(index) / 10.0
    actions[4:, 2] = 1.0
    context = np.zeros((steps + 1, 3), dtype=np.float32)
    context[:, 0] = 1.0
    context[:, 1] = float(index) / 100.0
    context[:, 2] = np.linspace(0.01, 0.02, steps + 1, dtype=np.float32)
    semantics = np.full((steps + 1, 2, 4), float(index), dtype=np.float32)
    semantics[:, :, 0] += np.arange(steps + 1, dtype=np.float32)[:, None]
    return EpisodeFeatures(
        dataset_id="libero",
        dataset_index=0,
        episode_index=index,
        task_index=0,
        source_episode_sha256=f"{index + 1:064x}",
        model_actions=actions,
        proprio=np.full((steps + 1, 8), float(index), dtype=np.float32),
        gripper=np.concatenate(
            [np.zeros(4, dtype=np.float32), np.ones(steps + 1 - 4, dtype=np.float32)]
        ),
        context_keys=context,
        semantic_features=semantics,
    )


def _save_cache(
    root: Path,
    episode: EpisodeFeatures,
    *,
    split: str,
    catalog_hash: str,
) -> Path:
    path = root / f"episode-{episode.episode_index}.npz"
    save_episode_feature_cache(
        path,
        episode,
        metadata=FeatureCacheMetadata.for_episode(
            episode,
            split=split,
            **{**HASHES, "catalog_hash": catalog_hash},
        ),
    )
    return path


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    episodes = [_episode(index) for index in range(4)]
    _, _, catalog, audit_report = write_test_catalog_and_audit(
        tmp_path / "data-contract",
        [
            (episodes[0], "train"),
            (episodes[1], "train"),
            (episodes[2], "dev"),
            (episodes[3], "dev"),
        ],
    )
    train_paths = [
        _save_cache(
            tmp_path / "train", episodes[0], split="train", catalog_hash=catalog.content_sha256
        ),
        _save_cache(
            tmp_path / "train", episodes[1], split="train", catalog_hash=catalog.content_sha256
        ),
    ]
    train = load_feature_cache_collection(train_paths)
    bank, _, provenance = build_event_bank_from_collection(
        train,
        mining_config=EventMiningConfig(action_horizon=4),
        start_mode="uniform",
        data_binding=FeatureDataBinding(
            catalog_sha256=catalog.content_sha256,
            audit_report_sha256=str(audit_report["report_sha256"]),
            split="train",
        ),
    )
    bank_root = tmp_path / "bank"
    bank.save(
        bank_root,
        action_normalizer={
            "file_sha256": HASHES["normalizer_hash"],
            "contract": ACTION_SPACE_CONTRACT,
        },
        encoder={"file_sha256": HASHES["encoder_hash"]},
        camera_layout={"file_sha256": HASHES["camera_hash"]},
        provenance=provenance,
    )
    first_dev = _save_cache(
        tmp_path / "dev", episodes[2], split="dev", catalog_hash=catalog.content_sha256
    )
    second_dev = _save_cache(
        tmp_path / "dev", episodes[3], split="dev", catalog_hash=catalog.content_sha256
    )
    return bank_root, first_dev, second_dev


def _base_args(bank: Path, feature: Path, output: Path) -> list[str]:
    feature_args: list[str] = []
    for cache_path in sorted(feature.parent.glob("*.npz")):
        feature_args.extend(["--feature-cache", str(cache_path)])
    return [
        "--bank",
        str(bank),
        "--catalog",
        str(bank.parent / "data-contract" / "catalog.json"),
        "--audit-report",
        str(bank.parent / "data-contract" / "audit.json"),
        *feature_args,
        "--output",
        str(output),
        "--query-stride",
        "4",
        "--top-k",
        "1,2",
        "--arm-dims",
        "0,1",
        "--gripper-dims",
        "2",
    ]


def test_main_evaluates_each_k_and_atomically_writes_provenance_json(
    tmp_path: Path,
) -> None:
    bank, first_dev, second_dev = _artifacts(tmp_path)
    feature_list = tmp_path / "dev" / "features.txt"
    feature_list.write_text(
        "# relative cache path\n" + second_dev.name + "\n", encoding="utf-8"
    )
    output = tmp_path / "reports" / "oracle.json"
    args = _base_args(bank, first_dev, output)
    args.extend(
        [
            "--feature-list",
            str(feature_list),
            "--top-k",
            "2",
            "--top-k",
            "3",
            "--arm-loss",
            "huber",
            "--huber-delta",
            "0.25",
            "--arm-weight",
            "2.0",
            "--gripper-state-weight",
            "0.5",
            "--gripper-timing-weight",
            "0.75",
            "--gripper-threshold",
            "0.1",
        ]
    )

    assert main(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    collection = load_feature_cache_collection([first_dev, second_dev])

    assert payload["schema"] == "warm.oracle-evaluation"
    assert payload["version"] == 1
    assert payload["bank_manifest_sha256"] == sha256_file(bank / "manifest.json")
    loaded_bank = EventBank.load(bank)
    assert loaded_bank.manifest is not None
    assert payload["bank_content_sha256"] == canonical_event_bank_content_hash(
        loaded_bank.manifest.content_hashes
    )
    assert payload["query_corpus_sha256"] == collection.content_hash
    assert payload["query_split"] == "dev"
    assert payload["catalog_sha256"] == collection.contract.catalog_hash
    assert payload["feature_cache_count"] == 2
    assert payload["query_count"] == 4
    assert payload["top_k_values"] == [1, 2, 3]
    assert list(payload["reports"]) == ["1", "2", "3"]
    assert all(report["query_count"] == 4 for report in payload["reports"].values())
    assert payload["bank_summary"]["action_horizon"] == 4
    assert payload["bank_summary"]["action_dim"] == 3
    assert payload["action_distance_config"] == {
        "action_dim": 3,
        "arm_dims": [0, 1],
        "gripper_dims": [2],
        "arm_loss": "huber",
        "huber_delta": 0.25,
        "arm_weight": 2.0,
        "gripper_state_weight": 0.5,
        "gripper_timing_weight": 0.75,
        "gripper_threshold": 0.1,
    }
    assert payload["action_space_contract"] == ACTION_SPACE_CONTRACT
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
    assert not (output.parent / f".{output.name}.warm-oracle.lock").exists()


def test_main_refuses_a_competing_oracle_publisher(tmp_path: Path) -> None:
    bank, feature, _ = _artifacts(tmp_path)
    output = tmp_path / "reports" / "oracle.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.warm-oracle.lock"

    with artifact_claim(lock_path, purpose="competing oracle writer"):
        with pytest.raises(ArtifactAlreadyClaimedError):
            main(_base_args(bank, feature, output))
        assert lock_path.exists()
        assert not output.exists()

    assert not output.exists()


def test_main_uses_action_dimensions_from_immutable_bank_contract(
    tmp_path: Path,
) -> None:
    bank, feature, _ = _artifacts(tmp_path)
    output = tmp_path / "contract-derived.json"
    args = _base_args(bank, feature, output)
    for option in ("--arm-dims", "--gripper-dims"):
        index = args.index(option)
        del args[index : index + 2]

    assert main(args) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["action_distance_config"]["arm_dims"] == [0, 1]
    assert payload["action_distance_config"]["gripper_dims"] == [2]
    assert payload["action_distance_config"]["gripper_threshold"] == 0.1


@pytest.mark.parametrize(
    "arm_dims,gripper_dims",
    [
        ("0,1", "1"),  # overlap and missing dimension 2
        ("0", "2"),  # dimension 1 is not assigned
        ("0,1,3", "2"),  # out of range
        ("zero,1", "2"),  # not integers
    ],
)
def test_main_rejects_wrong_dimension_layouts(
    tmp_path: Path, arm_dims: str, gripper_dims: str
) -> None:
    bank, feature, _ = _artifacts(tmp_path)
    output = tmp_path / "invalid.json"
    args = _base_args(bank, feature, output)
    arm_index = args.index("--arm-dims") + 1
    gripper_index = args.index("--gripper-dims") + 1
    args[arm_index] = arm_dims
    args[gripper_index] = gripper_dims

    with pytest.raises(SystemExit) as exc_info:
        main(args)
    assert exc_info.value.code == 2
    assert not output.exists()


def test_main_rejects_empty_or_nonpositive_top_k(tmp_path: Path) -> None:
    for invalid in ("", "1,0", "-2"):
        output = tmp_path / f"invalid-{invalid.replace(',', '-') or 'empty'}.json"
        args = [
            "--bank",
            str(tmp_path / "bank"),
            "--feature-cache",
            str(tmp_path / "feature.npz"),
            "--output",
            str(output),
            "--query-stride",
            "4",
            "--top-k",
            invalid,
            "--arm-dims",
            "0,1",
            "--gripper-dims",
            "2",
        ]
        with pytest.raises(SystemExit) as exc_info:
            main(args)
        assert exc_info.value.code == 2
        assert not output.exists()


def test_main_never_overwrites_bank_or_feature_artifacts(tmp_path: Path) -> None:
    bank, feature, _ = _artifacts(tmp_path)
    protected = (bank / "manifest.json", feature)
    before = {path: sha256_file(path) for path in protected}

    for output in protected:
        args = _base_args(bank, feature, output)
        with pytest.raises(SystemExit) as exc_info:
            main(args)
        assert exc_info.value.code == 2

    assert {path: sha256_file(path) for path in protected} == before
