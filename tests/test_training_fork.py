from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from fastwam.models.warm import training_fork
from fastwam.models.warm.training_attestation import (
    TrainingAttestationError,
    training_config_hashes,
)


def _config(*, allowlist):
    return {
        "model": {
            "_target_": "fastwam.models.warm.WarmRetrospectionFastWAM",
            "action_dit_config": {"dim": 1024, "num_layers": 30},
            "retrospection": {
                "action_dim": 14,
                "action_horizon": 32,
                "episode_action_chunk_size": 4,
                "semantic_dim": 768,
            },
        },
        "data": {
            "train": {
                "shape_meta": {
                    "images": [
                        {"key": "cam_high", "raw_shape": [3, 240, 320]},
                        {"key": "cam_left_wrist", "raw_shape": [3, 240, 320]},
                        {"key": "cam_right_wrist", "raw_shape": [3, 240, 320]},
                    ]
                },
                "num_frames": 33,
                "action_video_freq_ratio": 1,
                "video_size": [384, 320],
                "concat_multi_camera": "robotwin",
                "episode_task_allowlist": allowlist,
                "processor": {
                    "num_output_cameras": 3,
                    "action_output_dim": 14,
                    "proprio_output_dim": 14,
                },
            }
        },
        "wandb": {"name": "ignored", "group": "ignored"},
    }


def test_fork_manifest_binds_shared_parent_and_single_task_child(monkeypatch) -> None:
    parent = _config(allowlist=None)
    child = _config(allowlist=["blocks_ranking_try"])
    parent_hash = training_config_hashes(parent)[0]
    fake_attestation = SimpleNamespace(
        checkpoint_sha256="1" * 64,
        checkpoint_step=30000,
        git_commit="2" * 40,
        resolved_train_config_sha256=parent_hash,
    )
    monkeypatch.setattr(
        training_fork, "verify_training_attestation", lambda *_: fake_attestation
    )
    monkeypatch.setattr(training_fork, "sha256_file", lambda *_: "3" * 64)

    manifest = training_fork.build_training_fork_manifest(
        parent_checkpoint="shared.pt",
        parent_config=parent,
        child_config=child,
        fork_reason="shared to blocks specialist",
    )
    assert manifest["parent_checkpoint_step"] == 30000
    assert manifest["parent_resolved_train_config_sha256"] == parent_hash
    assert manifest["child_resolved_train_config_sha256"] == training_config_hashes(
        child
    )[0]
    assert training_fork.validate_training_fork_manifest(manifest) == manifest


def test_fork_rejects_task_specific_parent_or_tensor_contract_drift(monkeypatch) -> None:
    parent = _config(allowlist=["blocks_ranking_try"])
    child = _config(allowlist=["blocks_ranking_try"])
    fake_attestation = SimpleNamespace(
        checkpoint_sha256="1" * 64,
        checkpoint_step=30000,
        git_commit="2" * 40,
        resolved_train_config_sha256=training_config_hashes(parent)[0],
    )
    monkeypatch.setattr(
        training_fork, "verify_training_attestation", lambda *_: fake_attestation
    )
    monkeypatch.setattr(training_fork, "sha256_file", lambda *_: "3" * 64)
    with pytest.raises(TrainingAttestationError, match="task-shared"):
        training_fork.build_training_fork_manifest(
            parent_checkpoint="shared.pt",
            parent_config=parent,
            child_config=child,
            fork_reason="invalid parent",
        )

    parent = _config(allowlist=None)
    fake_attestation.resolved_train_config_sha256 = training_config_hashes(parent)[0]
    child = deepcopy(child)
    child["model"]["retrospection"]["episode_action_chunk_size"] = 10
    with pytest.raises(TrainingAttestationError, match="episode_action_chunk_size=4"):
        training_fork.build_training_fork_manifest(
            parent_checkpoint="shared.pt",
            parent_config=parent,
            child_config=child,
            fork_reason="incompatible child",
        )
