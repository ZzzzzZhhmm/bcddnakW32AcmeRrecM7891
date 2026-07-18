from __future__ import annotations

from json import dumps, loads
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy

import numpy as np
import pytest

from fastwam.memory.action_contract import ActionSpaceContract
from fastwam.memory.candidate_cache import canonical_event_bank_content_hash
from fastwam.memory.manifest import (
    sha256_canonical_json,
    sha256_file,
    sha256_path_tree,
)
from fastwam.models.warm.online_contract import WarmOnlineRunContract
from fastwam.models.warm.source_contract import WarmSourceRunContract
from fastwam.models.warm.training_attestation import (
    V1_SHARED_RECIPE_IGNORED_PATHS,
    WarmTrainingAttestation,
)
import scripts.build_warm_online_contract as online_cli


def _write(path: Path, value: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _m1_train_config() -> dict:
    shape_meta = {
        "images": [
            {"key": "image", "raw_shape": [3, 512, 512], "shape": [3, 224, 224]},
            {"key": "wrist_image", "raw_shape": [3, 512, 512], "shape": [3, 224, 224]},
        ],
        "action": [{"key": "default", "raw_shape": 7, "shape": 7}],
        "state": [{"key": "default", "raw_shape": 8, "shape": 8}],
    }
    transforms = [
        {"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"},
        {"_target_": "torchvision.transforms.Resize", "size": [224, 224]},
    ]
    return {
        "num_frames": 5,
        "video_size": [224, 448],
        "concat_multi_camera": "horizontal",
        "shape_meta": deepcopy(shape_meta),
        "processor": {
            "_target_": "fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor",
            "shape_meta": deepcopy(shape_meta),
            "num_obs_steps": 5,
            "num_output_cameras": 2,
            "action_output_dim": 7,
            "proprio_output_dim": 8,
            "delta_action_dim_mask": {"default": [True] * 6 + [False]},
            "action_state_transforms": None,
            "use_stepwise_action_norm": False,
            "norm_default_mode": "min/max",
            "norm_exception_mode": None,
            "action_state_merger": {
                "_target_": "fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign"
            },
            "train_transforms": deepcopy(transforms),
            "val_transforms": deepcopy(transforms),
        },
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], dict[str, Path]]:
    paths = {
        "bank": tmp_path / "bank",
        "source": tmp_path / "source.json",
        "validation_source": tmp_path / "validation_source.json",
        "checkpoint": _write(tmp_path / "warm.pt", b"warm-checkpoint"),
        "training_attestation": tmp_path / "warm.training.json",
        "base_checkpoint": _write(tmp_path / "fastwam.pt", b"fastwam-base"),
        "normalizer": _write(tmp_path / "normalizer.json", b'{"normalizer":"v1"}\n'),
        "encoder": tmp_path / "encoder.json",
        "camera": _write(tmp_path / "camera.json", b'{"camera":"fixed"}\n'),
        "data_config": tmp_path / "m1-data.json",
        "dino": tmp_path / "dino",
        "stats": _write(tmp_path / "stats.json", b'{"stats":"v1"}\n'),
        "catalog": _write(tmp_path / "catalog.json", b'{"catalog":"v1"}\n'),
        "audit": _write(tmp_path / "audit.json", b'{"audit":"v1"}\n'),
        "config": tmp_path / "resolved.json",
        "vae": _write(tmp_path / "vae.pt", b"vae"),
        "text": tmp_path / "text",
        "tokenizer": tmp_path / "tokenizer",
        "states": tmp_path / "states.npy",
        "bddl": _write(tmp_path / "task.bddl", b"(define (problem exact))\n"),
        "output": tmp_path / "online.json",
    }
    paths["bank"].mkdir()
    _write(paths["bank"] / "manifest.json", b'{"manifest":"v1"}\n')
    paths["dino"].mkdir()
    _write(paths["dino"] / "model.safetensors", b"dino")
    paths["text"].mkdir()
    _write(paths["text"] / "model.safetensors", b"text")
    paths["tokenizer"].mkdir()
    _write(paths["tokenizer"] / "tokenizer.json", b'{}\n')
    np.save(paths["states"], np.arange(12, dtype=np.float32).reshape(2, 6))
    train_config = _m1_train_config()
    paths["data_config"].write_text(
        dumps({"train": train_config}), encoding="utf-8"
    )

    dino_sha, dino_count = sha256_path_tree(paths["dino"])
    paths["encoder"].write_text(
        dumps(
            {
                "schema": "warm.feature-encoder",
                "version": 2,
                "official_complete": True,
                "runtime": {"python": "test", "device": "cuda"},
                "data_config_sha256": sha256_file(paths["data_config"]),
                "output_dtype": "float32",
                "compute": {
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "dino_batch_size": 1,
                    "vae_batch_size": None,
                },
                "dino": {
                    "model_id": "facebook/dinov2-small",
                    "revision": "abcdef0",
                    "checkpoint_tree_sha256": dino_sha,
                    "checkpoint_file_count": dino_count,
                    "hidden_size": 384,
                    "patch_size": [14, 14],
                    "patch_grid_size": [16, 16],
                    "register_token_count": 0,
                    "image_size": [224, 224],
                    "image_mean": [0.485, 0.456, 0.406],
                    "image_std": [0.229, 0.224, 0.225],
                    "semantic_pool": "adaptive_avg_pool_2x2_row_major",
                    "resize_in_encoder": False,
                },
                "context": {
                    "mode": "task-conditioned",
                    "task_vocabulary": ["put the bowl on the plate"],
                    "visual_task_energy": [0.5, 0.5],
                },
            }
        ),
        encoding="utf-8",
    )
    paths["camera"].write_text(
        dumps(
            {
                "schema": "warm.camera-layout",
                "version": 1,
                "source_camera_keys": [
                    "observation.images.image",
                    "observation.images.wrist_image",
                ],
                "processor_camera_mapping": {
                    "observation.images.image": "image",
                    "observation.images.wrist_image": "wrist_image",
                },
                "semantic_camera": "observation.images.image",
                "concat_mode": "horizontal",
                "per_camera_size": [224, 224],
                "decoded_range": [0.0, 1.0],
                "vae_model_range": [-1.0, 1.0],
                "baseline_quantization": (
                    "validated_0_1_times_255_to_uint8"
                ),
            }
        ),
        encoding="utf-8",
    )
    encoder_sha = sha256_file(paths["encoder"])
    camera_sha = sha256_file(paths["camera"])
    stats_sha = sha256_file(paths["stats"])
    action_contract = ActionSpaceContract(
        action_dim=7,
        arm_dims=(0, 1, 2, 3, 4, 5),
        gripper_dims=(6,),
        gripper_threshold=0.0,
        normalization_mode="min/max",
        normalization_stats_sha256=stats_sha,
        control_mode="delta_eef_gripper",
        embodiment="libero_panda",
    )
    paths["normalizer"].write_text(
        dumps(action_contract.to_dict()), encoding="utf-8"
    )
    catalog_hash = "1" * 64
    audit_hash = "2" * 64
    content_hashes = {"events.npy": "4" * 64, "payload.npz": "5" * 64}
    manifest = SimpleNamespace(
        content_hashes=content_hashes,
        encoder={
            "file_sha256": encoder_sha,
            "contract": loads(paths["encoder"].read_text(encoding="utf-8")),
        },
        camera_layout={
            "file_sha256": camera_sha,
            "contract": loads(paths["camera"].read_text(encoding="utf-8")),
        },
        action_normalizer={
            "file_sha256": sha256_file(paths["normalizer"]),
            "contract": action_contract.to_dict(),
        },
        provenance={
            "data_binding": {
                "catalog_sha256": catalog_hash,
                "audit_report_sha256": audit_hash,
                "split": "train",
            }
        },
    )
    fake_bank = SimpleNamespace(manifest=manifest)
    monkeypatch.setattr(online_cli.EventBank, "load", lambda _path: fake_bank)
    monkeypatch.setattr(
        online_cli,
        "validate_warm_v1_bank",
        lambda *_args, **_kwargs: SimpleNamespace(action_horizon=4, action_dim=7),
    )
    monkeypatch.setattr(
        online_cli.EpisodeCatalog,
        "load",
        lambda _path: SimpleNamespace(content_sha256=catalog_hash),
    )
    monkeypatch.setattr(
        online_cli,
        "load_audit_report",
        lambda _path: SimpleNamespace(
            report_sha256=audit_hash,
            catalog_sha256=catalog_hash,
        ),
    )
    monkeypatch.setattr(online_cli, "_git_identity", lambda _repo: ("a" * 40, False))

    source = WarmSourceRunContract(
        bank_manifest_sha256=sha256_file(paths["bank"] / "manifest.json"),
        bank_content_sha256=canonical_event_bank_content_hash(content_hashes),
        candidate_manifest_sha256="6" * 64,
        query_corpus_sha256="7" * 64,
        catalog_sha256=catalog_hash,
        audit_sha256=audit_hash,
        normalization_stats_sha256=stats_sha,
        action_space_contract_sha256=sha256_canonical_json(action_contract.to_dict()),
        base_checkpoint_sha256=sha256_file(paths["base_checkpoint"]),
        query_split="train",
        global_sample_stride=1,
        action_horizon=4,
        action_dim=7,
    )
    paths["source"].write_text(dumps(source.to_dict()), encoding="utf-8")
    validation_source_value = source.to_dict()
    validation_source_value.update(
        {
            "candidate_manifest_sha256": "9" * 64,
            "query_corpus_sha256": "a" * 64,
            "query_split": "dev",
        }
    )
    validation_source = WarmSourceRunContract.from_dict(validation_source_value)
    paths["validation_source"].write_text(
        dumps(validation_source.to_dict()), encoding="utf-8"
    )
    training_attestation = WarmTrainingAttestation.from_dict(
        {
            "schema": "warm.training-attestation",
            "version": 1,
            "checkpoint_sha256": sha256_file(paths["checkpoint"]),
            "checkpoint_step": 100,
            "source_policy": "fixed_context_top1",
            "resolved_train_config_sha256": "b" * 64,
            "shared_recipe_sha256": "c" * 64,
            "shared_recipe_ignored_paths": list(V1_SHARED_RECIPE_IGNORED_PATHS),
            "root_seed": 17,
            "actual_global_step": 100,
            "actual_max_steps": 100,
            "optimizer_name": "AdamW",
            "optimizer_learning_rate": 1.0e-4,
            "optimizer_weight_decay": 0.01,
            "optimizer_beta1": 0.9,
            "optimizer_beta2": 0.95,
            "optimizer_epsilon": 1.0e-8,
            "optimizer_amsgrad": False,
            "optimizer_wrapper_chain": ["torch.optim.adamw.AdamW"],
            "scheduler_type": "cosine",
            "scheduler_total_steps": 100,
            "scheduler_warmup_steps": 5,
            "scheduler_min_learning_rate": 1.0e-6,
            "scheduler_wrapper_chain": [
                "torch.optim.lr_scheduler.SequentialLR"
            ],
            "per_device_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "world_size": 1,
            "effective_batch_size": 4,
            "mixed_precision": "bf16",
            "training_runtime": {
                "python_version": "3.11.9",
                "platform": "Linux-test",
                "accelerate_version": "1.7.0",
                "deepspeed_version": None,
                "torch_version": "2.7.0",
                "torch_cuda_version": None,
                "cudnn_version": None,
                "cuda_available": False,
                "gpu_count": 0,
                "gpu_devices": [],
                "distributed_type": "no",
                "deepspeed_config": None,
                "deepspeed_zero_stage": None,
                "float32_matmul_precision": "highest",
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": False,
                "deterministic_algorithms_enabled": False,
                "cublas_workspace_config": None,
            },
            "train_source_contract_sha256": source.sha256,
            "dev_source_contract_sha256": validation_source.sha256,
            "base_checkpoint_sha256": source.base_checkpoint_sha256,
            "git_commit": "a" * 40,
        }
    )
    paths["training_attestation"].write_bytes(training_attestation.encode())
    resolved = {
        "seed": 17,
        "ckpt": str(paths["checkpoint"]),
        "model": {
            "source_policy": "fixed_context_top1",
            "memory_sigma": 0.2,
            "run_contract_path": str(paths["source"]),
            "validation_run_contract_path": str(paths["validation_source"]),
            "base_checkpoint_path": str(paths["base_checkpoint"]),
        },
        "data": {
            "train": {
                **deepcopy(train_config),
                "pretrained_norm_stats": str(paths["stats"]),
            }
        },
        "EVALUATION": {
            "task_suite_name": "libero_10",
            "task_id": 2,
            "action_horizon": None,
            "dataset_stats_path": str(paths["stats"]),
            "visualize_future_video": False,
            "use_action_ensembler": False,
            "warm_online": {
                "enabled": True,
                "contract_path": str(paths["output"]),
                "training_attestation_path": str(paths["training_attestation"]),
                "training_run_contract_path": str(paths["source"]),
                "validation_run_contract_path": str(paths["validation_source"]),
                "base_checkpoint_path": str(paths["base_checkpoint"]),
                "bank_directory": str(paths["bank"]),
                "normalizer_contract_path": str(paths["normalizer"]),
                "encoder_contract_path": str(paths["encoder"]),
                "camera_contract_path": str(paths["camera"]),
                "m1_data_config_path": str(paths["data_config"]),
                "pair_contract_path": str(tmp_path / "pair.json"),
                "parity_report_path": str(tmp_path / "parity.json"),
                "dino_checkpoint_path": str(paths["dino"]),
                "catalog_path": str(paths["catalog"]),
                "audit_report_path": str(paths["audit"]),
                "evaluation_namespace": "shared-eval-v1",
                "top_k": 3,
                "dino_device": "cuda",
            },
        },
    }
    paths["config"].write_text(dumps(resolved), encoding="utf-8")

    argv = [
        "--training-run-contract", str(paths["source"]),
        "--validation-run-contract", str(paths["validation_source"]),
        "--warm-checkpoint", str(paths["checkpoint"]),
        "--training-attestation", str(paths["training_attestation"]),
        "--bank", str(paths["bank"]),
        "--normalizer-contract", str(paths["normalizer"]),
        "--encoder-contract", str(paths["encoder"]),
        "--camera-contract", str(paths["camera"]),
        "--data-config", str(paths["data_config"]),
        "--dino-checkpoint", str(paths["dino"]),
        "--normalization-stats", str(paths["stats"]),
        "--catalog", str(paths["catalog"]),
        "--audit-report", str(paths["audit"]),
        "--resolved-eval-config", str(paths["config"]),
        "--vae-checkpoint", str(paths["vae"]),
        "--text-encoder", str(paths["text"]),
        "--tokenizer", str(paths["tokenizer"]),
        "--evaluation-namespace", "shared-eval-v1",
        "--task-suite", "libero_10",
        "--task-id", "2",
        "--task-description", "put the bowl on the plate",
        "--initial-states", str(paths["states"]),
        "--bddl", str(paths["bddl"]),
        "--root-seed", "17",
        "--top-k", "3",
        "--source-policy", "fixed_context_top1",
        "--memory-sigma", "0.2",
        "--action-horizon", "4",
        "--action-dim", "7",
        "--output", str(paths["output"]),
    ]
    return argv, paths


def test_cli_binds_every_online_identity_and_publishes_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    assert online_cli.main(argv) == 0

    written = loads(paths["output"].read_text(encoding="utf-8"))
    contract = WarmOnlineRunContract.from_dict(written)
    summary = loads(capsys.readouterr().out)
    assert summary["online_run_contract_sha256"] == contract.sha256
    assert contract.warm_checkpoint_sha256 == sha256_file(paths["checkpoint"])
    validation_source = WarmSourceRunContract.from_dict(
        loads(paths["validation_source"].read_text(encoding="utf-8"))
    )
    assert contract.validation_run_contract_sha256 == validation_source.sha256
    assert contract.resolved_eval_config_sha256 == sha256_canonical_json(
        loads(paths["config"].read_text(encoding="utf-8"))
    )
    assert contract.task_description == "put the bowl on the plate"
    assert contract.root_seed == 17
    assert contract.top_k == 3
    assert not list(paths["output"].parent.glob(".*.tmp"))


def test_full_retrospection_builds_one_checkpoint_contract_without_pair_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    config = loads(paths["config"].read_text(encoding="utf-8"))
    online = config["EVALUATION"]["warm_online"]
    online["mode"] = "full_retrospection"
    online.pop("pair_contract_path")
    online.pop("parity_report_path")
    paths["config"].write_text(dumps(config), encoding="utf-8")

    assert online_cli.main(argv) == 0
    contract = WarmOnlineRunContract.from_dict(
        loads(paths["output"].read_text(encoding="utf-8"))
    )
    assert contract.source_policy == "fixed_context_top1"
    assert contract.warm_checkpoint_sha256 == sha256_file(paths["checkpoint"])


def test_cli_rejects_config_that_disagrees_with_explicit_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    config = loads(paths["config"].read_text(encoding="utf-8"))
    config["seed"] = 18
    paths["config"].write_text(dumps(config), encoding="utf-8")

    with pytest.raises(online_cli.OnlineContractBuildError, match="root seed"):
        online_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_rejects_resolved_config_with_unbound_artifact_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    config = loads(paths["config"].read_text(encoding="utf-8"))
    config["EVALUATION"]["warm_online"]["bank_directory"] = str(
        tmp_path / "different-bank"
    )
    paths["config"].write_text(dumps(config), encoding="utf-8")

    with pytest.raises(
        online_cli.OnlineContractBuildError,
        match="event bank path disagrees",
    ):
        online_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_rejects_dino_device_that_differs_from_encoder_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    config = loads(paths["config"].read_text(encoding="utf-8"))
    config["EVALUATION"]["warm_online"]["dino_device"] = "cuda:0"
    paths["config"].write_text(dumps(config), encoding="utf-8")

    with pytest.raises(online_cli.OnlineContractBuildError, match="DINO device"):
        online_cli.main(argv)
    assert not paths["output"].exists()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("EVALUATION", "warm_online", "enabled"), False, "must enable"),
        (
            ("EVALUATION", "visualize_future_video"),
            True,
            "disable future-video",
        ),
        (
            ("EVALUATION", "use_action_ensembler"),
            True,
            "disable action ensembling",
        ),
    ],
)
def test_cli_rejects_noncanonical_online_eval_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: bool,
    message: str,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    config = loads(paths["config"].read_text(encoding="utf-8"))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    paths["config"].write_text(dumps(config), encoding="utf-8")

    with pytest.raises(online_cli.OnlineContractBuildError, match=message):
        online_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_refuses_publish_if_input_changes_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    original = online_cli._build_contract

    def _build_then_mutate(args):
        contract = original(args)
        paths["checkpoint"].write_bytes(b"changed")
        return contract

    monkeypatch.setattr(online_cli, "_build_contract", _build_then_mutate)
    with pytest.raises(online_cli.OnlineContractBuildError, match="checkpoint changed"):
        online_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_requires_clean_git_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(online_cli, "_git_identity", lambda _repo: ("b" * 40, True))
    with pytest.raises(online_cli.OnlineContractBuildError, match="clean Git tree"):
        online_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_rejects_dev_contract_as_online_training_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    value = loads(paths["source"].read_text(encoding="utf-8"))
    value["query_split"] = "dev"
    paths["source"].write_text(dumps(value), encoding="utf-8")

    with pytest.raises(online_cli.OnlineContractBuildError, match="train event-bank"):
        online_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_requires_independent_dev_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    train = loads(paths["source"].read_text(encoding="utf-8"))
    train["query_split"] = "dev"
    paths["validation_source"].write_text(dumps(train), encoding="utf-8")

    with pytest.raises(online_cli.OnlineContractBuildError, match="independent dev"):
        online_cli.main(argv)
    assert not paths["output"].exists()
