from __future__ import annotations

from contextlib import AbstractContextManager
from copy import deepcopy
from json import dumps, loads
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest

from fastwam.memory.manifest import sha256_canonical_json, sha256_file
from fastwam.models.warm.online_contract import (
    ONLINE_RETRIEVAL_IMPLEMENTATION,
    WarmOnlineRunContract,
)
from fastwam.models.warm.online_pair_contract import (
    ALLOWED_CONFIG_DIFFERENCE_PATHS,
    ONLINE_PAIR_KIND,
    WarmOnlinePairContract,
)
from fastwam.models.warm.training_attestation import (
    SHARED_RECIPE_IGNORED_PATHS,
    WarmTrainingAttestation,
)
import scripts.build_warm_online_pair_contract as pair_cli


def _config(
    *,
    policy: str,
    contract_path: Path,
    pair_contract_path: Path,
    checkpoint: str,
    training_attestation: Path,
    output_dir: str,
) -> dict[str, Any]:
    return {
        "seed": 17,
        "ckpt": checkpoint,
        "gpu_id": 0,
        "model": {
            "source_policy": policy,
            "memory_sigma": 0.2,
            "run_contract_path": "/artifacts/train-source.json",
            "validation_run_contract_path": "/artifacts/dev-source.json",
            "base_checkpoint_path": "/artifacts/fastwam.pt",
        },
        "data": {
            "train": {
                "num_frames": 5,
                "processor": {"action_output_dim": 3},
                "pretrained_norm_stats": "/artifacts/stats.json",
            }
        },
        "EVALUATION": {
            "task_suite_name": "libero_10",
            "task_id": 2,
            "num_trials": 50,
            "output_dir": output_dir,
            "num_steps_wait": 30,
            "replan_steps": 4,
            "binarize_gripper": True,
            "use_action_ensembler": False,
            "visualize_future_video": False,
            "action_horizon": 4,
            "num_inference_steps": 10,
            "sigma_shift": None,
            "text_cfg_scale": 1.0,
            "negative_prompt": "",
            "rand_device": "cpu",
            "tiled": False,
            "dataset_stats_path": "/artifacts/stats.json",
            "warm_online": {
                "enabled": True,
                "contract_path": str(contract_path.resolve()),
                "pair_contract_path": str(pair_contract_path.resolve()),
                "parity_report_path": str(
                    (pair_contract_path.parent / "parity.json").resolve()
                ),
                "training_attestation_path": str(training_attestation.resolve()),
                "training_run_contract_path": "/artifacts/train-source.json",
                "validation_run_contract_path": "/artifacts/dev-source.json",
                "base_checkpoint_path": "/artifacts/fastwam.pt",
                "bank_directory": "/artifacts/bank",
                "normalizer_contract_path": "/artifacts/action.json",
                "encoder_contract_path": "/artifacts/encoder.json",
                "camera_contract_path": "/artifacts/camera.json",
                "dino_checkpoint_path": "/artifacts/dino",
                "catalog_path": "/artifacts/catalog.json",
                "audit_report_path": "/artifacts/audit.json",
                "evaluation_namespace": "paper-m2-libero-v1",
                "top_k": 32,
                "dino_device": "cuda",
                "dino_batch_size": 1,
            },
        },
    }


def _online_contract(
    *,
    policy: str,
    checkpoint_digest: str,
    training_attestation_digest: str,
    config: dict[str, Any],
    shared_overrides: dict[str, Any] | None = None,
) -> WarmOnlineRunContract:
    value: dict[str, Any] = {
        "training_run_contract_sha256": "1" * 64,
        "validation_run_contract_sha256": "2" * 64,
        "warm_checkpoint_sha256": checkpoint_digest,
        "training_attestation_sha256": training_attestation_digest,
        "shared_training_recipe_sha256": "0" * 64,
        "training_runtime_sha256": "1" * 64,
        "bank_manifest_sha256": "3" * 64,
        "bank_content_sha256": "4" * 64,
        "encoder_contract_sha256": "5" * 64,
        "encoder_runtime_sha256": "0" * 64,
        "camera_contract_sha256": "6" * 64,
        "m1_data_config_sha256": "f" * 64,
        "dino_checkpoint_tree_sha256": "7" * 64,
        "dino_checkpoint_file_count": 3,
        "normalization_stats_sha256": "8" * 64,
        "action_space_contract_sha256": "9" * 64,
        "catalog_sha256": "a" * 64,
        "audit_sha256": "b" * 64,
        "resolved_eval_config_sha256": sha256_canonical_json(config),
        "vae_checkpoint_sha256": "c" * 64,
        "text_encoder_tree_sha256": "d" * 64,
        "tokenizer_tree_sha256": "e" * 64,
        "evaluation_namespace_sha256": "f" * 64,
        "task_suite": "libero_10",
        "task_id": 2,
        "task_description": "put the bowl on the plate",
        "root_seed": 17,
        "initial_states_sha256": "0" * 64,
        "bddl_sha256": "1" * 64,
        "retrieval_implementation": ONLINE_RETRIEVAL_IMPLEMENTATION,
        "top_k": 32,
        "source_policy": policy,
        "memory_sigma": 0.2,
        "action_horizon": 4,
        "action_dim": 3,
        "git_commit": "a" * 40,
        "git_dirty": False,
    }
    if shared_overrides:
        value.update(shared_overrides)
    return WarmOnlineRunContract(**value)


def _training_runtime() -> dict[str, Any]:
    return {
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
    }


def _training_attestation(
    *, policy: str, checkpoint: Path
) -> WarmTrainingAttestation:
    return WarmTrainingAttestation.from_dict(
        {
            "schema": "warm.training-attestation",
            "version": 1,
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_step": 100,
            "source_policy": policy,
            "resolved_train_config_sha256": (
                "2" * 64 if policy == "fixed_context_top1" else "3" * 64
            ),
            "shared_recipe_sha256": "0" * 64,
            "shared_recipe_ignored_paths": list(SHARED_RECIPE_IGNORED_PATHS),
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
            "training_runtime": _training_runtime(),
            "train_source_contract_sha256": "1" * 64,
            "dev_source_contract_sha256": "2" * 64,
            "base_checkpoint_sha256": "4" * 64,
            "git_commit": "a" * 40,
        }
    )


def _parity_report(contract: WarmOnlineRunContract) -> dict[str, Any]:
    return {
        "schema": "warm.online-retrieval-parity-report",
        "version": 1,
        "status": "pass",
        "scope": "catalog-bound-dev-task",
        "task": {
            "suite": contract.task_suite,
            "task_id": contract.task_id,
            "description": contract.task_description,
        },
        "numeric_comparison": {
            "mode": "exact",
            "rtol": 0.0,
            "atol": 0.0,
            "score_exactness": "float64-online-to-float32-cache-roundtrip",
        },
        "contracts": {
            "training_run_contract_sha256": contract.training_run_contract_sha256,
            "validation_run_contract_sha256": contract.validation_run_contract_sha256,
            "online_run_contract_sha256": contract.sha256,
            "evaluation_namespace_sha256": contract.evaluation_namespace_sha256,
            "encoder_contract_sha256": contract.encoder_contract_sha256,
            "camera_contract_sha256": contract.camera_contract_sha256,
            "normalization_stats_sha256": contract.normalization_stats_sha256,
            "action_space_contract_sha256": contract.action_space_contract_sha256,
            "catalog_sha256": contract.catalog_sha256,
            "audit_sha256": contract.audit_sha256,
        },
        "artifacts": {
            "bank_manifest_sha256": contract.bank_manifest_sha256,
            "bank_content_sha256": contract.bank_content_sha256,
            "candidate_manifest_sha256": "1" * 64,
            "candidate_payload_sha256": "2" * 64,
            "dev_query_corpus_sha256": "3" * 64,
            "dev_feature_artifact_set_sha256": "4" * 64,
            "dataset_metadata_set_sha256": "5" * 64,
            "raw_dev_artifact_set_sha256": "6" * 64,
            "raw_dev_artifact_count": 1,
            "dino_checkpoint_tree_sha256": contract.dino_checkpoint_tree_sha256,
            "dino_checkpoint_file_count": contract.dino_checkpoint_file_count,
            "resolved_eval_config_sha256": contract.resolved_eval_config_sha256,
            "resolved_eval_config_file_sha256": "7" * 64,
            "data_config_sha256": contract.m1_data_config_sha256,
        },
        "counts": {
            "dev_episode_count": 1,
            "task_episode_count": 1,
            "query_count": 1,
            "candidate_slot_count": 2,
            "valid_candidate_count": 2,
            "exact_context_key_count": 1,
            "exact_score_roundtrip_count": 1,
            "max_context_key_abs_error": 0.0,
            "max_cosine_score_abs_error": 0.0,
            "parity_transcript_sha256": "8" * 64,
        },
        "implementation": {
            "retriever": "exact_cosine_frozen_dino_v1",
            "offline_candidate": "exact_cosine_v1",
            "search_domain": "complete_event_bank",
            "query_stride": 1,
            "top_k": contract.top_k,
            "git_commit": contract.git_commit,
            "git_dirty": False,
            "device": "cuda",
            "encoder_runtime_sha256": contract.encoder_runtime_sha256,
        },
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], dict[str, Path]]:
    paths = {
        "fixed_contract": tmp_path / "fixed-online.json",
        "null_contract": tmp_path / "null-online.json",
        "fixed_config": tmp_path / "fixed-resolved.json",
        "null_config": tmp_path / "null-resolved.json",
        "output": tmp_path / "fixed-null-pair.json",
        "parity": tmp_path / "parity.json",
        "fixed_checkpoint": tmp_path / "warm-fixed.pt",
        "null_checkpoint": tmp_path / "warm-null.pt",
        "fixed_attestation": tmp_path / "warm-fixed.training.json",
        "null_attestation": tmp_path / "warm-null.training.json",
    }
    paths["fixed_checkpoint"].write_bytes(b"fixed checkpoint")
    paths["null_checkpoint"].write_bytes(b"null checkpoint")
    fixed_training = _training_attestation(
        policy="fixed_context_top1", checkpoint=paths["fixed_checkpoint"]
    )
    null_training = _training_attestation(
        policy="gaussian_null", checkpoint=paths["null_checkpoint"]
    )
    paths["fixed_attestation"].write_bytes(fixed_training.encode())
    paths["null_attestation"].write_bytes(null_training.encode())
    fixed_config = _config(
        policy="fixed_context_top1",
        contract_path=paths["fixed_contract"],
        pair_contract_path=paths["output"],
        checkpoint=str(paths["fixed_checkpoint"].resolve()),
        training_attestation=paths["fixed_attestation"],
        output_dir="/results/fixed",
    )
    null_config = _config(
        policy="gaussian_null",
        contract_path=paths["null_contract"],
        pair_contract_path=paths["output"],
        checkpoint=str(paths["null_checkpoint"].resolve()),
        training_attestation=paths["null_attestation"],
        output_dir="/results/null",
    )
    fixed_contract = _online_contract(
        policy="fixed_context_top1",
        checkpoint_digest=fixed_training.checkpoint_sha256,
        training_attestation_digest=sha256_file(paths["fixed_attestation"]),
        config=fixed_config,
        shared_overrides={
            "shared_training_recipe_sha256": fixed_training.shared_recipe_sha256,
            "training_runtime_sha256": fixed_training.training_runtime_sha256,
        },
    )
    null_contract = _online_contract(
        policy="gaussian_null",
        checkpoint_digest=null_training.checkpoint_sha256,
        training_attestation_digest=sha256_file(paths["null_attestation"]),
        config=null_config,
        shared_overrides={
            "shared_training_recipe_sha256": null_training.shared_recipe_sha256,
            "training_runtime_sha256": null_training.training_runtime_sha256,
        },
    )
    paths["fixed_config"].write_text(dumps(fixed_config), encoding="utf-8")
    paths["null_config"].write_text(dumps(null_config), encoding="utf-8")
    paths["fixed_contract"].write_text(
        dumps(fixed_contract.to_dict()), encoding="utf-8"
    )
    paths["null_contract"].write_text(
        dumps(null_contract.to_dict()), encoding="utf-8"
    )
    paths["parity"].write_text(
        dumps(_parity_report(fixed_contract)), encoding="utf-8"
    )
    monkeypatch.setattr(pair_cli, "_git_identity", lambda _repo: ("a" * 40, False))
    argv = [
        "--fixed-online-contract",
        str(paths["fixed_contract"]),
        "--fixed-resolved-eval-config",
        str(paths["fixed_config"]),
        "--fixed-training-attestation",
        str(paths["fixed_attestation"]),
        "--gaussian-null-online-contract",
        str(paths["null_contract"]),
        "--gaussian-null-resolved-eval-config",
        str(paths["null_config"]),
        "--gaussian-null-training-attestation",
        str(paths["null_attestation"]),
        "--parity-report",
        str(paths["parity"]),
        "--output",
        str(paths["output"]),
    ]
    return argv, paths


def _rewrite_contract_config_digest(contract_path: Path, config_path: Path) -> None:
    value = loads(contract_path.read_text(encoding="utf-8"))
    config = loads(config_path.read_text(encoding="utf-8"))
    value["resolved_eval_config_sha256"] = sha256_canonical_json(config)
    contract = WarmOnlineRunContract.from_dict(value)
    contract_path.write_text(dumps(contract.to_dict()), encoding="utf-8")


def test_cli_builds_deterministic_policy_checkpoint_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    assert pair_cli.main(argv) == 0

    pair = WarmOnlinePairContract.from_dict(
        loads(paths["output"].read_text(encoding="utf-8"))
    )
    summary = loads(capsys.readouterr().out)
    assert summary["online_pair_contract_sha256"] == pair.sha256
    assert pair.comparison_kind == ONLINE_PAIR_KIND
    assert pair.observed_config_difference_paths == ALLOWED_CONFIG_DIFFERENCE_PATHS
    assert pair.fixed_warm_checkpoint_sha256 == sha256_file(paths["fixed_checkpoint"])
    assert pair.gaussian_null_warm_checkpoint_sha256 == sha256_file(
        paths["null_checkpoint"]
    )
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob("*.warm-artifact.lock"))


def test_cli_rejects_training_runtime_or_optimizer_fairness_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    value = loads(paths["null_attestation"].read_text(encoding="utf-8"))
    value["optimizer_learning_rate"] = 2.0e-4
    changed = WarmTrainingAttestation.from_dict(value)
    paths["null_attestation"].write_bytes(changed.encode())
    online = loads(paths["null_contract"].read_text(encoding="utf-8"))
    online["training_attestation_sha256"] = sha256_file(
        paths["null_attestation"]
    )
    paths["null_contract"].write_text(
        dumps(WarmOnlineRunContract.from_dict(online).to_dict()),
        encoding="utf-8",
    )

    with pytest.raises(pair_cli.OnlinePairBuildError, match="fairness-critical"):
        pair_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_rejects_pair_path_not_equal_to_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    wrong = str((tmp_path / "wrong-pair.json").resolve())
    for config_key, contract_key in (
        ("fixed_config", "fixed_contract"),
        ("null_config", "null_contract"),
    ):
        config = loads(paths[config_key].read_text(encoding="utf-8"))
        config["EVALUATION"]["warm_online"]["pair_contract_path"] = wrong
        paths[config_key].write_text(dumps(config), encoding="utf-8")
        _rewrite_contract_config_digest(paths[contract_key], paths[config_key])

    with pytest.raises(pair_cli.OnlinePairBuildError, match="different pair contract"):
        pair_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_rejects_pair_path_values_that_are_not_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("WARM_PAIR_OUTPUT", str(paths["output"].resolve()))
    config = loads(paths["null_config"].read_text(encoding="utf-8"))
    config["EVALUATION"]["warm_online"][
        "pair_contract_path"
    ] = "$WARM_PAIR_OUTPUT"
    paths["null_config"].write_text(dumps(config), encoding="utf-8")
    _rewrite_contract_config_digest(paths["null_contract"], paths["null_config"])

    with pytest.raises(pair_cli.OnlinePairBuildError, match="must contain the same"):
        pair_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_rejects_checkpoint_paths_that_normalize_to_same_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    shared = str((tmp_path / "same-checkpoint.pt").resolve())
    monkeypatch.setenv("WARM_SHARED_CHECKPOINT", shared)
    fixed_config = loads(paths["fixed_config"].read_text(encoding="utf-8"))
    null_config = loads(paths["null_config"].read_text(encoding="utf-8"))
    fixed_config["ckpt"] = shared
    null_config["ckpt"] = "$WARM_SHARED_CHECKPOINT"
    paths["fixed_config"].write_text(dumps(fixed_config), encoding="utf-8")
    paths["null_config"].write_text(dumps(null_config), encoding="utf-8")
    _rewrite_contract_config_digest(paths["fixed_contract"], paths["fixed_config"])
    _rewrite_contract_config_digest(paths["null_contract"], paths["null_config"])

    with pytest.raises(pair_cli.OnlinePairBuildError, match="checkpoint paths"):
        pair_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_rejects_output_paths_that_normalize_to_same_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    shared = str((tmp_path / "same-results").resolve())
    monkeypatch.setenv("WARM_SHARED_RESULTS", shared)
    fixed_config = loads(paths["fixed_config"].read_text(encoding="utf-8"))
    null_config = loads(paths["null_config"].read_text(encoding="utf-8"))
    fixed_config["EVALUATION"]["output_dir"] = shared
    null_config["EVALUATION"]["output_dir"] = "$WARM_SHARED_RESULTS"
    paths["fixed_config"].write_text(dumps(fixed_config), encoding="utf-8")
    paths["null_config"].write_text(dumps(null_config), encoding="utf-8")
    _rewrite_contract_config_digest(paths["fixed_contract"], paths["fixed_config"])
    _rewrite_contract_config_digest(paths["null_contract"], paths["null_config"])

    with pytest.raises(pair_cli.OnlinePairBuildError, match="output directories"):
        pair_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_rejects_unapproved_sampler_or_trial_difference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    config = loads(paths["null_config"].read_text(encoding="utf-8"))
    config["EVALUATION"]["num_trials"] = 51
    paths["null_config"].write_text(dumps(config), encoding="utf-8")
    _rewrite_contract_config_digest(paths["null_contract"], paths["null_config"])

    with pytest.raises(pair_cli.OnlinePairBuildError, match="num_trials"):
        pair_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_rejects_different_tokenizer_or_other_science_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    value = loads(paths["null_contract"].read_text(encoding="utf-8"))
    value["tokenizer_tree_sha256"] = "9" * 64
    paths["null_contract"].write_text(dumps(value), encoding="utf-8")

    with pytest.raises(pair_cli.OnlinePairBuildError, match="tokenizer_tree_sha256"):
        pair_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_rejects_config_not_bound_by_its_online_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    config = loads(paths["fixed_config"].read_text(encoding="utf-8"))
    config["EVALUATION"]["replan_steps"] = 8
    paths["fixed_config"].write_text(dumps(config), encoding="utf-8")

    with pytest.raises(pair_cli.OnlinePairBuildError, match="does not match"):
        pair_cli.main(argv)


def test_cli_rejects_same_checkpoint_as_wrong_experiment_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    fixed = loads(paths["fixed_contract"].read_text(encoding="utf-8"))
    null = loads(paths["null_contract"].read_text(encoding="utf-8"))
    null["warm_checkpoint_sha256"] = fixed["warm_checkpoint_sha256"]
    paths["null_contract"].write_text(dumps(null), encoding="utf-8")

    with pytest.raises(pair_cli.OnlinePairBuildError, match="distinct checkpoint"):
        pair_cli.main(argv)


class _MutatingClaim(AbstractContextManager[object]):
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> object:
        config = loads(self.path.read_text(encoding="utf-8"))
        config["EVALUATION"]["replan_steps"] = 99
        self.path.write_text(dumps(config), encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback


def test_cli_rechecks_inputs_inside_publication_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pair_cli,
        "artifact_claim",
        lambda *_args, **_kwargs: _MutatingClaim(paths["null_config"]),
    )

    with pytest.raises(pair_cli.OnlinePairBuildError, match="does not match"):
        pair_cli.main(argv)
    assert not paths["output"].exists()


def test_cli_rejects_dirty_git_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    argv, paths = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(pair_cli, "_git_identity", lambda _repo: ("a" * 40, True))
    with pytest.raises(pair_cli.OnlinePairBuildError, match="clean Git"):
        pair_cli.main(argv)
    assert not paths["output"].exists()
