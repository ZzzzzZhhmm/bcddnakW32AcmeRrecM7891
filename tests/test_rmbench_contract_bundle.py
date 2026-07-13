from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fastwam.benchmarks import rmbench_runtime
from fastwam.memory.manifest import sha256_array, sha256_canonical_json, sha256_file
from scripts import build_warm_rmbench_contract_bundle as bundle


def _runtime_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    artifact_names = (
        "online_contract",
        "training_attestation",
        "training_run_contract",
        "validation_run_contract",
        "base_checkpoint",
        "event_bank",
        "normalizer_contract",
        "encoder_contract",
        "camera_contract",
        "m1_data_config",
        "dino_checkpoint",
        "catalog",
        "audit_report",
        "seed_protocol",
        "task_definition",
        "vae_checkpoint",
        "text_encoder",
        "tokenizer",
        "warm_checkpoint",
        "normalization_stats",
    )
    artifacts = {name: str((tmp_path / name).resolve()) for name in artifact_names}
    online = {
        "enabled": True,
        "mode": "full_retrospection",
        "contract_path": artifacts["online_contract"],
        "training_attestation_path": artifacts["training_attestation"],
        "training_run_contract_path": artifacts["training_run_contract"],
        "validation_run_contract_path": artifacts["validation_run_contract"],
        "base_checkpoint_path": artifacts["base_checkpoint"],
        "bank_directory": artifacts["event_bank"],
        "normalizer_contract_path": artifacts["normalizer_contract"],
        "encoder_contract_path": artifacts["encoder_contract"],
        "camera_contract_path": artifacts["camera_contract"],
        "m1_data_config_path": artifacts["m1_data_config"],
        "dino_checkpoint_path": artifacts["dino_checkpoint"],
        "catalog_path": artifacts["catalog"],
        "audit_report_path": artifacts["audit_report"],
        "evaluation_namespace": "warm-rmbench-full-v1",
        "top_k": 32,
        "dino_device": "cuda",
        "dino_batch_size": 1,
        "experiment_id": "full_warm",
        "ablation_mode": "full",
        "memory_corruption": "clean",
        "num_inference_steps": 10,
    }
    cfg = {
        "seed": 17,
        "ckpt": artifacts["warm_checkpoint"],
        "model": {
            "_target_": "fixture.Model",
            "source_policy": "fixed_context_top1",
            "memory_sigma": 0.2,
            "base_checkpoint_path": artifacts["base_checkpoint"],
        },
        "data": {"train": {"processor": {"fixture": True}}},
        "EVALUATION": {
            "task_suite_name": "rmbench",
            "task_id": 2,
            "task_description": "put_back_block",
            "dataset_stats_path": artifacts["normalization_stats"],
            "action_horizon": 32,
            "replan_steps": 10,
            "num_inference_steps": 10,
            "sigma_shift": None,
            "text_cfg_scale": 1.0,
            "negative_prompt": "",
            "rand_device": "cpu",
            "tiled": False,
            "visualize_future_video": False,
            "use_action_ensembler": False,
            "warm_online": online,
        },
    }
    arg_names = {
        "online_contract": "warm_online_contract_path",
        "training_attestation": "warm_training_attestation_path",
        "training_run_contract": "warm_training_run_contract_path",
        "validation_run_contract": "warm_validation_run_contract_path",
        "base_checkpoint": "warm_base_checkpoint_path",
        "event_bank": "warm_bank_directory",
        "normalizer_contract": "warm_normalizer_contract_path",
        "encoder_contract": "warm_encoder_contract_path",
        "camera_contract": "warm_camera_contract_path",
        "m1_data_config": "warm_m1_data_config_path",
        "dino_checkpoint": "warm_dino_checkpoint_path",
        "catalog": "warm_catalog_path",
        "audit_report": "warm_audit_report_path",
        "seed_protocol": "warm_initial_states_path",
        "task_definition": "warm_task_definition_path",
        "vae_checkpoint": "warm_vae_checkpoint_path",
        "text_encoder": "warm_text_encoder_path",
        "tokenizer": "warm_tokenizer_path",
    }
    args = {argument: artifacts[name] for name, argument in arg_names.items()}
    args.update(
        {
            "task_name": "put_back_block",
            "seed": 17,
            "ckpt_setting": artifacts["warm_checkpoint"],
            "dataset_stats_path": artifacts["normalization_stats"],
            "warm_experiment_id": "full_warm",
            "warm_ablation_mode": "full",
            "warm_memory_corruption": "clean",
            "num_inference_steps": 10,
            "action_horizon": 32,
            "replan_steps": 10,
            "sigma_shift": None,
            "text_cfg_scale": 1.0,
            "negative_prompt": "",
            "rand_device": "cpu",
            "tiled": False,
            "warm_top_k": 32,
            "warm_evaluation_namespace": "warm-rmbench-full-v1",
            "mixed_precision": "bf16",
            "device": "cuda",
            "warm_telemetry_path": str(tmp_path / "ignored-one.jsonl"),
        }
    )
    monkeypatch.setattr(
        rmbench_runtime,
        "extract_m1_processor_recipe",
        lambda *_args, **_kwargs: {"profile": "robotwin", "exact": True},
    )
    return cfg, args


def test_runtime_projection_binds_behavior_but_not_runner_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, args = _runtime_fixture(tmp_path, monkeypatch)
    first = rmbench_runtime.build_rmbench_policy_runtime_projection(cfg, args)
    args["warm_telemetry_path"] = str(tmp_path / "ignored-two.jsonl")
    args["gpu_id"] = 7
    second = rmbench_runtime.build_rmbench_policy_runtime_projection(cfg, args)
    assert first == second
    assert first["experiment"]["experiment_id"] == "full_warm"
    assert first["action_generation"]["sampler"]["num_inference_steps"] == 10
    assert "warm_telemetry_path" not in json.dumps(first)

    args["num_inference_steps"] = 8
    with pytest.raises(
        rmbench_runtime.RMBenchRuntimeProjectionError,
        match="num_inference_steps",
    ):
        rmbench_runtime.build_rmbench_policy_runtime_projection(cfg, args)


def test_root_seed_protocol_is_namespace_not_accepted_seed_claim() -> None:
    values = bundle.root_seed_namespace(17)
    assert values.shape == (100, 2)
    assert values.dtype == np.int64
    assert values[0].tolist() == [0, 1_800_000]
    assert values[-1].tolist() == [99, 1_800_099]
    assert np.all(np.diff(values[:, 1]) == 1)


def test_bundle_validator_rejects_tamper_and_accepts_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "bundle"
    seed_path = root / "seeds" / "put_back_block.seed_protocol.npy"
    seed_path.parent.mkdir(parents=True)
    seed = bundle.root_seed_namespace(17)
    np.save(seed_path, seed, allow_pickle=False)

    projection = {
        "experiment": {
            "experiment_id": "full_warm",
            "ablation_mode": "full",
            "memory_corruption": "clean",
        }
    }
    projection_path = root / "runtime_projections" / "full_warm" / "put_back_block.json"
    projection_path.parent.mkdir(parents=True)
    projection_path.write_text(json.dumps(projection), encoding="utf-8")
    contract_path = root / "full_warm" / "put_back_block.json"
    contract_path.parent.mkdir(parents=True)
    contract_path.write_text("{}", encoding="utf-8")
    projection_sha = sha256_canonical_json(projection)
    fake_contract = SimpleNamespace(
        sha256="f" * 64,
        resolved_eval_config_sha256=projection_sha,
        task_suite="rmbench",
        task_id=2,
        task_description="put_back_block",
        root_seed=17,
        initial_states_sha256=sha256_array(seed),
        action_horizon=32,
        action_dim=14,
    )
    monkeypatch.setattr(bundle, "_load_contract", lambda _path: fake_contract)
    experiment = {
        "id": "full_warm",
        "ablation_mode": "full",
        "memory_corruption": "clean",
        "num_inference_steps": 10,
    }
    manifest = {
        "schema": bundle.BUNDLE_SCHEMA,
        "schema_version": bundle.BUNDLE_SCHEMA_VERSION,
        "complete": True,
        "task_manifest_sha256": bundle.RMBENCH_TASK_MANIFEST_SHA256,
        "root_seed": 17,
        "experiments": [experiment],
        "tasks": [{"name": "put_back_block", "task_id": 2}],
        "cell_count": 1,
        "seed_protocol": {
            "schema": bundle.SEED_PROTOCOL_SCHEMA,
            "schema_version": bundle.SEED_PROTOCOL_VERSION,
            "not_accepted_seed_claim": True,
            "tasks": [
                {
                    "task": "put_back_block",
                    "relative_path": "seeds/put_back_block.seed_protocol.npy",
                    "file_sha256": sha256_file(seed_path),
                    "array_sha256": sha256_array(seed),
                    "not_accepted_seed_claim": True,
                }
            ],
        },
        "cells": [
            {
                "experiment": experiment,
                "task": "put_back_block",
                "contract_relpath": "full_warm/put_back_block.json",
                "contract_file_sha256": sha256_file(contract_path),
                "online_run_contract_sha256": fake_contract.sha256,
                "runtime_projection_relpath": (
                    "runtime_projections/full_warm/put_back_block.json"
                ),
                "runtime_projection_sha256": sha256_file(projection_path),
                "runtime_projection_canonical_sha256": projection_sha,
            }
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = bundle.validate_contract_bundle(
        root, experiment_id="full_warm", task_names=["put_back_block"]
    )
    assert result["tasks"][0]["contract_sha256"] == "f" * 64

    projection_path.write_text('{"experiment":{}}', encoding="utf-8")
    with pytest.raises(bundle.RMBenchBundleError, match="projection file drift"):
        bundle.validate_contract_bundle(
            root, experiment_id="full_warm", task_names=["put_back_block"]
        )
