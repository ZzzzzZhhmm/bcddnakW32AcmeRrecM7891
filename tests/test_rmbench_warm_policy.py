from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[1]
POLICY = PROJECT / "experiments" / "rmbench" / "warm_policy"
_SPEC = importlib.util.spec_from_file_location(
    "warm_rmbench_policy_core", POLICY / "policy_core.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_CORE = importlib.util.module_from_spec(_SPEC)
sys_modules = __import__("sys").modules
sys_modules[_SPEC.name] = _CORE
_SPEC.loader.exec_module(_CORE)
JsonlEvidenceWriter = _CORE.JsonlEvidenceWriter
PROCESSOR_CAMERA_KEYS = _CORE.PROCESSOR_CAMERA_KEYS
RMBenchPolicyBoundaryError = _CORE.RMBenchPolicyBoundaryError
RecedingHorizonQueue = _CORE.RecedingHorizonQueue
factual_cameras = _CORE.factual_cameras
factual_joint_state = _CORE.factual_joint_state
resolve_task_bundle_paths = _CORE.resolve_task_bundle_paths
validate_warm_model_telemetry = _CORE.validate_warm_model_telemetry


def _observation() -> dict:
    def image(value: int) -> np.ndarray:
        return np.full((12, 18, 3), value, dtype=np.uint8)

    return {
        "observation": {
            "head_camera": {"rgb": image(1)},
            "left_camera": {"rgb": image(2)},
            "right_camera": {"rgb": image(3)},
        },
        "joint_action": {"vector": np.arange(14, dtype=np.float32)},
    }


def test_policy_core_maps_exact_three_camera_and_native_qpos_contract() -> None:
    observation = _observation()
    cameras = factual_cameras(observation)
    assert tuple(cameras) == PROCESSOR_CAMERA_KEYS
    assert [int(cameras[key][0, 0, 0]) for key in cameras] == [1, 2, 3]
    assert all(value.flags.c_contiguous for value in cameras.values())
    assert np.array_equal(
        factual_joint_state(observation), np.arange(14, dtype=np.float32)
    )


def test_policy_core_rejects_ambiguous_camera_or_action_shapes() -> None:
    observation = _observation()
    observation["observation"]["head_camera"]["rgb"] = np.zeros(
        (3, 12, 18), dtype=np.uint8
    )
    with pytest.raises(RMBenchPolicyBoundaryError, match="uint8 HWC"):
        factual_cameras(observation)

    observation = _observation()
    observation["joint_action"]["vector"] = np.zeros((13,), dtype=np.float32)
    with pytest.raises(RMBenchPolicyBoundaryError, match=r"\[14\]"):
        factual_joint_state(observation)


def test_receding_horizon_queue_retains_paired_exact_actions() -> None:
    queue = RecedingHorizonQueue(replan_steps=3, action_horizon=32)
    model = np.arange(32 * 14, dtype=np.float32).reshape(32, 14)
    environment = model + np.float32(0.5)
    queue.publish(model, environment)
    assert len(queue) == 3
    for index in range(3):
        item = queue.pop()
        assert np.array_equal(item.model_space, model[index])
        assert np.array_equal(item.environment_space, environment[index])
    assert not queue

    queue.publish(model, environment)
    with pytest.raises(RuntimeError, match="non-empty"):
        queue.publish(model, environment)


def test_jsonl_evidence_is_exclusive_canonical_and_durable(tmp_path: Path) -> None:
    path = tmp_path / "evidence.jsonl"
    writer = JsonlEvidenceWriter(path)
    writer.append({"z": 2, "a": [1, 3]})
    writer.close()
    assert path.read_text(encoding="utf-8") == '{"a":[1,3],"z":2}\n'
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": [1, 3], "z": 2}
    with pytest.raises(FileExistsError):
        JsonlEvidenceWriter(path)


def test_task_bundle_root_resolves_one_contract_per_task(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    experiment = bundle / "full_clean_10step"
    experiment.mkdir(parents=True)
    contract_file = experiment / "put_back_block.json"
    contract_file.write_text("{}", encoding="utf-8")
    (bundle / "seeds").mkdir()
    seed_file = bundle / "seeds" / "put_back_block.seed_protocol.npy"
    np.save(seed_file, np.arange(3, dtype=np.int64))
    official = tmp_path / "official"
    (official / "envs").mkdir(parents=True)
    task_file = official / "envs" / "put_back_block.py"
    task_file.write_text("# pinned\n", encoding="utf-8")
    contract, initial, definition = resolve_task_bundle_paths(
        bundle,
        task_name="put_back_block",
        experiment_id="full_clean_10step",
        official_root=official,
    )
    assert contract == contract_file
    assert initial == seed_file
    assert definition == task_file


def _telemetry_fixture(*, valid: tuple[bool, ...] = (True, True)) -> tuple[object, dict]:
    events = tuple(
        SimpleNamespace(
            dataset_id="rmbench",
            dataset_index=0,
            episode_index=index,
            start_frame=4 * index,
        )
        if is_valid
        else None
        for index, is_valid in enumerate(valid)
    )
    step = SimpleNamespace(
        candidate_valid_mask=np.asarray(valid, dtype=np.bool_),
        event_ids=events,
        bank_rows=np.arange(len(valid), dtype=np.int64),
        step_sha256="step",
        prompt_sha256="prompt",
        proprio_sha256="proprio",
        model_input_sha256="model-input",
        derived_seed=41,
        query_id=SimpleNamespace(
            dataset_id="rmbench-eval",
            dataset_index=0,
            episode_index=3,
            frame_index=20,
        ),
        cosine_scores=np.linspace(0.9, 0.8, len(valid), dtype=np.float32),
    )
    telemetry = {
        "experiment": {
            "experiment_id": "full_clean_10step",
            "ablation_mode": "full",
            "memory_corruption": "clean",
            "corruption_applied": False,
            "corruption_fallback": False,
        },
        "retrieval": {
            "online_contract_sha256": "online",
            "training_run_contract_sha256": "train",
            "validation_run_contract_sha256": "dev",
            "bank_manifest_sha256": "manifest",
            "bank_content_sha256": "bank",
            "step_sha256": "step",
            "prompt_sha256": "prompt",
            "proprio_sha256": "proprio",
            "model_input_sha256": "model-input",
            "bank_rows": list(range(len(valid))),
            "candidate_valid_mask": list(valid),
            "query_id": {
                "dataset_id": "rmbench-eval",
                "dataset_index": 0,
                "episode_index": 3,
                "frame_index": 20,
            },
            "ranked_event_ids": [
                (
                    {
                        "dataset_id": "rmbench",
                        "dataset_index": 0,
                        "episode_index": index,
                        "start_frame": 4 * index,
                    }
                    if is_valid
                    else None
                )
                for index, is_valid in enumerate(valid)
            ],
            "cosine_scores": [
                float(item)
                for item in np.linspace(0.9, 0.8, len(valid), dtype=np.float32)
            ],
        },
        "source": {
            "policy": "consequence_aligned_retrospection",
            "component": 1,
            "selected_rank": 0,
            "selected_event_id": {
                "dataset_id": "rmbench",
                "dataset_index": 0,
                "episode_index": 0,
                "start_frame": 0,
            },
            "candidate_selected": True,
            "memory_selected": True,
            "memory_sigma": 0.2,
            "gate": 0.7,
            "memory_relevance_gate": 1.0,
            "learned_gate": 0.8,
            "source_quality": 0.875,
            "selected_probability": 0.7,
            "probability_margin": 0.4,
            "normalized_entropy": 0.2,
            "stagnation_score": 0.0,
            "thread_prior": 0.5,
            "thread_source_eligible": True,
            "thread_action_offset": 4,
            "thread_phase_elapsed_actions": 8,
            "episode_query_delta_norm": 0.25,
            "derived_seed": 41,
        },
    }
    return step, telemetry


def _validate_telemetry(step: object, telemetry: dict, **overrides: object) -> dict:
    arguments = {
        "experiment_id": "full_clean_10step",
        "ablation_mode": "full",
        "memory_corruption": "clean",
        "online_contract_sha256": "online",
        "training_run_contract_sha256": "train",
        "validation_run_contract_sha256": "dev",
        "bank_manifest_sha256": "manifest",
        "bank_content_sha256": "bank",
        "memory_sigma": 0.2,
    }
    arguments.update(overrides)
    return validate_warm_model_telemetry(
        telemetry, online_step=step, **arguments
    )


def test_model_telemetry_validation_binds_experiment_retrieval_and_source() -> None:
    step, telemetry = _telemetry_fixture()
    assert _validate_telemetry(step, telemetry) == telemetry

    changed = deepcopy(telemetry)
    changed["experiment"]["ablation_mode"] = "context_only"
    with pytest.raises(RMBenchPolicyBoundaryError, match="different experiment"):
        _validate_telemetry(step, changed)

    changed = deepcopy(telemetry)
    changed["retrieval"]["step_sha256"] = "different"
    with pytest.raises(RMBenchPolicyBoundaryError, match="bound retrieval"):
        _validate_telemetry(step, changed)


def test_model_telemetry_validation_enforces_corruption_fallback_semantics() -> None:
    step, telemetry = _telemetry_fixture(valid=(True,))
    telemetry["experiment"].update(
        {
            "experiment_id": "wrong_event_10step",
            "memory_corruption": "wrong_event",
            "corruption_applied": True,
            "corruption_fallback": True,
        }
    )
    telemetry["source"].update(
        {
            "component": 0,
            "selected_rank": None,
            "selected_event_id": None,
            "candidate_selected": False,
            "memory_selected": False,
        }
    )
    assert _validate_telemetry(
        step,
        telemetry,
        experiment_id="wrong_event_10step",
        memory_corruption="wrong_event",
    ) == telemetry

    telemetry["experiment"]["corruption_fallback"] = False
    with pytest.raises(RMBenchPolicyBoundaryError, match="not applied"):
        _validate_telemetry(
            step,
            telemetry,
            experiment_id="wrong_event_10step",
            memory_corruption="wrong_event",
        )


def test_model_telemetry_allows_candidate_considered_but_source_rejected() -> None:
    step, telemetry = _telemetry_fixture()
    telemetry["source"].update(
        {
            "component": 0,
            "memory_selected": False,
            "gate": 0.0,
            "memory_relevance_gate": 0.0,
            "source_quality": 0.0,
            "normalized_entropy": 1.0,
        }
    )
    assert _validate_telemetry(step, telemetry) == telemetry


def test_deploy_policy_exposes_official_signatures_and_full_warm_boundaries() -> None:
    path = POLICY / "deploy_policy.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: [argument.arg for argument in node.args.args]
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert functions["get_model"] == ["usr_args"]
    assert functions["eval"] == ["TASK_ENV", "model", "observation"]
    assert functions["reset_model"] == ["model"]
    required_fragments = (
        'benchmark_profile="robotwin"',
        "OnlineRetrospectiveEpisodeMemory(",
        "action_dim=14",
        "action_horizon=32",
        "gripper_indices=ROBOTWIN_GRIPPER_DIMS",
        'episode_namespace="rmbench-eval"',
        "OnlineEpisodeController(",
        "self.controller.issue_query_id(frame_index)",
        "self.controller.note_executed_action(",
        "self.controller.commit_factual_replan_observation(",
        'task_env.take_action(queued.environment_space, action_type="qpos")',
        "atexit.register(self._atexit)",
        "configure_experiment(",
        "build_rmbench_policy_runtime_projection(",
        '"policy_runtime_projection_sha256": self.runtime_projection_sha256',
    )
    for fragment in required_fragments:
        assert fragment in source


def test_deploy_yaml_has_no_unattested_artifact_defaults() -> None:
    value = yaml.safe_load((POLICY / "deploy_policy.yml").read_text(encoding="utf-8"))
    assert value["sim_task"] == "rmbench_warm_online_3cam384_full"
    assert value["action_horizon"] == 32
    assert value["replan_steps"] == 10
    assert value["warm_ablation_mode"] == "full"
    assert value["warm_memory_corruption"] == "clean"
    assert value["warm_experiment_id"] is None
    required = (
        "warm_online_contract_path",
        "warm_training_attestation_path",
        "warm_training_run_contract_path",
        "warm_validation_run_contract_path",
        "warm_base_checkpoint_path",
        "warm_bank_directory",
        "warm_normalizer_contract_path",
        "warm_encoder_contract_path",
        "warm_camera_contract_path",
        "warm_m1_data_config_path",
        "warm_dino_checkpoint_path",
        "warm_catalog_path",
        "warm_audit_report_path",
        "warm_initial_states_path",
        "warm_task_definition_path",
        "warm_telemetry_path",
        "warm_vae_checkpoint_path",
        "warm_text_encoder_path",
        "warm_tokenizer_path",
    )
    assert all(value[key] is None for key in required)
