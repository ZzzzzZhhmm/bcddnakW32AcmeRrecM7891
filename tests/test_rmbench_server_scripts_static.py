from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_common_server_guard_is_private_clean_pinned_and_offline_by_default() -> None:
    source = _read("warm_server_common.sh")
    assert "ZzzzzZhhmm/WARM.git" in source
    assert "exactly one remote named origin" in source
    assert "git status --porcelain" in source
    assert 'WANDB_MODE:-offline' in source
    assert 'HF_HUB_OFFLINE:-1' in source
    assert "WARM_ALLOW_NETWORK_LOGGING" in source
    assert "push URL must be the literal DISABLED" in source


def test_rmbench_artifact_script_is_exact_three_camera_h32_pipeline() -> None:
    source = _read("prepare_warm_rmbench_artifacts.sh")
    assert "57ee09cbc6267bc36ca0ac2d8d1c5c3b245c112c" in source
    assert "855e90e1213d150bf4889130e83398f107314681" in source
    assert "validate_rmbench_conversion.py" in source
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        assert f"observation.images.{camera}" in source
    assert "--benchmark-profile robotwin" in source
    assert "--action-horizon 32" in source
    assert "--expected-action-dim 14" in source
    assert "--query-split train" in source
    assert "--query-split dev" in source


def test_rmbench_training_and_eval_launchers_bind_contracts_and_closed_controls() -> None:
    training = _read("train_warm_rmbench_server.sh")
    evaluation = _read("evaluate_warm_rmbench_server.sh")
    assert "task=rmbench_warm_3cam384_1e-4" in training
    assert "hybrid_h32_train_source.json" in training
    assert "hybrid_h32_dev_source.json" in training
    assert "wandb.mode=${WANDB_MODE}" in training
    assert "wandb.mode=online" not in training
    assert "protected formal-training override" in training
    assert "data.*|model|model.*|output_dir|resume" in training
    assert "run_rmbench_manager.py" in evaluation
    assert "WARM_RMBENCH_ONLINE_CONTRACT" in evaluation
    assert '${EXPERIMENT_ID}/${task_name}.json' in evaluation
    assert 'seeds/${task_name}.seed_protocol.npy' in evaluation
    assert "warm_m1_data_config_path=$(pwd)/configs/data/rmbench_3cam.yaml" in evaluation
    assert evaluation.count("warm_m1_data_config_path=") == 1
    assert "full|context_only|source_only_no_consequence" in evaluation
    assert "clean|wrong_event|reversed_action|phase_shift|effect_mismatch" in evaluation
    assert "2|4|8|10" in evaluation
    assert "MULTIRUN.num_gpus" in evaluation


def test_score_oriented_task_launchers_bind_registry_batch_and_online_memory() -> None:
    training = _read("train_warm_rmbench_server.sh")
    contract = _read("build_warm_rmbench_sota_task_contract_server.sh")
    evaluation = _read("evaluate_warm_rmbench_task_server.sh")

    assert "WARM_RMBENCH_STAGE" in training
    assert "WARM_RMBENCH_SPECIALIST_TASK" in training
    assert "TARGET_GLOBAL_BATCH_SIZE" in training
    assert "episode_task_allowlist" in training
    assert "rmbench_task_event_balanced" not in training  # resolved by registry
    assert "sampler.mode=${SAMPLER_MODE}" in training
    assert "rmbench_sota_matrix.json" in contract
    assert "WARM_RMBENCH_TASK" in contract
    assert "seed=3407" in evaluation
    assert "warm_recent_event_capacity" in evaluation
    assert "warm_action_summary_capacity" in evaluation
    assert "warm_experiment_id" in evaluation


def test_same_data_fastwam_baseline_has_separate_train_and_eval_launchers() -> None:
    training = _read("train_fastwam_rmbench_server.sh")
    evaluation = _read("evaluate_fastwam_rmbench_server.sh")
    assert "task=rmbench_fastwam_3cam384_1e-4" in training
    assert "resume=${FASTWAM_BASE_CHECKPOINT}" in training
    assert "must be a weight file" in training
    assert "RMBENCH_EPISODE_CATALOG" in training
    assert "warm_candidates" not in training
    assert "policy_kind=fastwam_baseline" in evaluation
    assert "experiments/robotwin/fastwam_policy" in evaluation
    assert "WARM_RMBENCH_ONLINE_CONTRACT" in evaluation
    assert "warm_bank_directory" not in evaluation
    assert "evaluation_namespace=" in evaluation


def test_fastwam_baseline_policy_uses_paired_per_replan_seeds() -> None:
    source = (
        ROOT / "experiments" / "robotwin" / "fastwam_policy" / "deploy_policy.py"
    ).read_text(encoding="utf-8")
    assert "derive_rmbench_policy_query_seed" in source
    assert "episode_index=self.active_episode_index" in source
    assert "frame_index=self.step_count" in source
