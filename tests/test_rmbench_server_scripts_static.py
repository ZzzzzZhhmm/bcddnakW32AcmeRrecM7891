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
    assert "warm_configure_job_local_caches" in source
    assert 'export TRITON_CACHE_DIR="${root}/triton/autotune"' in source
    assert 'export TORCHINDUCTOR_CACHE_DIR="${root}/torchinductor"' in source
    assert 'export CUDA_CACHE_PATH="${root}/cuda"' in source
    assert 'available_kib < 1048576' in source
    assert 'job-local cache resolved to a network filesystem' in source
    assert "push URL must be the literal DISABLED" in source
    assert "warm_register_safe_directory" in source
    assert 'git config --global --add safe.directory "${canonical}"' in source
    assert "safe.directory '*'" not in source
    assert "git: ${git_error}" in source


def test_formal_server_entrypoints_register_the_exact_project_checkout() -> None:
    entrypoints = (
        "build_warm_rmbench_contract_bundle_server.sh",
        "evaluate_fastwam_rmbench_server.sh",
        "evaluate_warm_rmbench_server.sh",
        "evaluate_warm_rmbench_task_server.sh",
        "prepare_warm_rmbench_artifacts.sh",
        "train_fastwam_rmbench_server.sh",
        "train_warm_rmbench_server.sh",
    )
    for entrypoint in entrypoints:
        source = _read(entrypoint)
        assert 'PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"' in source
        assert 'warm_register_safe_directory "${PROJECT_ROOT}"' in source
        assert source.index("warm_register_safe_directory") < source.index(
            "warm_require_private_checkout"
        )


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
    acp_specialist = _read("acp_warm_rmbench_specialist.sh")
    contract = _read("build_warm_rmbench_sota_task_contract_server.sh")
    evaluation = _read("evaluate_warm_rmbench_task_server.sh")

    assert "WARM_RMBENCH_STAGE" in training
    assert "WARM_RMBENCH_SPECIALIST_TASK" in training
    assert "TARGET_GLOBAL_BATCH_SIZE" in training
    assert "episode_task_allowlist" in training
    assert "rmbench_task_event_balanced" not in training  # resolved by registry
    assert "sampler.mode=${SAMPLER_MODE}" in training
    assert "WARM_RESUME_STATE" in training
    assert "trainer_state.json" in training
    assert "WARM_PREFLIGHT_RESOLVE" in training
    assert "--cfg job --resolve" in training
    assert "git pull" not in acp_specialist
    assert "PER_DEVICE_BATCH_SIZE=8" in acp_specialist
    assert "GRADIENT_ACCUMULATION_STEPS=4" in acp_specialist
    assert "TARGET_GLOBAL_BATCH_SIZE=128" in acp_specialist
    assert "official50-dev45" in acp_specialist
    assert "save_every=1000" in acp_specialist
    assert "worktree add --detach" in acp_specialist
    assert 'WARM_TRAIN_CODE_DIR="${WARM_TRAIN_CODE_DIR:-}"' in acp_specialist
    assert 'WARM_CODE_REVISION="${TRAIN_COMMIT}"' in acp_specialist
    assert 'PROJECT_DIR="${PROJECT_DIR}"' in acp_specialist
    assert "formal-training-worktree.lock" in acp_specialist
    assert "flock -x 9" in acp_specialist
    assert "PYTHONDONTWRITEBYTECODE=1" in acp_specialist
    assert "status --porcelain --untracked-files=all" in acp_specialist
    assert 'cd "${WARM_TRAIN_CODE_DIR}"' in acp_specialist
    assert "fastwam import escaped the isolated training worktree" in acp_specialist
    assert "warm_configure_job_local_caches" in acp_specialist
    assert "WARM_JOB_LOCAL_CACHE_ROOT" in acp_specialist
    assert acp_specialist.index("warm_configure_job_local_caches") < acp_specialist.index(
        "torch.cuda.device_count()"
    )
    assert "rmbench_sota_matrix.json" in contract
    assert "WARM_RMBENCH_TASK" in contract
    assert "seed=3407" in evaluation
    assert "warm_recent_event_capacity" in evaluation
    assert "warm_action_summary_capacity" in evaluation
    assert "warm_experiment_id" in evaluation


def test_rmbench_specialist_eval_launcher_pins_checkpoint_training_commit() -> None:
    source = _read("acp_warm_rmbench_specialist_eval.sh")
    assert "git pull" not in source
    assert "git fetch" not in source
    assert "worktree add --detach" in source
    assert 'TRAIN_COMMIT' in source
    assert 'WARM_CODE_REVISION="${TRAIN_COMMIT}"' in source
    assert 'register_git_safe_directory "${PROJECT_DIR}"' in source
    assert 'register_git_safe_directory "${RMBENCH_ROOT}"' in source
    assert 'register_git_safe_directory "${EVAL_CODE}"' in source
    assert "flock -x 9" in source
    assert "PYTHONDONTWRITEBYTECODE=1" in source
    assert "build_warm_rmbench_sota_task_contract_server.sh" in source
    assert "evaluate_warm_rmbench_task_server.sh" in source
    assert "formal100-s3407-v4" in source
    assert "-s3407-v4" in source
    runtime_check = _read("check_warm_rmbench_eval_runtime.py")
    assert "expected exactly one visible CUDA device" in runtime_check
    assert "rmbench_f77_contract_v1/run_contract_bundle.py" in source
    assert "KNOWN_F77_ONLINE_BUILDER_SHA256" in source
    assert "CONTRACT_COMPAT_SUFFIX" in source
    assert 'WARM_EVALUATION_NAMESPACE_BASE' in source
    assert "warm-rmbench-eval" in source
    assert "check_warm_rmbench_eval_runtime.py" in source
    assert "bootstrap_warm_rmbench_eval_env.sh once in CCI" in source
    assert source.index("check_warm_rmbench_eval_runtime.py") < source.index(
        "BUILD RMBENCH SPECIALIST CONTRACT"
    )


def test_rmbench_eval_bootstrap_preserves_warm_runtime_and_pins_simulator() -> None:
    source = _read("bootstrap_warm_rmbench_eval_env.sh")
    assert "--system-site-packages" in source
    assert "warm-rmbench-eval" in source
    assert "warm_rmbench_eval.constraints" in source
    assert '"sapien==3.0.0b1"' in source
    assert '"mplib==0.2.1"' in source
    assert '"open3d==0.18.0"' not in source
    assert "install_open3d_rgb_guard.py" in source
    assert "WARM_RMBENCH_WHEELHOUSE" in source
    assert "PIP_CACHE_DIR" in source
    assert "download_verified_http_ranges.py" in source
    assert "mirrors.aliyun.com" in source
    assert "trimesh[easy]" in source
    assert "--no-deps" in source
    assert '"Cython==0.29.37"' in source
    assert "BUILD_PACKAGES=(" in source
    assert "--no-build-isolation" in source
    assert "toppra-0.6.3-*.whl" in source
    assert '--wheel-dir "${WARM_RMBENCH_WHEELHOUSE}"' in source
    assert "http.version=HTTP/1.1" in source
    assert "CUROBO_FETCH_ATTEMPTS" in source
    assert "--filter=blob:none" in source
    assert 'refs/tags/${CUROBO_TAG}^{commit}' in source
    assert "sparse-checkout set" in source
    assert "'!/src/curobo/content/assets/'" in source
    assert source.index(
        'safe.directory "${CUROBO_SOURCE}"'
    ) < source.index('git -C "${CUROBO_SOURCE}" init')
    assert '"warp-lang==1.11.1"' in source
    assert '"scikit-image==0.22.0"' in source
    assert '"lazy_loader==0.4"' in source
    assert '"tifffile==2024.9.20"' in source
    assert '"pillow==11.1.0"' in source
    assert "--force-reinstall --no-deps" in source
    assert "d64c4b005459db10c5dd867d8b30a87d5bda9bdb" in source
    assert "check_warm_rmbench_eval_runtime.py" in source
    assert "bootstrap_warm_rmbench_assets.py" in source
    assert "RMBENCH_ASSET_STORE" in source
    assert "HF_XET_HIGH_PERFORMANCE" in source
    assert "RMBENCH_ASSET_MAX_WORKERS" in source
    assert "RMBENCH_EVAL_ENV_READY" in source
    assert "script/requirements.txt" in source
    assert "pip install -r" not in source


def test_rmbench_contract_builder_retains_validated_compute_device() -> None:
    builder = _read("build_warm_online_contract.py")
    assert builder.count(
        "_, _, _, compute_device = validate_online_encoder_contract("
        "encoder_contract)"
    ) == 2

    compatibility = _read(
        "evaluation_compat/rmbench_f77_contract_v1/run_contract_bundle.py"
    )
    assert "f77c63385c747fdc1424386489cbf9c7ea57ddc5" in compatibility
    assert "EXPECTED_ONLINE_BUILDER_SHA256" in compatibility
    assert "EXPECTED_BUNDLE_BUILDER_SHA256" in compatibility
    assert "online_builder.compute_device = compute_device" in compatibility
    assert "module.main(arguments)" in compatibility


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
