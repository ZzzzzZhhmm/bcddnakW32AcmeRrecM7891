from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acp_warm_libero_eval.sh"
SETUP_SCRIPT = ROOT / "scripts" / "setup_warm_libero_eval_env.sh"


def test_acp_eval_entrypoint_is_offline_and_commit_bound() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "git pull" not in source
    assert "git fetch" not in source
    assert "DIFFSYNTH_SKIP_DOWNLOAD=true" in source
    assert 'EVAL_ACTION="${EVAL_ACTION:-run}"' in source
    assert "verify_training_attestation" in source
    assert 'worktree add --detach "${EVAL_CODE}" "${TRAIN_COMMIT}"' in source
    project_safe = source.index('register_git_safe_directory "${PROJECT_DIR}"')
    commit_probe = source.index('cat-file -e "${TRAIN_COMMIT}^{commit}"')
    assert project_safe < commit_probe
    assert 'register_git_safe_directory "${EVAL_CODE}"' in source
    assert "cannot register Git safe.directory" in source
    assert "safe.directory '*'" not in source
    assert 'status --porcelain' not in source
    assert 'WARM_FORMAL_EVAL_LAUNCHER' in source
    assert '${PROJECT_DIR}/scripts/evaluate_warm_full_server.sh' in source
    assert 'cd "${EVAL_CODE}"' in source
    assert 'bash "${WARM_FORMAL_EVAL_LAUNCHER}"' in source
    assert "KNOWN_ACTION_SIGNATURE_SOURCE_SHA256" in source
    assert "KNOWN_RUNTIME_FINGERPRINT_SOURCE_SHA256" in source
    assert "KNOWN_LIBERO_EVALUATOR_SOURCE_SHA256" in source
    assert "warm-step019100-eval-v2" in source
    assert "TRACKED_COMPATIBILITY_SHA256" in source
    assert "WARM_EVAL_COMPAT_PYTHONPATH" in source
    assert "WARM_EVAL_COMPAT_ENCODER_CONTRACT_PATH" in source
    assert "WARM_EVAL_COMPAT_ENCODER_DEVICE" in source
    assert "-compat-${WARM_EVAL_COMPATIBILITY_SHA256:0:12}" in source


def test_acp_eval_entrypoint_prepares_exact_libero_inputs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for suite in (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    ):
        assert suite in source
    assert "get_task_init_states" in source
    assert "task.problem_folder" in source
    assert "task.bddl_file" in source
    assert "np.array_equal" in source
    assert "WARM_TASK_DESCRIPTION" in source
    assert "WARM_INITIAL_STATES" in source
    assert "WARM_BDDL" in source
    assert "libero_render_smoke_ok" in source
    assert "WARM_PREPARE_RENDER_TIMEOUT_SECONDS" in source
    assert "MUJOCO_EGL_DEVICE_ID" in source
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" in source
    assert "unset TORCH_FORCE_WEIGHTS_ONLY_LOAD" in source
    assert "contextlib.redirect_stdout(io.StringIO())" in source
    assert "task-input helper returned a malformed metadata path" in source
    assert "task metadata was not published" in source


def test_acp_eval_entrypoint_keeps_per_run_outputs_immutable() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "WARM_EVAL_LABEL" in source
    assert "immutable evaluation root already exists" in source
    assert 'tee "${WARM_EVAL_ROOT}.console.log"' in source
    assert "WARM_EVALUATION_NAMESPACE" in source
    assert "evaluation_compatibility_sha256=" in source


def test_acp_eval_entrypoint_owns_noninteractive_libero_config() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "LIBERO_CONFIG_PATH" in source
    assert "importlib.util.find_spec" in source
    assert "setup_warm_libero_eval_env.sh" in source
    assert '"bddl_files"' in source
    assert '"init_states"' in source


def test_libero_setup_does_not_install_legacy_requirement_bundle() -> None:
    source = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "requirements.txt" in source
    assert "pip install -r" not in source
    assert '"mujoco==3.3.2"' in source
    assert '"robosuite==1.4.0"' in source
    assert "--no-deps" in source
    assert "8f1084e3132a39270c3a13ebe37270a43ece2a01" in source
    assert "WARM_BOOTSTRAP_DIR" in source
    assert "WARM_PIP_CACHE_DIR" in source
    assert "ROBOSUITE_WHEEL_SHA256" in source
    assert "LIBERO_SOURCE_ARCHIVE_SHA256" in source
    assert "--no-cache-dir" not in source
    assert "--no-same-owner" in source
    assert "--no-same-permissions" in source
    assert "cleanup_temporary_source" in source
    assert "warm_pinned_libero_source.pth" in source
    assert "sysconfig.get_path" in source


def test_acp_eval_explicitly_exposes_pinned_libero_source() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "WARM_LIBERO_SOURCE_DIR" in source
    assert '${WARM_LIBERO_SOURCE_DIR}' in source
    assert "libero/libero/__init__.py" in source
