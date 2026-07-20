from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acp_warm_libero_eval.sh"


def test_acp_eval_entrypoint_is_offline_and_commit_bound() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "git pull" not in source
    assert "git fetch" not in source
    assert "DIFFSYNTH_SKIP_DOWNLOAD=true" in source
    assert 'EVAL_ACTION="${EVAL_ACTION:-run}"' in source
    assert "verify_training_attestation" in source
    assert 'worktree add --detach "${EVAL_CODE}" "${TRAIN_COMMIT}"' in source
    assert 'safe.directory "${EVAL_CODE}"' in source
    assert "safe.directory '*'" not in source
    assert 'status --porcelain' in source
    assert 'bash scripts/evaluate_warm_full_server.sh' in source


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


def test_acp_eval_entrypoint_keeps_per_run_outputs_immutable() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "WARM_EVAL_LABEL" in source
    assert "immutable evaluation root already exists" in source
    assert 'tee "${WARM_EVAL_ROOT}.console.log"' in source
    assert "WARM_EVALUATION_NAMESPACE" in source
