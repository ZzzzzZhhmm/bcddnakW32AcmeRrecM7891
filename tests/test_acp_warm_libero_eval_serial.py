from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "acp_warm_libero_eval_serial.sh"


def test_serial_eval_covers_the_full_standard_libero_matrix() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "libero_spatial libero_object libero_goal libero_10" in source
    assert "0 1 2 3 4 5 6 7 8 9" in source
    assert 'WARM_EVAL_SEEDS="${WARM_EVAL_SEEDS:-17}"' in source
    assert 'WARM_EXPECTED_TRIALS="${WARM_EXPECTED_TRIALS:-50}"' in source


def test_serial_eval_is_single_gpu_resumable_and_result_validating() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES must contain one device" in source
    assert "plan.tsv" in source
    assert ".success.json" in source
    assert "skip: already completed" in source
    assert "recovering completion marker" in source
    assert "incomplete immutable evaluation root exists" in source
    assert "warm_online_episodes" in source
    assert 'header.get("side") != "full_retrospection"' in source
    assert "result_sha256" in source
    assert "summary.json" in source


def test_serial_eval_delegates_each_task_to_the_attested_entrypoint() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '[[ -f "${SCRIPT_DIR}/acp_warm_libero_eval.sh" ]]' in source
    assert '[[ -x "${SCRIPT_DIR}/acp_warm_libero_eval.sh" ]]' not in source
    assert 'EVAL_ACTION=run' in source
    assert 'bash "${SCRIPT_DIR}/acp_warm_libero_eval.sh"' in source
    assert 'WARM_EVAL_ROOT="${eval_root}"' in source
    assert 'WARM_TASK_SUITE="${suite}"' in source
    assert 'WARM_TASK_ID="${task_id}"' in source
    assert 'WARM_ROOT_SEED="${seed}"' in source
    assert "git pull" not in source
    assert "git fetch" not in source
