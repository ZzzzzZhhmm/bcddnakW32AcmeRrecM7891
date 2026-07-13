from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_warm_rmbench_matrix import (
    ALLOWED_CORRUPTIONS,
    ALLOWED_ODE_STEPS,
    MatrixError,
    REQUIRED_EXECUTION_ENV,
    _accepted_seed_hashes,
    _preflight_formal_inputs,
    _validate_fastwam_reference,
    _validate_result,
    load_matrix,
    main,
    select_experiments,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs" / "ablation" / "rmbench_reproducible_matrix.json"


def test_checked_in_matrix_covers_claimed_ablation_corruption_and_ode_axes() -> None:
    matrix = load_matrix(MATRIX)

    assert matrix.suite == "official9"
    assert {item.ablation_mode for item in matrix.experiments} >= {
        "context_only",
        "source_only_no_consequence",
        "full",
    }
    assert {item.memory_corruption for item in matrix.experiments} == set(
        ALLOWED_CORRUPTIONS
    )
    assert {
        item.num_inference_steps
        for item in matrix.experiments
        if item.ablation_mode == "full" and item.memory_corruption == "clean"
    } == set(ALLOWED_ODE_STEPS)
    assert len(matrix.sha256) == 64


def test_matrix_selection_is_registered_and_order_preserving() -> None:
    matrix = load_matrix(MATRIX)
    selected = select_experiments(matrix, ["full_warm_ode04", "full_warm"])
    assert [item.id for item in selected] == ["full_warm_ode04", "full_warm"]
    with pytest.raises(MatrixError, match="unknown experiment"):
        select_experiments(matrix, ["not_registered"])
    with pytest.raises(MatrixError, match="must not repeat"):
        select_experiments(matrix, ["full_warm", "full_warm"])


def test_matrix_plan_only_does_not_require_server_environment(capsys) -> None:
    assert main(["--matrix", str(MATRIX), "--experiment", "full_warm", "--plan-only"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["complete_matrix"] is False
    assert plan["experiments"] == [
        {
            "id": "full_warm",
            "ablation_mode": "full",
            "memory_corruption": "clean",
            "num_inference_steps": 10,
        }
    ]


def test_matrix_rejects_missing_effect_corruption(tmp_path: Path) -> None:
    value = json.loads(MATRIX.read_text(encoding="utf-8"))
    value["experiments"] = [
        item
        for item in value["experiments"]
        if item["memory_corruption"] != "effect_mismatch"
    ]
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(MatrixError, match="corruption suite"):
        load_matrix(path)


def test_formal_preflight_requires_every_selected_task_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = load_matrix(MATRIX)
    experiment = select_experiments(matrix, ["full_warm"])[0]
    bundle = tmp_path / "bundle"
    tasks = (
        "observe_and_pickup",
        "rearrange_blocks",
        "put_back_block",
        "swap_blocks",
        "swap_T",
        "blocks_ranking_try",
        "press_button",
        "cover_blocks",
        "battery_try",
    )
    for task in tasks:
        seed = bundle / "seeds" / f"{task}.seed_protocol.npy"
        contract = bundle / experiment.id / f"{task}.json"
        seed.parent.mkdir(parents=True, exist_ok=True)
        contract.parent.mkdir(parents=True, exist_ok=True)
        seed.write_bytes(b"seed")
        contract.write_text("{}", encoding="utf-8")
    for name in REQUIRED_EXECUTION_ENV:
        path = bundle if name == "WARM_RMBENCH_ONLINE_CONTRACT" else tmp_path / name
        if path != bundle:
            path.write_text("fixture", encoding="utf-8")
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("WARM_EXPECTED_ORIGIN", "private-origin")

    def fake_git(*args: str) -> str:
        if args == ("remote",):
            return "origin"
        if args in {
            ("remote", "get-url", "origin"),
            ("remote", "get-url", "--push", "origin"),
        }:
            return "private-origin"
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        if args == ("status", "--porcelain", "--untracked-files=normal"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr("scripts.run_warm_rmbench_matrix._git", fake_git)
    _preflight_formal_inputs((experiment,), suite="official9")
    (bundle / experiment.id / "battery_try.json").unlink()
    with pytest.raises(MatrixError, match="missing experiment/task contract"):
        _preflight_formal_inputs((experiment,), suite="official9")


def test_matrix_summary_requires_per_task_actual_seed_evidence(tmp_path: Path) -> None:
    per_task = []
    for index, task_name in enumerate(
        (
            "observe_and_pickup",
            "rearrange_blocks",
            "put_back_block",
            "swap_blocks",
            "swap_T",
            "blocks_ranking_try",
            "press_button",
            "cover_blocks",
            "battery_try",
        )
    ):
        per_task.append(
            {
                "task_name": task_name,
                "actual_accepted_seed_sha256": f"{index + 1:064x}",
            }
        )
    payload = {
        "suite": "official9",
        "is_official_nine_task_score": True,
        "per_task": per_task,
        "aggregates": {
            "complete_tasks": 9,
            "expected_tasks": 9,
            "macro_success_rate": 0.0,
            "micro_success_rate": 0.0,
        },
        "failures": [],
        "run_identity": {
            "policy_kind": "fastwam_baseline",
            "policy_name": "fastwam_policy",
            "root_seed": 17,
            "checkpoint_sha256": "a" * 64,
            "warm_git_revision": "b" * 40,
            "action_generation": {
                "action_horizon": 32,
                "replan_steps": 10,
                "num_inference_steps": 10,
            },
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = _validate_result(path, suite="official9")
    assert _accepted_seed_hashes(restored)["put_back_block"] == f"{3:064x}"

    payload["per_task"][0].pop("actual_accepted_seed_sha256")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MatrixError, match="accepted-seed hash"):
        _validate_result(path, suite="official9")


def test_reference_must_be_the_same_seed_fastwam_h32_ode10(tmp_path: Path) -> None:
    per_task = [
        {"task_name": task, "actual_accepted_seed_sha256": "1" * 64}
        for task in (
            "observe_and_pickup",
            "rearrange_blocks",
            "put_back_block",
            "swap_blocks",
            "swap_T",
            "blocks_ranking_try",
            "press_button",
            "cover_blocks",
            "battery_try",
        )
    ]
    summary = {
        "suite": "official9",
        "is_official_nine_task_score": True,
        "per_task": per_task,
        "aggregates": {
            "complete_tasks": 9,
            "expected_tasks": 9,
            "macro_success_rate": 0.2,
            "micro_success_rate": 0.2,
        },
        "failures": [],
        "run_identity": {
            "policy_kind": "fastwam_baseline",
            "policy_name": "fastwam_policy",
            "root_seed": 17,
            "checkpoint_sha256": "a" * 64,
            "warm_git_revision": "b" * 40,
            "action_generation": {
                "action_horizon": 32,
                "replan_steps": 10,
                "num_inference_steps": 10,
            },
        },
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    restored = _validate_result(path, suite="official9")
    _validate_fastwam_reference(restored, expected_root_seed=17)
    restored["run_identity"]["policy_kind"] = "warm"
    with pytest.raises(MatrixError, match="FastWAM baseline"):
        _validate_fastwam_reference(restored, expected_root_seed=17)
