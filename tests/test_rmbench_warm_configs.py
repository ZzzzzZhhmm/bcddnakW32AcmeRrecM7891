from __future__ import annotations

from pathlib import Path

import pytest

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from fastwam.memory.processor_contract import extract_m1_robotwin_processor_recipe


ROOT = Path(__file__).resolve().parents[1]


def test_rmbench_data_profile_is_native_three_camera_14d() -> None:
    raw = OmegaConf.load(ROOT / "configs" / "data" / "rmbench_3cam.yaml")
    wrapped = OmegaConf.create({"data": raw})
    OmegaConf.resolve(wrapped)
    value = OmegaConf.to_container(wrapped.data, resolve=True)
    recipe = extract_m1_robotwin_processor_recipe(value)
    assert recipe["processor"]["action_output_dim"] == 14
    assert recipe["processor"]["norm_default_mode"] == "z-score"
    assert recipe["concat_multi_camera"] == "robotwin"
    assert recipe["shape_meta"]["images"][0]["raw_shape"] == [3, 240, 320]


def test_rmbench_model_context_width_matches_nine_task_vocabulary() -> None:
    value = OmegaConf.load(ROOT / "configs" / "model" / "warm_robotwin.yaml")
    assert int(value.retrospection.context_dim) == 768 + 9
    assert int(value.retrospection.action_dim) == 14
    assert int(value.retrospection.action_horizon) == 32
    assert int(value.retrospection.timing_dim) == 8
    assert value.retrospection.canonical_action_mode == "start_proprio_delta"
    assert list(value.retrospection.canonical_gripper_dims) == [6, 13]


def test_fastwam_baseline_profile_has_no_candidate_adapter() -> None:
    baseline = OmegaConf.load(
        ROOT / "configs" / "data" / "rmbench_3cam_train_dev.yaml"
    )
    assert "warm_candidates" not in baseline
    train_task = (
        ROOT / "configs" / "task" / "rmbench_fastwam_3cam384_1e-4.yaml"
    ).read_text(encoding="utf-8")
    online_task = (
        ROOT / "configs" / "task" / "rmbench_fastwam_online_3cam384.yaml"
    ).read_text(encoding="utf-8")
    assert "rmbench_3cam_train_dev" in train_task
    assert "override /model: fastwam" in train_task
    assert "override /model: fastwam" in online_task
