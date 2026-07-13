from __future__ import annotations

from pathlib import Path

import pytest
import yaml
OmegaConf = pytest.importorskip("omegaconf").OmegaConf

from fastwam.memory.processor_contract import (
    M1_ROBOTWIN_PROCESSOR_RECIPE,
    ProcessorContractError,
    extract_m1_robotwin_processor_recipe,
)


def _config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "data" / "robotwin.yaml"
    raw = OmegaConf.load(path)
    wrapped = OmegaConf.create({"data": raw})
    OmegaConf.resolve(wrapped)
    value = OmegaConf.to_container(wrapped.data, resolve=True)
    assert isinstance(value, dict)
    return value


def test_robotwin_processor_recipe_binds_native_shapes_and_layout() -> None:
    recipe = extract_m1_robotwin_processor_recipe(_config())
    assert recipe["schema"] == M1_ROBOTWIN_PROCESSOR_RECIPE
    assert recipe["video_size"] == [384, 320]
    assert recipe["concat_multi_camera"] == "robotwin"


def test_rmbench_largeview_raw_resolution_is_bound_without_fake_upsampling() -> None:
    value = _config()
    for item in value["train"]["shape_meta"]["images"]:
        item["raw_shape"] = [3, 240, 320]
    value["train"]["processor"]["shape_meta"] = value["train"]["shape_meta"]
    recipe = extract_m1_robotwin_processor_recipe(value)
    assert {
        tuple(item["raw_shape"]) for item in recipe["shape_meta"]["images"]
    } == {(3, 240, 320)}
    assert recipe["processor"]["action_output_dim"] == 14
    assert recipe["processor"]["proprio_output_dim"] == 14
    assert recipe["processor"]["norm_default_mode"] == "z-score"
    assert [item["key"] for item in recipe["shape_meta"]["images"]] == [
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ]


def test_robotwin_processor_recipe_rejects_libero_dimensions() -> None:
    value = _config()
    value["train"]["shape_meta"]["action"][0]["shape"] = 7
    with pytest.raises(ProcessorContractError, match="native default 14D"):
        extract_m1_robotwin_processor_recipe(value)
