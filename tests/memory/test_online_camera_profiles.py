from __future__ import annotations

from copy import deepcopy

import pytest

from fastwam.memory.online_retrieval import (
    OnlineArtifactContractError,
    validate_online_camera_contract,
)


def _libero() -> dict[str, object]:
    return {
        "schema": "warm.camera-layout",
        "version": 1,
        "source_camera_keys": ["observation.image", "observation.wrist_image"],
        "processor_camera_mapping": {
            "observation.image": "image",
            "observation.wrist_image": "wrist_image",
        },
        "semantic_camera": "observation.image",
        "concat_mode": "horizontal",
        "per_camera_size": [224, 224],
        "decoded_range": [0.0, 1.0],
        "vae_model_range": [-1.0, 1.0],
        "baseline_quantization": "validated_0_1_times_255_to_uint8",
    }


def _robotwin() -> dict[str, object]:
    return {
        "schema": "warm.camera-layout",
        "version": 1,
        "benchmark_profile": "robotwin",
        "source_camera_keys": [
            "observation.head_camera.rgb",
            "observation.left_camera.rgb",
            "observation.right_camera.rgb",
        ],
        "processor_camera_mapping": {
            "observation.head_camera.rgb": "cam_high",
            "observation.left_camera.rgb": "cam_left_wrist",
            "observation.right_camera.rgb": "cam_right_wrist",
        },
        "semantic_camera": "observation.head_camera.rgb",
        "concat_mode": "robotwin",
        "per_camera_size": [240, 320],
        "dino_semantic_size": [224, 224],
        "vae_composite_size": [384, 320],
        "decoded_range": [0.0, 1.0],
        "vae_model_range": [-1.0, 1.0],
        "baseline_quantization": "validated_0_1_times_255_to_uint8",
    }


def test_libero_default_profile_remains_byte_contract_compatible() -> None:
    result = validate_online_camera_contract(_libero())
    assert result == (
        ("observation.image", "observation.wrist_image"),
        ("image", "wrist_image"),
        "image",
        "horizontal",
    )


def test_robotwin_profile_binds_three_camera_dual_path() -> None:
    result = validate_online_camera_contract(
        _robotwin(), benchmark_profile="robotwin"
    )
    assert result == (
        (
            "observation.head_camera.rgb",
            "observation.left_camera.rgb",
            "observation.right_camera.rgb",
        ),
        ("cam_high", "cam_left_wrist", "cam_right_wrist"),
        "cam_high",
        "robotwin",
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.pop("benchmark_profile"), "benchmark_profile"),
        (lambda value: value.update(concat_mode="horizontal"), "robotwin"),
        (lambda value: value.update(per_camera_size=[224, 224]), "240,320"),
        (lambda value: value.update(dino_semantic_size=[240, 320]), "DINO"),
        (
            lambda value: value.update(
                semantic_camera="observation.left_camera.rgb"
            ),
            "first.*head",
        ),
    ],
)
def test_robotwin_camera_profile_fails_closed_on_layout_drift(mutation, match) -> None:
    value = deepcopy(_robotwin())
    mutation(value)
    with pytest.raises(OnlineArtifactContractError, match=match):
        validate_online_camera_contract(value, benchmark_profile="robotwin")


def test_explicit_profile_cannot_be_silently_cross_loaded() -> None:
    with pytest.raises(OnlineArtifactContractError, match="benchmark_profile"):
        validate_online_camera_contract(_robotwin())
    with pytest.raises(OnlineArtifactContractError, match="benchmark_profile"):
        validate_online_camera_contract(_libero(), benchmark_profile="robotwin")
    with pytest.raises(OnlineArtifactContractError, match="exactly"):
        validate_online_camera_contract(_libero(), benchmark_profile="other")
