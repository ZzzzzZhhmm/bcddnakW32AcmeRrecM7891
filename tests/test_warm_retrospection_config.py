from __future__ import annotations

import math
from pathlib import Path

import pytest

from fastwam.models.warm.retrospection_config import (
    WarmRetrospectionConfig,
    WarmRetrospectionConfigError,
)


def _config_payload() -> dict[str, object]:
    return {
        "context_dim": 10,
        "semantic_dim": 8,
        "video_dim": 12,
        "text_dim": 16,
        "proprio_dim": 7,
        "action_dim": 3,
        "action_horizon": 4,
        "timing_dim": 4,
        "gist_dim": 16,
        "event_model_dim": 8,
        "bridge_heads": 2,
        "gist_heads": 4,
        "event_heads": 2,
        "reranker_hidden_dim": 12,
        "gate_hidden_dim": 5,
        "episode_action_chunk_size": 2,
        "canonical_action_mode": "none",
        "canonical_gripper_dims": (),
        "video_adapter_rank": 2,
        "video_adapter_layers": (1, 3),
        "video_adapter_scale": 1.0,
        "source_sigma_min": 0.2,
        "residual_rho": 0.25,
        "consequence_weight": 1.0,
        "support_weight": 0.1,
        "magnitude_weight": 0.25,
        "selection_temperature": 0.5,
        "minimum_candidate_probability": 0.06,
        "maximum_candidate_entropy": 0.94,
        "inference_source_gate_threshold": 0.15,
        "stagnation_decay": 3.0,
        "stagnation_hard_threshold": 0.75,
        "thread_score_weight": 0.75,
        "thread_switch_penalty": 0.25,
        "thread_reuse_penalty": 1.0,
        "thread_backtrack_tolerance": 16,
        "thread_forward_window": 256,
        "thread_max_null_steps": 3,
        "utility_effect_weight": 1.0,
        "utility_temperature": 0.5,
        "gate_effect_weight": 1.0,
        "gate_temperature": 0.75,
        "loss_retrieval": 0.1,
        "loss_bridge": 0.05,
        "loss_gist": 0.05,
        "loss_effect": 0.05,
        "loss_gate": 0.05,
        "loss_adaptation": 0.05,
        "corruption_normal_weight": 0.5,
        "corruption_drop_weight": 0.125,
        "corruption_null_weight": 0.125,
        "corruption_hard_negative_weight": 0.25,
        "corruption_seed": (1 << 64) - 1,
    }


def test_retrospection_config_round_trip_is_closed_and_lossless() -> None:
    payload = _config_payload()
    config = WarmRetrospectionConfig.from_dict(payload)

    assert config.to_dict() == payload
    assert config.episode_action_summary_dim == 3 * config.action_dim + 4
    assert WarmRetrospectionConfig.from_dict(config.to_dict()) == config


def test_retrospection_config_json_domain_normalizes_sequence_containers() -> None:
    config = WarmRetrospectionConfig.from_dict(_config_payload())

    serialized = config.to_json_dict()

    assert serialized["video_adapter_layers"] == [1, 3]
    assert serialized["canonical_gripper_dims"] == []
    assert WarmRetrospectionConfig.from_dict(serialized) == config


def test_retrospection_config_rejects_unknown_and_missing_fields() -> None:
    unknown = _config_payload()
    unknown["silently_ignored_architecture_knob"] = 1
    with pytest.raises(
        WarmRetrospectionConfigError,
        match="unknown retrospection config fields",
    ):
        WarmRetrospectionConfig.from_dict(unknown)

    missing = _config_payload()
    missing.pop("action_dim")
    with pytest.raises(
        WarmRetrospectionConfigError,
        match="missing required fields",
    ):
        WarmRetrospectionConfig.from_dict(missing)

    with pytest.raises(TypeError, match="must be a mapping"):
        WarmRetrospectionConfig.from_dict([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("context_dim", True, "positive integer"),
        ("action_horizon", 0, "positive integer"),
        ("timing_dim", -1, "positive integer"),
        ("episode_action_chunk_size", 5, "cannot exceed action_horizon"),
        ("video_adapter_rank", 12, "smaller than video_dim"),
        ("video_adapter_layers", (1, 1), "must not contain duplicates"),
        ("canonical_action_mode", "learned", "must be 'none' or"),
        (
            "canonical_gripper_dims",
            (3,),
            "must contain valid action indices",
        ),
        ("video_adapter_scale", 0.0, "positive"),
        ("source_sigma_min", 0.0, "positive"),
        ("residual_rho", float("nan"), "finite"),
        ("utility_temperature", float("inf"), "finite"),
        ("gate_temperature", -0.1, "positive"),
        ("loss_gate", -1.0, "non-negative"),
        ("consequence_weight", -0.01, "non-negative"),
        ("corruption_seed", True, "unsigned 64-bit"),
        ("corruption_seed", -1, "unsigned 64-bit"),
        ("corruption_seed", 1 << 64, "unsigned 64-bit"),
    ],
)
def test_retrospection_config_rejects_invalid_scalar_contracts(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _config_payload()
    payload[field] = value
    with pytest.raises(WarmRetrospectionConfigError, match=message):
        WarmRetrospectionConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("semantic_dim", 7, "semantic_dim=7.*heads=2"),
        ("gist_dim", 14, "gist_dim=14.*heads=4"),
        ("event_model_dim", 10, "event_model_dim=10.*heads=4"),
    ],
)
def test_retrospection_config_rejects_attention_width_mismatch(
    field: str,
    value: int,
    message: str,
) -> None:
    payload = _config_payload()
    if field == "event_model_dim":
        payload["event_heads"] = 4
    payload[field] = value
    with pytest.raises(WarmRetrospectionConfigError, match=message):
        WarmRetrospectionConfig.from_dict(payload)


def test_retrospection_config_output_contains_only_finite_numbers() -> None:
    config = WarmRetrospectionConfig.from_dict(_config_payload())
    for value in config.to_dict().values():
        if isinstance(value, float):
            assert math.isfinite(value)


def test_start_proprio_canonicalization_contract_is_closed() -> None:
    payload = _config_payload()
    payload.update(
        {
            "proprio_dim": 3,
            "canonical_action_mode": "start_proprio_delta",
            "canonical_gripper_dims": [2],
        }
    )
    config = WarmRetrospectionConfig.from_dict(payload)

    assert config.canonical_gripper_dims == (2,)
    assert WarmRetrospectionConfig.from_dict(config.to_dict()) == config

    mismatched = dict(payload, proprio_dim=4)
    with pytest.raises(
        WarmRetrospectionConfigError,
        match="requires proprio_dim == action_dim",
    ):
        WarmRetrospectionConfig.from_dict(mismatched)

    duplicate = dict(payload, canonical_gripper_dims=[2, 2])
    with pytest.raises(
        WarmRetrospectionConfigError,
        match="must not contain duplicates",
    ):
        WarmRetrospectionConfig.from_dict(duplicate)

    noncanonical = _config_payload()
    noncanonical["canonical_gripper_dims"] = [2]
    with pytest.raises(
        WarmRetrospectionConfigError,
        match="must be empty",
    ):
        WarmRetrospectionConfig.from_dict(noncanonical)


def test_full_training_task_enables_mot_activation_checkpointing() -> None:
    task = (
        Path(__file__).resolve().parents[1]
        / "configs/task/libero_warm_2cam224_1e-4.yaml"
    ).read_text(encoding="utf-8")
    assert "mot_checkpoint_mixed_attn: true" in task
