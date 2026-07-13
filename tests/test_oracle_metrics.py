from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from fastwam.memory.event_bank import EventBank
from fastwam.memory.oracle_metrics import (
    ActionDistanceConfig,
    OracleMetricError,
    OracleQuery,
    action_distance,
    evaluate_oracle_retrieval,
    oracle_metrics_report_to_dict,
)
from fastwam.memory.payload_names import (
    FEATURE_EPISODE_SHA256,
    MODEL_SPACE_ACTION,
    SOURCE_EPISODE_SHA256,
)
from fastwam.memory.schema import EventId


def _config(**overrides) -> ActionDistanceConfig:
    values = {
        "action_dim": 3,
        "arm_dims": (0, 1),
        "gripper_dims": (2,),
        "arm_loss": "mse",
        "arm_weight": 1.0,
        "gripper_state_weight": 1.0,
        "gripper_timing_weight": 1.0,
    }
    values.update(overrides)
    return ActionDistanceConfig(**values)


def _action(arm_value: float, gripper: list[float]) -> np.ndarray:
    result = np.full((len(gripper), 3), arm_value, dtype=np.float32)
    result[:, 2] = np.asarray(gripper, dtype=np.float32)
    return result


def _content_payloads(ids: list[EventId]) -> dict[str, np.ndarray]:
    episode_values: dict[tuple[str, int, int], int] = {}
    for event_id in ids:
        episode_values.setdefault(event_id.episode_key, len(episode_values) + 1)
    source = np.stack(
        [
            np.frombuffer(
                bytes.fromhex(f"{episode_values[event_id.episode_key]:064x}"),
                dtype=np.uint8,
            )
            for event_id in ids
        ]
    )
    feature = np.stack(
        [
            np.frombuffer(
                bytes.fromhex(f"{episode_values[event_id.episode_key] + 100:064x}"),
                dtype=np.uint8,
            )
            for event_id in ids
        ]
    )
    return {
        SOURCE_EPISODE_SHA256: source,
        FEATURE_EPISODE_SHA256: feature,
    }


def _query_hashes() -> dict[str, str]:
    return {
        "source_episode_sha256": "e" * 64,
        "feature_episode_sha256": "f" * 64,
    }


def test_action_distance_reports_arm_state_and_timing_components() -> None:
    target = _action(0.0, [-1.0, -1.0, 1.0, 1.0])
    candidate = _action(1.0, [-1.0, 1.0, 1.0, -1.0])

    distance = action_distance(candidate, target, _config())

    assert distance.arm == pytest.approx(1.0)
    assert distance.gripper_state == pytest.approx(0.5)
    assert distance.gripper_timing == pytest.approx(1.0)
    assert distance.total == pytest.approx(2.5)

    huber = action_distance(
        candidate,
        target,
        _config(
            arm_loss="huber",
            huber_delta=0.5,
            gripper_state_weight=0.0,
            gripper_timing_weight=0.0,
        ),
    )
    assert huber.arm == pytest.approx(0.375)
    assert huber.total == pytest.approx(0.375)


def test_oracle_forces_episode_leave_out_and_finds_best_action_in_top_k() -> None:
    ids = [
        EventId("libero", 0, 8, 0),
        EventId("libero", 0, 8, 16),
        EventId("libero", 0, 9, 0),
        EventId("libero", 0, 10, 0),
    ]
    keys = np.asarray(
        [[1.0, 0.0], [0.999, 0.001], [0.98, 0.02], [0.90, 0.10]],
        dtype=np.float32,
    )
    gt = _action(0.0, [-1.0, -1.0, 1.0, 1.0])
    actions = np.stack(
        [
            gt,  # perfect but leaked: same complete episode
            gt,  # a second leaked start from the same episode
            _action(1.0, [1.0, 1.0, 1.0, 1.0]),  # context top-1, poor action
            gt,  # context rank 2, oracle action
        ]
    )
    bank = EventBank.from_arrays(
        ids,
        keys,
        **{MODEL_SPACE_ACTION: actions, **_content_payloads(ids)},
    )
    recent = _action(0.5, [-1.0, -1.0, 1.0, 1.0])
    query = OracleQuery(
        query_key=np.asarray([1.0, 0.0], dtype=np.float32),
        gt_model_action=gt,
        episode_key=("libero", 0, 8),
        recent_action=recent,
        **_query_hashes(),
    )

    report = evaluate_oracle_retrieval(bank, [query], _config(), top_k=2)
    result = report.per_query[0]

    assert result.candidate_count == 2
    assert result.context_top1_event_id == ids[2]
    assert result.oracle_event_id == ids[3]
    assert result.oracle_rank == 2
    assert result.oracle_distance.total == pytest.approx(0.0)
    assert result.context_top1_distance.total > 0.0
    assert result.recent_baseline_distance.total == pytest.approx(0.25)
    assert result.relative_improvement_vs_context == pytest.approx(1.0)
    assert result.relative_improvement_vs_recent == pytest.approx(1.0)
    assert report.coverage == pytest.approx(1.0)
    assert report.context_top1_mean == result.context_top1_distance
    assert report.oracle_topk_mean == result.oracle_distance
    assert report.recent_baseline_mean == result.recent_baseline_distance
    assert report.relative_improvement_vs_context == pytest.approx(1.0)
    assert report.relative_improvement_vs_recent == pytest.approx(1.0)

    serialized = oracle_metrics_report_to_dict(report)
    assert serialized["per_query"][0]["episode_key"] == {
        "dataset_id": "libero",
        "dataset_index": 0,
        "episode_index": 8,
    }
    assert serialized["per_query"][0]["oracle_event_id"] == ids[3].to_dict()
    assert serialized["per_query"][0]["oracle_distance"]["total"] == 0.0
    # The standard library must accept the result with strict NaN handling.
    json.dumps(serialized, allow_nan=False)


def test_no_legal_candidate_reports_zero_coverage_without_nan() -> None:
    ids = [EventId("libero", 0, 3, 0), EventId("libero", 0, 3, 16)]
    keys = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    gt = _action(0.0, [-1.0, 1.0])
    actions = np.stack([gt, gt])
    bank = EventBank.from_arrays(
        ids,
        keys,
        **{MODEL_SPACE_ACTION: actions, **_content_payloads(ids)},
    )
    query = OracleQuery(
        np.asarray([1.0, 0.0], dtype=np.float32),
        gt,
        ("libero", 0, 3),
        **_query_hashes(),
        recent_action=_action(0.5, [-1.0, 1.0]),
    )

    report = evaluate_oracle_retrieval(bank, [query], _config(), top_k=32)
    result = report.per_query[0]

    assert report.coverage == 0.0
    assert report.covered_query_count == 0
    assert report.context_top1_mean is None
    assert report.oracle_topk_mean is None
    assert report.relative_improvement_vs_context is None
    assert report.relative_improvement_vs_recent is None
    assert report.recent_baseline_query_count == 1
    assert report.recent_comparable_query_count == 0
    assert report.recent_baseline_mean.total == pytest.approx(0.25)
    assert result.candidate_count == 0
    assert result.context_top1_event_id is None
    assert result.oracle_event_id is None
    assert result.oracle_distance is None

    serialized = oracle_metrics_report_to_dict(report)
    assert serialized["context_top1_mean"] is None
    assert serialized["per_query"][0]["context_top1_event_id"] is None
    assert serialized["per_query"][0]["oracle_distance"] is None
    assert serialized["per_query"][0]["recent_baseline_distance"]["total"] == pytest.approx(
        0.25
    )


def test_oracle_excludes_duplicate_feature_content_across_episode_ids() -> None:
    ids = [EventId("libero", 0, 1, 0), EventId("libero", 0, 2, 0)]
    keys = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    gt = _action(0.0, [-1.0, 1.0])
    actions = np.stack([gt, _action(1.0, [-1.0, 1.0])])
    source_hashes = np.stack(
        [
            np.frombuffer(bytes.fromhex(value * 64), dtype=np.uint8)
            for value in ("1", "2")
        ]
    )
    feature_hashes = np.stack(
        [
            np.frombuffer(bytes.fromhex(value * 64), dtype=np.uint8)
            for value in ("a", "b")
        ]
    )
    bank = EventBank.from_arrays(
        ids,
        keys,
        **{
            MODEL_SPACE_ACTION: actions,
            SOURCE_EPISODE_SHA256: source_hashes,
            FEATURE_EPISODE_SHA256: feature_hashes,
        },
    )
    query = OracleQuery(
        np.asarray([1.0, 0.0], dtype=np.float32),
        gt,
        ("libero", 0, 9),
        source_episode_sha256="f" * 64,
        feature_episode_sha256="a" * 64,
    )

    report = evaluate_oracle_retrieval(bank, [query], _config(), top_k=2)

    assert report.per_query[0].candidate_count == 1
    assert report.per_query[0].context_top1_event_id == ids[1]


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"arm_dims": (0, 2), "gripper_dims": (2,)}, OracleMetricError),
        ({"arm_dims": (0,), "gripper_dims": (2,)}, OracleMetricError),
        ({"arm_dims": (0, 1, 3), "gripper_dims": (2,)}, OracleMetricError),
        ({"arm_loss": "l1"}, OracleMetricError),
        ({"huber_delta": float("nan")}, OracleMetricError),
    ],
)
def test_action_config_rejects_ambiguous_dimensions_or_nonfinite_values(
    kwargs, exception
) -> None:
    with pytest.raises(exception):
        _config(**kwargs)


def test_query_and_evaluator_validate_shape_and_finite_contracts() -> None:
    gt = _action(0.0, [-1.0, 1.0])
    with pytest.raises(OracleMetricError, match="finite"):
        OracleQuery(
            np.asarray([np.nan, 0.0], dtype=np.float32),
            gt,
            ("libero", 0, 0),
            **_query_hashes(),
        )
    with pytest.raises(OracleMetricError, match="rank 2"):
        OracleQuery(
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            ("libero", 0, 0),
            **_query_hashes(),
        )
    with pytest.raises(OracleMetricError, match="recent_action must match"):
        OracleQuery(
            np.asarray([1.0, 0.0], dtype=np.float32),
            gt,
            ("libero", 0, 0),
            **_query_hashes(),
            recent_action=np.zeros((3, 3), dtype=np.float32),
        )

    ids = [EventId("libero", 0, 1, 0)]
    keys = np.asarray([[1.0, 0.0]], dtype=np.float32)
    malformed_actions = np.zeros((1, 2), dtype=np.float32)
    malformed_bank = EventBank.from_arrays(
        ids,
        keys,
        **{MODEL_SPACE_ACTION: malformed_actions, **_content_payloads(ids)},
    )
    query = OracleQuery(
        np.asarray([1.0, 0.0], dtype=np.float32),
        gt,
        ("libero", 0, 0),
        **_query_hashes(),
    )
    with pytest.raises(OracleMetricError, match="events, horizon, action_dim"):
        evaluate_oracle_retrieval(malformed_bank, [query], _config())

    valid_bank = EventBank.from_arrays(
        ids,
        keys,
        **{MODEL_SPACE_ACTION: np.stack([gt]), **_content_payloads(ids)},
    )
    wrong_key_query = OracleQuery(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        gt,
        ("libero", 0, 0),
        **_query_hashes(),
    )
    with pytest.raises(OracleMetricError, match="key must have shape"):
        evaluate_oracle_retrieval(valid_bank, [wrong_key_query], _config())

    with pytest.raises(OracleMetricError, match="must not be empty"):
        evaluate_oracle_retrieval(valid_bank, [], _config())


def test_report_serialization_rejects_top_level_and_nested_nan() -> None:
    ids = [EventId("libero", 0, 1, 0)]
    keys = np.asarray([[1.0, 0.0]], dtype=np.float32)
    gt = _action(0.0, [-1.0, 1.0])
    bank = EventBank.from_arrays(
        ids,
        keys,
        **{MODEL_SPACE_ACTION: np.stack([gt]), **_content_payloads(ids)},
    )
    query = OracleQuery(
        np.asarray([1.0, 0.0], dtype=np.float32),
        gt,
        ("libero", 0, 0),
        **_query_hashes(),
    )
    report = evaluate_oracle_retrieval(bank, [query], _config())

    with pytest.raises(OracleMetricError, match="coverage must be finite"):
        oracle_metrics_report_to_dict(replace(report, coverage=float("nan")))

    result = report.per_query[0]
    invalid_distance = replace(result.context_top1_distance, total=float("nan"))
    invalid_result = replace(result, context_top1_distance=invalid_distance)
    with pytest.raises(OracleMetricError, match=r"context_top1_distance.total.*finite"):
        oracle_metrics_report_to_dict(replace(report, per_query=(invalid_result,)))
