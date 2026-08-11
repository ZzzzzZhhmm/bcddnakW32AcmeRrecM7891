from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qualify_warm_rmbench_training_smoke",
    PROJECT / "scripts" / "qualify_warm_rmbench_training_smoke.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record(step: int, *, stage: str = "shared") -> dict:
    gradients = {
        name: 0.1
        for name in (*MODULE.COMMON_GRADIENT_GROUPS, "warm_grad_action_expert")
    }
    return {
        "schema": MODULE.METRIC_SCHEMA,
        "version": 1,
        "step": step,
        "stage": stage,
        "loss": 1.0,
        "grad_norm": 0.5,
        "learning_rate": 1.0e-5,
        "steps_per_second": 0.1,
        "metrics": {
            **gradients,
            **{name: 0.2 for name in MODULE.WARM_LOSSES},
            "warm_gate_positive_row_rate": 0.25,
            "warm_gate_negative_row_rate": 0.75,
            "warm_normal_source_exposure_mean": 0.1,
            "warm_forced_source_leak_mean": 0.0,
        },
    }


def test_training_smoke_gate_accepts_complete_shared_evidence(tmp_path: Path) -> None:
    records = [_record(step) for step in range(10, 121, 10)]
    path = tmp_path / "training_metrics.jsonl"
    path.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n",
        encoding="utf-8",
    )

    report = MODULE.qualify_training_smoke(
        MODULE.load_training_metrics(path), stage="shared"
    )

    assert report["status"] == "qualified"
    assert report["record_count"] == 12
    assert report["gradient_evidence"]["warm_grad_action_expert"][
        "nonzero_fraction"
    ] == 1.0


def test_specialist_gate_does_not_require_frozen_action_expert() -> None:
    records = [_record(step, stage="specialist") for step in range(10, 121, 10)]
    for record in records:
        record["metrics"].pop("warm_grad_action_expert")

    report = MODULE.qualify_training_smoke(records, stage="specialist")

    assert "warm_grad_action_expert" not in report["gradient_evidence"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda rows: rows[0]["metrics"].pop("warm_grad_gist"), "gradient group"),
        (
            lambda rows: [
                row["metrics"].__setitem__("warm_grad_source_gate", 0.0)
                for row in rows
            ],
            "nonzero fraction",
        ),
        (
            lambda rows: [
                row["metrics"].__setitem__("warm_gate_positive_row_rate", 0.0)
                for row in rows
            ],
            "both helpful",
        ),
        (
            lambda rows: rows[0]["metrics"].__setitem__(
                "warm_forced_source_leak_mean", 0.1
            ),
            "source leak",
        ),
    ],
)
def test_training_smoke_gate_rejects_missing_or_dead_learning_paths(
    mutation, match: str
) -> None:
    records = [_record(step) for step in range(10, 121, 10)]
    mutation(records)

    with pytest.raises(MODULE.TrainingSmokeQualificationError, match=match):
        MODULE.qualify_training_smoke(records, stage="shared")


def test_metric_loader_rejects_duplicate_or_nonfinite_steps(tmp_path: Path) -> None:
    records = [_record(10), _record(10)]
    path = tmp_path / "bad.jsonl"
    path.write_text(
        "\n".join(json.dumps(item) for item in records) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        MODULE.TrainingSmokeQualificationError, match="strictly increasing"
    ):
        MODULE.load_training_metrics(path)
