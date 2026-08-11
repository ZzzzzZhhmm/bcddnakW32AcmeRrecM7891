#!/usr/bin/env python3
"""Fail-closed qualification for a short shared/specialist WARM run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "warm.training-smoke-qualification"
VERSION = 1
METRIC_SCHEMA = "warm.training-metrics"
COMMON_GRADIENT_GROUPS = (
    "warm_grad_proprio_bridge",
    "warm_grad_semantic_bridge",
    "warm_grad_gist",
    "warm_grad_event_adapter",
    "warm_grad_reranker",
    "warm_grad_source_gate",
    "warm_grad_episode_memory",
    "warm_grad_video_adapters",
)
WARM_LOSSES = (
    "loss_warm_retrieval",
    "loss_warm_bridge",
    "loss_warm_gist",
    "loss_warm_effect",
    "loss_warm_gate",
    "loss_warm_adaptation",
)


class TrainingSmokeQualificationError(ValueError):
    """Raised when a short run does not prove that WARM can learn safely."""


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingSmokeQualificationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingSmokeQualificationError(f"{field} must be finite")
    return result


def load_training_metrics(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"training metrics do not exist: {source}")
    records: list[dict[str, Any]] = []
    previous_step = -1
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise TrainingSmokeQualificationError(
                f"invalid JSON at {source}:{line_number}"
            ) from error
        if not isinstance(record, dict):
            raise TrainingSmokeQualificationError(
                f"metric record {line_number} must be a mapping"
            )
        if record.get("schema") != METRIC_SCHEMA or record.get("version") != 1:
            raise TrainingSmokeQualificationError(
                f"metric record {line_number} has an incompatible schema"
            )
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= previous_step:
            raise TrainingSmokeQualificationError(
                "training metric steps must be positive and strictly increasing"
            )
        previous_step = step
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise TrainingSmokeQualificationError(
                f"metric record {line_number} lacks a metrics mapping"
            )
        for field in ("loss", "grad_norm", "learning_rate", "steps_per_second"):
            _finite_number(record.get(field), field=f"record[{line_number}].{field}")
        for key, value in metrics.items():
            if not isinstance(key, str) or not key:
                raise TrainingSmokeQualificationError("metric names must be strings")
            _finite_number(value, field=f"record[{line_number}].metrics.{key}")
        records.append(record)
    if not records:
        raise TrainingSmokeQualificationError("training metrics are empty")
    return records


def qualify_training_smoke(
    records: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    minimum_records: int = 10,
    minimum_step_span: int = 100,
    minimum_nonzero_gradient_fraction: float = 0.8,
    minimum_gradient_norm: float = 1.0e-12,
    minimum_source_exposure: float = 1.0e-4,
    maximum_forced_source_leak: float = 1.0e-7,
) -> dict[str, Any]:
    if stage not in {"shared", "specialist"}:
        raise ValueError("stage must be shared or specialist")
    if minimum_records < 1 or minimum_step_span < 0:
        raise ValueError("minimum record/span thresholds are invalid")
    if not 0.0 <= minimum_nonzero_gradient_fraction <= 1.0:
        raise ValueError("minimum_nonzero_gradient_fraction must be in [0,1]")
    if minimum_gradient_norm < 0.0 or minimum_source_exposure < 0.0:
        raise ValueError("gradient/source thresholds must be non-negative")
    if maximum_forced_source_leak < 0.0:
        raise ValueError("maximum_forced_source_leak must be non-negative")
    if len(records) < minimum_records:
        raise TrainingSmokeQualificationError(
            f"smoke run has {len(records)} metric records, requires {minimum_records}"
        )
    steps = [int(record["step"]) for record in records]
    if steps[-1] - steps[0] < minimum_step_span:
        raise TrainingSmokeQualificationError(
            f"smoke metric span is {steps[-1] - steps[0]}, requires {minimum_step_span}"
        )
    for index, record in enumerate(records):
        if record.get("stage") != stage:
            raise TrainingSmokeQualificationError(
                f"metric record {index} stage {record.get('stage')!r} != {stage!r}"
            )

    metrics = [record["metrics"] for record in records]
    required_gradients = list(COMMON_GRADIENT_GROUPS)
    if stage == "shared":
        required_gradients.insert(0, "warm_grad_action_expert")
    gradient_evidence: dict[str, Any] = {}
    for name in required_gradients:
        values = [
            _finite_number(row.get(name), field=name)
            if name in row
            else None
            for row in metrics
        ]
        if any(value is None for value in values):
            raise TrainingSmokeQualificationError(
                f"smoke metrics lack required gradient group {name}"
            )
        numeric = [float(value) for value in values if value is not None]
        fraction = sum(value > minimum_gradient_norm for value in numeric) / len(numeric)
        if fraction < minimum_nonzero_gradient_fraction:
            raise TrainingSmokeQualificationError(
                f"{name} nonzero fraction {fraction:.3f} is below "
                f"{minimum_nonzero_gradient_fraction:.3f}"
            )
        gradient_evidence[name] = {
            "nonzero_fraction": fraction,
            "minimum": min(numeric),
            "maximum": max(numeric),
        }

    for name in WARM_LOSSES:
        if any(name not in row for row in metrics):
            raise TrainingSmokeQualificationError(
                f"smoke metrics lack required WARM loss {name}"
            )

    def maximum(name: str) -> float:
        if any(name not in row for row in metrics):
            raise TrainingSmokeQualificationError(
                f"smoke metrics lack required field {name}"
            )
        return max(_finite_number(row[name], field=name) for row in metrics)

    positive_rate = maximum("warm_gate_positive_row_rate")
    negative_rate = maximum("warm_gate_negative_row_rate")
    if positive_rate <= 0.0 or negative_rate <= 0.0:
        raise TrainingSmokeQualificationError(
            "gate supervision did not observe both helpful and rejectable rows"
        )
    source_exposure = maximum("warm_normal_source_exposure_mean")
    if source_exposure < minimum_source_exposure:
        raise TrainingSmokeQualificationError(
            f"memory source exposure {source_exposure:.3e} is below "
            f"{minimum_source_exposure:.3e}"
        )
    leak = maximum("warm_forced_source_leak_mean")
    if leak > maximum_forced_source_leak:
        raise TrainingSmokeQualificationError(
            f"forced-negative source leak {leak:.3e} exceeds "
            f"{maximum_forced_source_leak:.3e}"
        )
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "qualified",
        "stage": stage,
        "record_count": len(records),
        "first_step": steps[0],
        "last_step": steps[-1],
        "step_span": steps[-1] - steps[0],
        "gradient_evidence": gradient_evidence,
        "gate_positive_row_rate_max": positive_rate,
        "gate_negative_row_rate_max": negative_rate,
        "normal_source_exposure_max": source_exposure,
        "forced_source_leak_max": leak,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--stage", choices=("shared", "specialist"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-records", type=int, default=10)
    parser.add_argument("--minimum-step-span", type=int, default=100)
    args = parser.parse_args()
    report = qualify_training_smoke(
        load_training_metrics(args.metrics),
        stage=args.stage,
        minimum_records=args.minimum_records,
        minimum_step_span=args.minimum_step_span,
    )
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(f"TRAINING_SMOKE_QUALIFIED stage={args.stage} output={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
