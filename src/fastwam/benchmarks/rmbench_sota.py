"""Validated score-oriented RMBench training and evaluation profiles.

The official benchmark protocol remains defined in :mod:`fastwam.benchmarks.rmbench`.
This module owns WARM's *training* choices: deterministic data volumes, one
root seed, task-specific replay capacity, and closed hyperparameter ranges.
Keeping those choices in one machine-readable registry prevents shell launch
scripts from silently drifting apart.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from fastwam.datasets.rmbench.constants import OFFICIAL_RMBENCH_TASKS


RMBENCH_SOTA_ROOT_SEED = 3407


@dataclass(frozen=True)
class RMBenchDataProfile:
    name: str
    episodes_per_task: int
    dev_per_task: int
    strict_official_source: bool

    @property
    def train_per_task(self) -> int:
        return self.episodes_per_task - self.dev_per_task

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("RMBench data-profile name must be normalized")
        if self.episodes_per_task < 2:
            raise ValueError("episodes_per_task must be at least two")
        if self.dev_per_task < 1 or self.dev_per_task >= self.episodes_per_task:
            raise ValueError("data profiles require non-empty train and dev splits")
        if not isinstance(self.strict_official_source, bool):
            raise TypeError("strict_official_source must be a boolean")


RMBENCH_DATA_PROFILES: Mapping[str, RMBenchDataProfile] = {
    "official50-dev45": RMBenchDataProfile(
        name="official50-dev45",
        episodes_per_task=50,
        dev_per_task=5,
        strict_official_source=True,
    ),
    "scale200-dev190": RMBenchDataProfile(
        name="scale200-dev190",
        episodes_per_task=200,
        dev_per_task=10,
        strict_official_source=False,
    ),
    "scale500-dev480": RMBenchDataProfile(
        name="scale500-dev480",
        episodes_per_task=500,
        dev_per_task=20,
        strict_official_source=False,
    ),
}


def data_profile(name: str) -> RMBenchDataProfile:
    try:
        return RMBENCH_DATA_PROFILES[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported RMBench data profile {name!r}; expected one of "
            f"{sorted(RMBENCH_DATA_PROFILES)}"
        ) from exc


@dataclass(frozen=True)
class RMBenchTaskProfile:
    task_name: str
    memory_regime: str
    train_steps: int
    recent_event_capacity: int
    action_summary_capacity: int
    replan_steps: int
    inference_steps: int
    top_k: int

    def __post_init__(self) -> None:
        if self.task_name not in OFFICIAL_RMBENCH_TASKS:
            raise ValueError(f"unknown official RMBench task {self.task_name!r}")
        if self.memory_regime not in {"M(1)", "M(n)"}:
            raise ValueError("memory_regime must be M(1) or M(n)")
        for field in (
            "train_steps",
            "recent_event_capacity",
            "action_summary_capacity",
            "replan_steps",
            "inference_steps",
            "top_k",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.inference_steps not in {2, 4, 8, 10}:
            raise ValueError("inference_steps must be one of 2, 4, 8, 10")
        if self.recent_event_capacity > 32:
            raise ValueError("recent_event_capacity exceeds the closed limit 32")
        if self.action_summary_capacity > 16:
            raise ValueError("action_summary_capacity exceeds the closed limit 16")
        if self.replan_steps > 32:
            raise ValueError("replan_steps cannot exceed action horizon 32")
        if self.top_k > 32:
            raise ValueError("top_k exceeds the fixed K=32 candidate-cache contract")


def load_task_profiles(path: str | Path) -> Mapping[str, RMBenchTaskProfile]:
    """Load the closed nine-task registry used by server launchers."""

    source = Path(path).expanduser().resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "root_seed",
        "shared",
        "tasks",
    }:
        raise ValueError("RMBench SOTA registry has unexpected top-level fields")
    if document["schema_version"] != 1:
        raise ValueError("unsupported RMBench SOTA registry schema")
    if document["root_seed"] != RMBENCH_SOTA_ROOT_SEED:
        raise ValueError(
            f"RMBench SOTA registry root_seed must be {RMBENCH_SOTA_ROOT_SEED}"
        )
    shared = document["shared"]
    if not isinstance(shared, dict) or set(shared) != {
        "data_profile",
        "train_steps",
        "sampler_mode",
        "event_boost",
        "recent_event_capacity",
        "action_summary_capacity",
        "replan_steps",
    }:
        raise ValueError("RMBench SOTA registry shared profile is incomplete")
    data_profile(str(shared["data_profile"]))
    if (
        isinstance(shared["train_steps"], bool)
        or not isinstance(shared["train_steps"], int)
        or shared["train_steps"] <= 0
    ):
        raise ValueError("shared.train_steps must be a positive integer")
    if shared["sampler_mode"] != "rmbench_task_event_balanced":
        raise ValueError("shared.sampler_mode must enable task/event balancing")
    event_boost = shared["event_boost"]
    if (
        isinstance(event_boost, bool)
        or not isinstance(event_boost, (int, float))
        or not 1.0 <= float(event_boost) <= 4.0
    ):
        raise ValueError("shared.event_boost must lie in [1, 4]")
    for field, minimum in (
        ("recent_event_capacity", 2),
        ("action_summary_capacity", 1),
        ("replan_steps", 1),
    ):
        value = shared[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"shared.{field} must be an integer >= {minimum}")
    if shared["replan_steps"] > 32:
        raise ValueError("shared.replan_steps cannot exceed action horizon 32")

    tasks = document["tasks"]
    if not isinstance(tasks, dict) or tuple(tasks) != OFFICIAL_RMBENCH_TASKS:
        raise ValueError(
            "RMBench SOTA registry tasks must use the exact official paper order"
        )
    parsed: dict[str, RMBenchTaskProfile] = {}
    required = {
        "memory_regime",
        "train_steps",
        "recent_event_capacity",
        "action_summary_capacity",
        "replan_steps",
        "inference_steps",
        "top_k",
    }
    for task_name, raw in tasks.items():
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError(f"task profile {task_name!r} has unexpected fields")
        parsed[task_name] = RMBenchTaskProfile(task_name=task_name, **raw)
    return parsed


def load_registry_document(path: str | Path) -> dict[str, Any]:
    """Return a validated plain registry document for CLI orchestration."""

    source = Path(path).expanduser().resolve()
    profiles = load_task_profiles(source)
    document = json.loads(source.read_text(encoding="utf-8"))
    if tuple(profiles) != OFFICIAL_RMBENCH_TASKS:  # pragma: no cover - defensive
        raise AssertionError("validated task order changed while reading registry")
    return document


__all__ = [
    "RMBENCH_DATA_PROFILES",
    "RMBENCH_SOTA_ROOT_SEED",
    "RMBenchDataProfile",
    "RMBenchTaskProfile",
    "data_profile",
    "load_registry_document",
    "load_task_profiles",
]
