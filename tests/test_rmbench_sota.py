from __future__ import annotations

from pathlib import Path

import pytest

from fastwam.benchmarks.rmbench_sota import (
    RMBENCH_DATA_PROFILES,
    RMBENCH_SOTA_ROOT_SEED,
    data_profile,
    load_registry_document,
    load_task_profiles,
)
from fastwam.datasets.rmbench.constants import OFFICIAL_RMBENCH_TASKS
from scripts.plan_warm_rmbench_sota import main as plan_main


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "rmbench" / "sota_v1.json"


def test_data_profiles_are_closed_and_leave_true_development_sets() -> None:
    assert tuple(RMBENCH_DATA_PROFILES) == (
        "official50-dev45",
        "scale200-dev190",
        "scale500-dev480",
    )
    assert data_profile("official50-dev45").train_per_task == 45
    assert data_profile("scale200-dev190").train_per_task == 190
    assert data_profile("scale500-dev480").train_per_task == 480
    with pytest.raises(ValueError, match="unsupported RMBench data profile"):
        data_profile("untracked-custom")


def test_sota_registry_binds_seed_task_order_and_closed_ranges() -> None:
    document = load_registry_document(REGISTRY)
    tasks = load_task_profiles(REGISTRY)

    assert document["root_seed"] == RMBENCH_SOTA_ROOT_SEED == 3407
    assert tuple(tasks) == OFFICIAL_RMBENCH_TASKS
    assert tasks["blocks_ranking_try"].memory_regime == "M(n)"
    assert tasks["blocks_ranking_try"].recent_event_capacity > tasks[
        "observe_and_pickup"
    ].recent_event_capacity
    assert all(profile.top_k <= 32 for profile in tasks.values())


def test_planner_profile_override_is_not_shadowed_by_registry(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert plan_main(
        [
            "--registry",
            str(REGISTRY),
            "--data-profile",
            "official50-dev45",
            "--format",
            "tsv",
        ]
    ) == 0
    fields = capsys.readouterr().out.strip().split("\t")
    assert fields[:2] == ["3407", "official50-dev45"]
