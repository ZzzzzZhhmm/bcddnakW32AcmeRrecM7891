from __future__ import annotations

from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from fastwam.utils.samplers import ResumableTaskEventBalancedSampler


class _StratifiedDataset:
    def __init__(self) -> None:
        self.strata = (
            *(("long", False),) * 90,
            *(("long", True),) * 10,
            *(("short", False),) * 9,
            ("short", True),
        )

    def __len__(self) -> int:
        return len(self.strata)

    def sampling_strata(self):
        return self.strata


def test_balanced_sampler_is_deterministic_resumable_and_task_balanced() -> None:
    dataset = _StratifiedDataset()
    first = ResumableTaskEventBalancedSampler(
        dataset, seed=3407, batch_size=2, num_processes=2, event_boost=1.5
    )
    second = ResumableTaskEventBalancedSampler(
        dataset, seed=3407, batch_size=2, num_processes=2, event_boost=1.5
    )
    indices = list(first)
    assert indices == list(second)
    task_mass = {
        task: sum(
            float(first._weights[index])
            for index, (candidate_task, _) in enumerate(dataset.strata)
            if candidate_task == task
        )
        for task in ("long", "short")
    }
    assert task_mass["long"] == pytest.approx(task_mass["short"])
    counts = Counter(dataset.strata[index][0] for index in indices)
    assert abs(counts["long"] - counts["short"]) < len(dataset) * 0.25

    resumed = ResumableTaskEventBalancedSampler(
        dataset, seed=3407, batch_size=2, num_processes=2, event_boost=1.5
    )
    resumed.set_resume_batch_offset(3)
    assert list(resumed) == indices[12:]


def test_balanced_sampler_rejects_unbounded_event_boost() -> None:
    with pytest.raises(ValueError, match="event_boost"):
        ResumableTaskEventBalancedSampler(
            _StratifiedDataset(),
            seed=3407,
            batch_size=1,
            num_processes=1,
            event_boost=10.0,
        )
