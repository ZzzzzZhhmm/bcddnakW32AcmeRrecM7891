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


class _HierarchicalDataset:
    def __init__(self) -> None:
        # Task A contains one very long and one short demonstration; task B
        # contains one demonstration.  Each episode has two progress bins,
        # with deliberately different numbers of ordinary/critical frames.
        self.strata = (
            *(("a", "a-long", 0, False),) * 80,
            *(("a", "a-long", 0, True),) * 20,
            *(("a", "a-long", 1, False),) * 50,
            *(("a", "a-short", 0, False),) * 5,
            (("a", "a-short", 1, True),),
            *(("b", "b-only", 0, False),) * 9,
            (("b", "b-only", 1, True),),
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


def test_hierarchical_sampler_equalizes_task_episode_and_progress_mass() -> None:
    dataset = _HierarchicalDataset()
    sampler = ResumableTaskEventBalancedSampler(
        dataset,
        seed=3407,
        batch_size=2,
        num_processes=2,
        event_boost=2.0,
    )
    assert sampler.sampling_schema == "task_episode_progress_event_v2"

    def mass(predicate) -> float:
        return sum(
            float(sampler._weights[index])
            for index, stratum in enumerate(dataset.strata)
            if predicate(stratum)
        )

    assert mass(lambda item: item[0] == "a") == pytest.approx(
        mass(lambda item: item[0] == "b")
    )
    assert mass(lambda item: item[1] == "a-long") == pytest.approx(
        mass(lambda item: item[1] == "a-short")
    )
    for episode in ("a-long", "a-short", "b-only"):
        assert mass(lambda item, episode=episode: item[1] == episode and item[2] == 0) == pytest.approx(
            mass(lambda item, episode=episode: item[1] == episode and item[2] == 1)
        )


def test_sampler_rejects_mixed_or_invalid_hierarchical_schema() -> None:
    class _Bad:
        strata = (("a", "episode", 0, False), ("a", True))

        def __len__(self):
            return 2

        def sampling_strata(self):
            return self.strata

    with pytest.raises(TypeError, match="consistent schema"):
        ResumableTaskEventBalancedSampler(
            _Bad(), seed=3407, batch_size=1, num_processes=1
        )
