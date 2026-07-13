"""Deterministic, task-stratified RMBench episode splitting."""

from __future__ import annotations

from hashlib import sha256
from typing import Iterable, Mapping


def deterministic_task_split(
    task_episode_indices: Mapping[str, Iterable[int]],
    *,
    dev_per_task: int,
    seed: int,
) -> dict[tuple[str, int], str]:
    """Return a stable episode-level train/dev assignment.

    Ranking is performed independently for every official task.  No windows or
    frames are split, so every source episode belongs to exactly one side.
    """

    if isinstance(dev_per_task, bool) or not isinstance(dev_per_task, int):
        raise TypeError("dev_per_task must be an integer")
    if dev_per_task < 0:
        raise ValueError("dev_per_task must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    assignments: dict[tuple[str, int], str] = {}
    for task, raw_indices in task_episode_indices.items():
        if not isinstance(task, str) or not task or task.strip() != task:
            raise ValueError("task names must be non-empty normalized strings")
        indices = tuple(int(value) for value in raw_indices)
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate source episode index for task {task!r}")
        if any(value < 0 for value in indices):
            raise ValueError("source episode indices must be non-negative")
        if not indices:
            raise ValueError(f"task {task!r} has no episodes")
        if dev_per_task >= len(indices):
            raise ValueError(
                f"dev_per_task must leave at least one train episode for {task!r}"
            )
        ranked = sorted(
            indices,
            key=lambda value: (
                sha256(f"{seed}:{task}:{value}".encode("utf-8")).digest(),
                value,
            ),
        )
        dev = set(ranked[:dev_per_task])
        for value in sorted(indices):
            assignments[(task, value)] = "dev" if value in dev else "train"
    return assignments


__all__ = ["deterministic_task_split"]
