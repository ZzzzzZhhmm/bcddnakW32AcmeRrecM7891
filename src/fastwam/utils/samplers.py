from collections import Counter, defaultdict
import math
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class ResumableTaskEventBalancedSampler(ResumableEpochSampler):
    """Deterministic hierarchical balancing with a bounded event boost.

    Sampling with replacement keeps the epoch length unchanged.  Every rank
    constructs the same global index stream and Accelerate performs the usual
    distributed sharding, so full-state resume retains exact batch semantics.

    New RMBench datasets expose ``(task, episode, progress_bin, critical)``.
    This prevents long demonstrations and long approach phases from dominating
    a specialist.  Legacy ``(task, recent_event)`` metadata remains accepted
    for non-RMBench checkpoints and unit fixtures.
    """

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        *,
        event_boost: float = 1.5,
    ) -> None:
        super().__init__(dataset, seed, batch_size, num_processes)
        if not math.isfinite(float(event_boost)) or not 1.0 <= float(event_boost) <= 4.0:
            raise ValueError("event_boost must be finite and lie in [1, 4]")
        metadata = getattr(dataset, "sampling_strata", None)
        if not callable(metadata):
            raise TypeError(
                "task/event-balanced sampling requires dataset.sampling_strata()"
            )
        strata = tuple(metadata())
        if len(strata) != len(dataset) or not strata:
            raise ValueError("sampling strata must align exactly with the dataset")
        tasks: list[str] = []
        recent: list[bool] = []
        hierarchical: list[tuple[str, str, int, bool]] = []
        schema_width: int | None = None
        for index, item in enumerate(strata):
            if not isinstance(item, tuple) or len(item) not in {2, 4}:
                raise TypeError(f"invalid sampling stratum at index {index}: {item!r}")
            if schema_width is None:
                schema_width = len(item)
            elif len(item) != schema_width:
                raise TypeError("sampling strata must use one consistent schema")
            task = item[0]
            if not isinstance(task, str) or not task:
                raise TypeError(f"invalid sampling task at index {index}: {task!r}")
            if len(item) == 2:
                if not isinstance(item[1], bool):
                    raise TypeError(
                        f"invalid recent-event flag at index {index}: {item[1]!r}"
                    )
                tasks.append(task)
                recent.append(item[1])
                continue
            episode, progress_bin, critical = item[1:]
            if not isinstance(episode, str) or not episode:
                raise TypeError(
                    f"invalid episode key at index {index}: {episode!r}"
                )
            if (
                isinstance(progress_bin, bool)
                or not isinstance(progress_bin, int)
                or progress_bin < 0
            ):
                raise TypeError(
                    f"invalid progress bin at index {index}: {progress_bin!r}"
                )
            if not isinstance(critical, bool):
                raise TypeError(
                    f"invalid critical-event flag at index {index}: {critical!r}"
                )
            hierarchical.append((task, episode, progress_bin, critical))

        if schema_width == 2:
            task_counts = Counter(tasks)
            recent_counts = Counter(
                task
                for task, is_recent in zip(tasks, recent, strict=True)
                if is_recent
            )
            task_mass = {
                task: (task_counts[task] - recent_counts[task])
                + float(event_boost) * recent_counts[task]
                for task in task_counts
            }
            weights = [
                (float(event_boost) if is_recent else 1.0) / task_mass[task]
                for task, is_recent in zip(tasks, recent, strict=True)
            ]
            self.sampling_schema = "task_event_v1"
        else:
            episodes_by_task: dict[str, set[str]] = defaultdict(set)
            bins_by_episode: dict[tuple[str, str], set[int]] = defaultdict(set)
            phase_mass: Counter[tuple[str, str, int]] = Counter()
            for task, episode, progress_bin, critical in hierarchical:
                episodes_by_task[task].add(episode)
                bins_by_episode[(task, episode)].add(progress_bin)
                phase_mass[(task, episode, progress_bin)] += (
                    float(event_boost) if critical else 1.0
                )
            weights = []
            for task, episode, progress_bin, critical in hierarchical:
                factor = float(event_boost) if critical else 1.0
                weights.append(
                    factor
                    / float(len(episodes_by_task[task]))
                    / float(len(bins_by_episode[(task, episode)]))
                    / float(phase_mass[(task, episode, progress_bin)])
                )
            self.sampling_schema = "task_episode_progress_event_v2"
        self.event_boost = float(event_boost)
        self._weights = torch.tensor(weights, dtype=torch.float64)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.multinomial(
            self._weights,
            num_samples=len(self.dataset),
            replacement=True,
            generator=generator,
        ).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = (
                self.resume_batch_offset * self.batch_size * self.num_processes
            )
            indices = indices[sample_offset:]
        return iter(indices)
