"""Immutable, episode-level catalogs for WARM preprocessing.

Event mining operates on complete episodes, before any overlapping training
windows are sampled.  This module intentionally has no Torch or LeRobot
runtime dependency so catalog construction and split audits can run on CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CATALOG_SCHEMA_VERSION = 1
DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


@dataclass(frozen=True)
class DatasetDescriptor:
    dataset_id: str
    dataset_index: int
    fps: float
    total_episodes: int
    chunks_size: int
    data_path_template: str
    info_sha256: str
    episodes_sha256: str

    def __post_init__(self) -> None:
        if not self.dataset_id or self.dataset_id.strip() != self.dataset_id:
            raise ValueError("dataset_id must be a non-empty normalized string")
        if self.dataset_index < 0:
            raise ValueError("dataset_index must be non-negative")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.total_episodes < 0:
            raise ValueError("total_episodes must be non-negative")
        if self.chunks_size <= 0:
            raise ValueError("chunks_size must be positive")
        if len(self.info_sha256) != 64 or len(self.episodes_sha256) != 64:
            raise ValueError("metadata hashes must be SHA-256 hex digests")


@dataclass(frozen=True)
class EpisodeRecord:
    dataset_id: str
    dataset_index: int
    episode_index: int
    length: int
    fps: float
    tasks: tuple[str, ...]
    data_relpath: str
    split: str = "unassigned"

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if self.dataset_index < 0 or self.episode_index < 0:
            raise ValueError("dataset_index and episode_index must be non-negative")
        if self.length <= 0:
            raise ValueError("episode length must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.split not in {"unassigned", "train", "dev", "test"}:
            raise ValueError(f"Unsupported split: {self.split}")
        relpath = Path(self.data_relpath)
        if relpath.is_absolute() or ".." in relpath.parts:
            raise ValueError("data_relpath must stay within the dataset root")

    @property
    def global_episode_id(self) -> tuple[int, int]:
        return self.dataset_index, self.episode_index

    @property
    def primary_task(self) -> str:
        return self.tasks[0] if self.tasks else f"dataset:{self.dataset_id}"


@dataclass(frozen=True)
class EpisodeCatalog:
    datasets: tuple[DatasetDescriptor, ...]
    episodes: tuple[EpisodeRecord, ...]
    schema_version: int = CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported catalog schema {self.schema_version}; "
                f"expected {CATALOG_SCHEMA_VERSION}"
            )
        dataset_indices = [item.dataset_index for item in self.datasets]
        dataset_ids = [item.dataset_id for item in self.datasets]
        if len(set(dataset_indices)) != len(dataset_indices):
            raise ValueError("dataset_index values must be unique")
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("dataset_id values must be unique")
        descriptors = {item.dataset_index: item for item in self.datasets}
        seen: set[tuple[int, int]] = set()
        counts = {item.dataset_index: 0 for item in self.datasets}
        for episode in self.episodes:
            descriptor = descriptors.get(episode.dataset_index)
            if descriptor is None or descriptor.dataset_id != episode.dataset_id:
                raise ValueError("episode references an unknown or mismatched dataset")
            if episode.fps != descriptor.fps:
                raise ValueError("episode fps disagrees with its dataset descriptor")
            if episode.global_episode_id in seen:
                raise ValueError(f"Duplicate global episode id: {episode.global_episode_id}")
            seen.add(episode.global_episode_id)
            counts[episode.dataset_index] += 1
        for descriptor in self.datasets:
            if counts[descriptor.dataset_index] != descriptor.total_episodes:
                raise ValueError(
                    f"Dataset {descriptor.dataset_id} declares {descriptor.total_episodes} "
                    f"episodes but catalog contains {counts[descriptor.dataset_index]}"
                )

    def _payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "datasets": [asdict(item) for item in self.datasets],
            "episodes": [
                {**asdict(item), "tasks": list(item.tasks)} for item in self.episodes
            ],
        }

    @property
    def content_sha256(self) -> str:
        return sha256(_canonical_json(self._payload())).hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = self._payload()
        document["content_sha256"] = self.content_sha256
        target.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "EpisodeCatalog":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        expected_hash = document.pop("content_sha256", None)
        if not isinstance(expected_hash, str):
            raise ValueError("Catalog is missing content_sha256")
        actual_hash = sha256(_canonical_json(document)).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("Catalog content hash mismatch")
        catalog = cls(
            schema_version=int(document["schema_version"]),
            datasets=tuple(DatasetDescriptor(**item) for item in document["datasets"]),
            episodes=tuple(
                EpisodeRecord(**{**item, "tasks": tuple(item.get("tasks", ()))})
                for item in document["episodes"]
            ),
        )
        if catalog.content_sha256 != expected_hash:
            raise ValueError("Catalog did not round-trip canonically")
        return catalog


def scan_lerobot_datasets(
    dataset_roots: Sequence[str | Path],
    *,
    dataset_ids: Sequence[str] | None = None,
    require_episode_data: bool = True,
) -> EpisodeCatalog:
    """Scan LeRobot v2.x metadata without decoding frames or training windows."""

    if not dataset_roots:
        raise ValueError("At least one dataset root is required")
    if dataset_ids is not None and len(dataset_ids) != len(dataset_roots):
        raise ValueError("dataset_ids must match dataset_roots length")

    descriptors: list[DatasetDescriptor] = []
    episodes: list[EpisodeRecord] = []
    resolved_ids = list(dataset_ids) if dataset_ids is not None else [
        Path(root).resolve().name for root in dataset_roots
    ]
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError("dataset_ids must be unique")

    for dataset_index, (root_value, dataset_id) in enumerate(
        zip(dataset_roots, resolved_ids, strict=True)
    ):
        root = Path(root_value).resolve()
        info_path = root / "meta" / "info.json"
        episodes_path = root / "meta" / "episodes.jsonl"
        if not info_path.is_file() or not episodes_path.is_file():
            raise FileNotFoundError(
                f"Expected meta/info.json and meta/episodes.jsonl under {root}"
            )
        info = json.loads(info_path.read_text(encoding="utf-8"))
        rows = _read_jsonl(episodes_path)
        fps = float(info["fps"])
        chunks_size = int(info.get("chunks_size", 1000))
        declared_total = int(info.get("total_episodes", len(rows)))
        if declared_total != len(rows):
            raise ValueError(
                f"{dataset_id}: info declares {declared_total} episodes, "
                f"episodes.jsonl contains {len(rows)}"
            )
        data_template = str(info.get("data_path", DEFAULT_DATA_PATH))
        descriptor = DatasetDescriptor(
            dataset_id=dataset_id,
            dataset_index=dataset_index,
            fps=fps,
            total_episodes=declared_total,
            chunks_size=chunks_size,
            data_path_template=data_template,
            info_sha256=_file_sha256(info_path),
            episodes_sha256=_file_sha256(episodes_path),
        )
        descriptors.append(descriptor)

        for row in rows:
            episode_index = int(row["episode_index"])
            episode_chunk = episode_index // chunks_size
            try:
                relpath = data_template.format(
                    episode_chunk=episode_chunk,
                    episode_index=episode_index,
                )
            except KeyError as exc:
                raise ValueError(f"Unsupported data path template: {data_template}") from exc
            data_path = root / relpath
            if require_episode_data and not data_path.is_file():
                raise FileNotFoundError(f"Missing episode table: {data_path}")
            raw_tasks = row.get("tasks", ())
            if isinstance(raw_tasks, str):
                raw_tasks = (raw_tasks,)
            episodes.append(
                EpisodeRecord(
                    dataset_id=dataset_id,
                    dataset_index=dataset_index,
                    episode_index=episode_index,
                    length=int(row["length"]),
                    fps=fps,
                    tasks=tuple(str(task) for task in raw_tasks),
                    data_relpath=Path(relpath).as_posix(),
                )
            )

    return EpisodeCatalog(tuple(descriptors), tuple(episodes))


def assign_task_stratified_dev_split(
    catalog: EpisodeCatalog,
    *,
    dev_per_task: int,
    seed: int,
) -> EpisodeCatalog:
    """Assign a deterministic episode-level split while retaining train data."""

    if dev_per_task < 0:
        raise ValueError("dev_per_task must be non-negative")
    groups: dict[tuple[str, str], list[EpisodeRecord]] = {}
    for episode in catalog.episodes:
        groups.setdefault((episode.dataset_id, episode.primary_task), []).append(episode)

    dev_ids: set[tuple[int, int]] = set()
    for group_key, members in groups.items():
        ranked = sorted(
            members,
            key=lambda item: sha256(
                f"{seed}:{group_key[0]}:{group_key[1]}:{item.episode_index}".encode("utf-8")
            ).digest(),
        )
        count = min(dev_per_task, max(0, len(ranked) - 1))
        dev_ids.update(item.global_episode_id for item in ranked[:count])

    episodes = tuple(
        replace(
            episode,
            split="dev" if episode.global_episode_id in dev_ids else "train",
        )
        for episode in catalog.episodes
    )
    return EpisodeCatalog(catalog.datasets, episodes, catalog.schema_version)


def episode_indices_by_dataset(
    catalog: EpisodeCatalog,
    *,
    split: str,
    configured_episode_totals: Sequence[int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Resolve exact per-dataset episode indices for a configured split.

    Dataset order is part of the catalog contract and must match the order in
    the Hydra ``dataset_dirs`` list.  This avoids using path-dependent IDs in
    the immutable catalog while still detecting a mismatched dataset set.
    """

    if split not in {"train", "dev", "test"}:
        raise ValueError(f"split must be train/dev/test, got {split!r}")
    descriptors = sorted(catalog.datasets, key=lambda item: item.dataset_index)
    if [item.dataset_index for item in descriptors] != list(range(len(descriptors))):
        raise ValueError("catalog dataset indices must be contiguous from zero")
    if configured_episode_totals is not None:
        totals = tuple(int(value) for value in configured_episode_totals)
        expected = tuple(item.total_episodes for item in descriptors)
        if totals != expected:
            raise ValueError(
                f"configured dataset episode totals {totals} do not match catalog {expected}"
            )

    output: list[tuple[int, ...]] = []
    for descriptor in descriptors:
        indices = tuple(
            sorted(
                episode.episode_index
                for episode in catalog.episodes
                if episode.dataset_index == descriptor.dataset_index
                and episode.split == split
            )
        )
        if not indices:
            raise ValueError(
                f"catalog has no {split!r} episodes for dataset {descriptor.dataset_id}"
            )
        output.append(indices)
    return tuple(output)
