from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastwam.datasets.lerobot.episode_catalog import (
    EpisodeCatalog,
    assign_task_stratified_dev_split,
    episode_indices_by_dataset,
    scan_lerobot_datasets,
)


def _write_dataset(root: Path, *, task: str, episode_count: int = 3) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 20,
                "total_episodes": episode_count,
                "chunks_size": 2,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for episode_index in range(episode_count):
        rows.append(
            {
                "episode_index": episode_index,
                "length": 40 + episode_index,
                "tasks": [task],
            }
        )
        chunk = episode_index // 2
        path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    (root / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_scan_uses_dataset_index_to_disambiguate_episode_numbers(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_dataset(first, task="pick")
    _write_dataset(second, task="place")

    catalog = scan_lerobot_datasets([first, second])

    assert len(catalog.episodes) == 6
    assert catalog.episodes[0].global_episode_id == (0, 0)
    assert catalog.episodes[3].global_episode_id == (1, 0)
    assert catalog.episodes[2].data_relpath == "data/chunk-001/episode_000002.parquet"


def test_catalog_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(root, task="pick")
    catalog = scan_lerobot_datasets([root])
    path = tmp_path / "catalog.json"
    catalog.save(path)

    assert EpisodeCatalog.load(path) == catalog

    document = json.loads(path.read_text(encoding="utf-8"))
    document["episodes"][0]["length"] += 1
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        EpisodeCatalog.load(path)


def test_task_stratified_split_is_deterministic_and_retains_train(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(root, task="pick", episode_count=6)
    catalog = scan_lerobot_datasets([root])

    split_a = assign_task_stratified_dev_split(catalog, dev_per_task=2, seed=42)
    split_b = assign_task_stratified_dev_split(catalog, dev_per_task=2, seed=42)

    assert split_a == split_b
    assert sum(item.split == "dev" for item in split_a.episodes) == 2
    assert sum(item.split == "train" for item in split_a.episodes) == 4


def test_scan_rejects_missing_episode_payload(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _write_dataset(root, task="pick")
    missing = root / "data" / "chunk-000" / "episode_000001.parquet"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="Missing episode table"):
        scan_lerobot_datasets([root])


def test_resolve_exact_split_indices_in_configured_dataset_order(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_dataset(first, task="pick", episode_count=4)
    _write_dataset(second, task="place", episode_count=4)
    catalog = assign_task_stratified_dev_split(
        scan_lerobot_datasets([first, second]),
        dev_per_task=1,
        seed=7,
    )

    train = episode_indices_by_dataset(
        catalog,
        split="train",
        configured_episode_totals=[4, 4],
    )
    dev = episode_indices_by_dataset(catalog, split="dev")

    assert all(len(indices) == 3 for indices in train)
    assert all(len(indices) == 1 for indices in dev)
    assert all(set(train[i]).isdisjoint(dev[i]) for i in range(2))

    with pytest.raises(ValueError, match="do not match catalog"):
        episode_indices_by_dataset(
            catalog,
            split="train",
            configured_episode_totals=[4, 3],
        )
