"""Small synthetic catalog/audit artifacts shared by WARM CLI tests."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

from fastwam.datasets.lerobot.audit import (
    AUDIT_SCHEMA,
    AUDIT_VERSION,
    write_audit_report,
)
from fastwam.datasets.lerobot.episode_catalog import (
    DatasetDescriptor,
    EpisodeCatalog,
    EpisodeRecord,
)
from fastwam.memory.bank_builder import EpisodeFeatures


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def write_test_catalog_and_audit(
    root: Path,
    episodes: Sequence[tuple[EpisodeFeatures, str]],
) -> tuple[Path, Path, EpisodeCatalog, dict[str, object]]:
    """Write a self-consistent immutable catalog and hashed audit fixture."""

    if not episodes:
        raise ValueError("episodes must not be empty")
    grouped: dict[tuple[str, int], list[tuple[EpisodeFeatures, str]]] = defaultdict(list)
    for episode, split in episodes:
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev, or test")
        grouped[(episode.dataset_id, episode.dataset_index)].append((episode, split))

    descriptors = []
    records = []
    for (dataset_id, dataset_index), members in sorted(
        grouped.items(), key=lambda item: item[0][1]
    ):
        descriptors.append(
            DatasetDescriptor(
                dataset_id=dataset_id,
                dataset_index=dataset_index,
                fps=20.0,
                total_episodes=len(members),
                chunks_size=1000,
                data_path_template=(
                    "data/chunk-{episode_chunk:03d}/"
                    "episode_{episode_index:06d}.parquet"
                ),
                info_sha256=_digest(f"info:{dataset_id}:{dataset_index}"),
                episodes_sha256=_digest(f"episodes:{dataset_id}:{dataset_index}"),
            )
        )
        for episode, split in members:
            records.append(
                EpisodeRecord(
                    dataset_id=dataset_id,
                    dataset_index=dataset_index,
                    episode_index=episode.episode_index,
                    length=int(episode.semantic_features.shape[0]),
                    fps=20.0,
                    tasks=(f"task-{episode.task_index}",),
                    data_relpath=(
                        f"data/chunk-{episode.episode_index // 1000:03d}/"
                        f"episode_{episode.episode_index:06d}.parquet"
                    ),
                    split=split,
                )
            )

    catalog = EpisodeCatalog(tuple(descriptors), tuple(records))
    root.mkdir(parents=True, exist_ok=True)
    catalog_path = root / "catalog.json"
    catalog.save(catalog_path)

    proof_rows = [
        {
            "dataset_id": episode.dataset_id,
            "dataset_index": episode.dataset_index,
            "episode_index": episode.episode_index,
            "split": split,
            "table_sha256": episode.source_episode_sha256,
            "ordered_camera_sha256": [],
            "camera_bundle_sha256": None,
            "source_episode_sha256": episode.source_episode_sha256,
        }
        for episode, split in sorted(
            episodes,
            key=lambda item: (
                item[0].dataset_index,
                item[0].episode_index,
            ),
        )
    ]
    hashes: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in proof_rows:
        hashes[str(row["source_episode_sha256"])].append(row)
    def episode_ref(row: dict[str, object]) -> dict[str, object]:
        return {
            "dataset_id": row["dataset_id"],
            "dataset_index": row["dataset_index"],
            "episode_index": row["episode_index"],
            "split": row["split"],
        }
    cross_split = [
        {"sha256": digest, "episodes": [episode_ref(row) for row in rows]}
        for digest, rows in sorted(hashes.items())
        if len(rows) > 1 and len({str(row["split"]) for row in rows}) > 1
    ]
    split_counts = Counter(split for _, split in episodes)
    report: dict[str, object] = {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "catalog_sha256": catalog.content_sha256,
        "audited_camera_keys": [],
        "datasets": [],
        "summary": {
            "dataset_count": len(descriptors),
            "episode_count": len(records),
            "task_count": len({record.primary_task for record in records}),
            "split_counts": dict(sorted(split_counts.items())),
            "episode_tables_hashed": True,
            # With no cameras, table and full-source identities are equal but
            # remain separately audited duplicate dimensions.
            "cross_split_duplicate_count": 2 * len(cross_split),
            "cross_split_table_duplicate_count": len(cross_split),
            "cross_split_video_duplicate_count": 0,
            "cross_split_camera_bundle_duplicate_count": 0,
            "cross_split_source_bundle_duplicate_count": len(cross_split),
        },
        "cross_split_table_duplicates": cross_split,
        "cross_split_video_duplicates": [],
        "cross_split_camera_bundle_duplicates": [],
        "cross_split_source_bundle_duplicates": cross_split,
        "episode_source_hashes": proof_rows,
    }
    report["report_sha256"] = _canonical_hash(report)
    audit_path = root / "audit.json"
    write_audit_report(report, audit_path)
    return catalog_path, audit_path, catalog, report


__all__ = ["write_test_catalog_and_audit"]
