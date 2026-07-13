#!/usr/bin/env python3
"""Build a hashed, episode-level WARM preprocessing catalog."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

from fastwam.datasets.lerobot.episode_catalog import (
    assign_task_stratified_dev_split,
    scan_lerobot_datasets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan complete LeRobot episodes before window sampling and write an "
            "immutable, content-hashed WARM catalog."
        )
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        required=True,
        type=Path,
        help="LeRobot v2 dataset root; repeat in the configured dataset order.",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=None,
        help="Stable dataset identifier; repeat once per --dataset-root.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dev-per-task",
        type=int,
        default=5,
        help="Deterministically reserve this many episodes per task for development.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing-episode-data",
        action="store_true",
        help="Metadata-only audit mode; production catalogs should not use this.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = scan_lerobot_datasets(
        args.dataset_root,
        dataset_ids=args.dataset_id,
        require_episode_data=not args.allow_missing_episode_data,
    )
    catalog = assign_task_stratified_dev_split(
        catalog,
        dev_per_task=args.dev_per_task,
        seed=args.seed,
    )
    catalog.save(args.output)

    split_counts = Counter(episode.split for episode in catalog.episodes)
    task_count = len({(episode.dataset_id, episode.primary_task) for episode in catalog.episodes})
    print(f"catalog={args.output}")
    print(f"sha256={catalog.content_sha256}")
    print(f"datasets={len(catalog.datasets)} tasks={task_count} episodes={len(catalog.episodes)}")
    print("splits=" + ",".join(f"{key}:{split_counts[key]}" for key in sorted(split_counts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
