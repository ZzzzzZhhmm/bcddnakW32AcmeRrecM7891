#!/usr/bin/env python3
"""Convert a validated WARM RMBench data profile to LeRobot v2.1."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fastwam.datasets.rmbench.converter import (
    RMBenchConversionConfig,
    convert_rmbench_dataset,
)
from fastwam.benchmarks.rmbench_sota import (
    RMBENCH_DATA_PROFILES,
    RMBENCH_SOTA_ROOT_SEED,
    data_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert all nine RMBench tasks using one closed WARM data profile "
            "into an atomic, hashed FastWAM LeRobot v2.1 tree."
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help=(
            "Official RMBench repository root or its data/ directory. No files "
            "under this path are modified."
        ),
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="New destination directory; it must not already exist.",
    )
    parser.add_argument(
        "--source-revision",
        required=True,
        help="Immutable official dataset revision, preferably a Hugging Face commit SHA.",
    )
    parser.add_argument(
        "--source-dataset",
        default="TianxingChen/RMBench",
        help=(
            "Immutable source dataset identifier. Official50 requires "
            "TianxingChen/RMBench; scaled auto-collected profiles must name "
            "their own private snapshot."
        ),
    )
    parser.add_argument(
        "--rmbench-code-revision",
        required=True,
        help="Immutable official RMBench Git commit used to define the source layout.",
    )
    parser.add_argument(
        "--data-revision",
        required=True,
        help="WARM-owned revision label for the converted dataset artifact.",
    )
    parser.add_argument(
        "--dataset-id",
        default="rmbench_demo_clean_v1",
        help="Stable local dataset ID embedded in the WARM episode catalog.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Official RoboTwin/RMBench control and camera rate.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(RMBENCH_DATA_PROFILES),
        default="official50-dev45",
        help=(
            "Closed data-volume/split profile. Scaled profiles expect an "
            "already collected, automatically generated source tree with the "
            "same nine-task/camera/action contract."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=RMBENCH_SOTA_ROOT_SEED,
        help="Deterministic task-stratified episode split seed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent per-episode converters; each encoder itself uses one thread.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = data_profile(args.profile)
    config = RMBenchConversionConfig(
        source_root=args.source_root,
        output_root=args.output_root,
        source_revision=args.source_revision,
        source_dataset=args.source_dataset,
        data_revision=args.data_revision,
        rmbench_code_revision=args.rmbench_code_revision,
        dataset_id=args.dataset_id,
        fps=args.fps,
        dev_per_task=profile.dev_per_task,
        split_seed=args.split_seed,
        workers=args.workers,
        episodes_per_task=profile.episodes_per_task,
        strict_official_contract=profile.strict_official_source,
        data_profile=profile.name,
        progress=True,
    )
    manifest = convert_rmbench_dataset(config)
    print(f"dataset={config.output_root.resolve()}")
    print(f"manifest_sha256={manifest['manifest_sha256']}")
    print(f"artifact_tree_sha256={manifest['output']['artifact_tree_sha256']}")
    print(f"source_tree_sha256={manifest['source']['source_tree_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
