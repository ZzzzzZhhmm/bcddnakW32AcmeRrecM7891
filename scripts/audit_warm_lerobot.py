#!/usr/bin/env python3
"""Audit a WARM episode catalog against its LeRobot datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from fastwam.datasets.lerobot.audit import (
    DEFAULT_AUDITED_CAMERA_KEYS,
    audit_lerobot_catalog,
    write_audit_report,
)
from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    hashing = parser.add_mutually_exclusive_group()
    hashing.add_argument(
        "--hash-episode-tables",
        dest="hash_episode_tables",
        action="store_true",
        help="Hash every parquet and audited camera MP4 (the default).",
    )
    hashing.add_argument(
        "--metadata-only",
        dest="hash_episode_tables",
        action="store_false",
        help="Explicit non-production metadata-only audit without source proofs.",
    )
    parser.set_defaults(hash_episode_tables=True)
    parser.add_argument(
        "--camera-key",
        action="append",
        help=(
            "Ordered LeRobot camera feature to bind. Repeat for every camera. "
            "Defaults to external image then wrist_image."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.hash_episode_tables and args.camera_key:
        raise ValueError("--camera-key cannot be used with --metadata-only")
    cameras = tuple(args.camera_key or DEFAULT_AUDITED_CAMERA_KEYS)
    if not args.hash_episode_tables:
        cameras = ()
    catalog = EpisodeCatalog.load(args.catalog)
    report = audit_lerobot_catalog(
        catalog,
        args.dataset_root,
        hash_episode_tables=args.hash_episode_tables,
        camera_keys=cameras,
    )
    write_audit_report(report, args.output)
    summary = report["summary"]
    print(f"report={args.output}")
    print(f"sha256={report['report_sha256']}")
    print(
        f"datasets={summary['dataset_count']} tasks={summary['task_count']} "
        f"episodes={summary['episode_count']} duplicates={summary['cross_split_duplicate_count']} "
        f"table={summary['cross_split_table_duplicate_count']} "
        f"video={summary['cross_split_video_duplicate_count']} "
        f"camera_bundle={summary['cross_split_camera_bundle_duplicate_count']} "
        f"source_bundle={summary['cross_split_source_bundle_duplicate_count']}"
    )
    return 1 if summary["cross_split_duplicate_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
