#!/usr/bin/env python3
"""Validate and query the closed RMBench score-optimization registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.benchmarks.rmbench_sota import (
    data_profile,
    load_registry_document,
    load_task_profiles,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/rmbench/sota_v1.json"),
    )
    parser.add_argument("--task", default=None)
    parser.add_argument(
        "--data-profile",
        default=None,
        help="Explicit artifact profile override (validated against the manifest).",
    )
    parser.add_argument(
        "--format", choices=("json", "tsv"), default="json"
    )
    parser.add_argument("--validate-manifest", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = load_registry_document(args.registry)
    profiles = load_task_profiles(args.registry)
    profile = data_profile(
        document["shared"]["data_profile"]
        if args.data_profile is None
        else args.data_profile
    )
    if args.validate_manifest is not None:
        manifest = json.loads(
            args.validate_manifest.expanduser().resolve().read_text(encoding="utf-8")
        )
        actual = manifest.get("protocol", {}).get("data_profile")
        if actual != profile.name:
            raise ValueError(
                f"artifact data profile mismatch: registry={profile.name!r} "
                f"manifest={actual!r}"
            )
    if args.task is None:
        payload = {
            **document["shared"],
            "root_seed": document["root_seed"],
            "data_profile": profile.name,
            "episodes_per_task": profile.episodes_per_task,
            "train_per_task": profile.train_per_task,
            "dev_per_task": profile.dev_per_task,
        }
        if args.format == "tsv":
            print(
                "\t".join(
                    str(payload[key])
                    for key in (
                        "root_seed",
                        "data_profile",
                        "train_steps",
                        "sampler_mode",
                        "event_boost",
                        "recent_event_capacity",
                        "action_summary_capacity",
                        "replan_steps",
                    )
                )
            )
        else:
            print(json.dumps(payload, sort_keys=True))
        return 0
    try:
        task = profiles[args.task]
    except KeyError as exc:
        raise ValueError(f"unknown RMBench task {args.task!r}") from exc
    payload = {
        "task_name": task.task_name,
        "memory_regime": task.memory_regime,
        "train_steps": task.train_steps,
        "recent_event_capacity": task.recent_event_capacity,
        "action_summary_capacity": task.action_summary_capacity,
        "replan_steps": task.replan_steps,
        "inference_steps": task.inference_steps,
        "top_k": task.top_k,
    }
    if args.format == "tsv":
        print(
            "\t".join(
                str(payload[key])
                for key in (
                    "task_name",
                    "memory_regime",
                    "train_steps",
                    "recent_event_capacity",
                    "action_summary_capacity",
                    "replan_steps",
                    "inference_steps",
                    "top_k",
                )
            )
        )
    else:
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
