"""Unified stage selection, with no robot/network execution side effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def main(argv=None, *, required_adapter=None):
    parser = argparse.ArgumentParser(description="Prepare WARM training data and factual memory artifacts")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("plan", "prepare", "features", "memory", "all"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if required_adapter is not None and cfg["adapter"] != required_adapter:
        parser.error(f"This entrypoint only accepts {required_adapter}")
    if args.stage == "plan":
        print(json.dumps({"adapter": cfg["adapter"], "source": cfg["source"], "output": cfg["output"],
                          "profile": cfg["profile"], "stages": ["prepare (CPU)", "features (GPU)", "memory (CPU)"],
                          "raw_assets_checked": False, "robot_motion": False}, ensure_ascii=False, indent=2))
        return 0
    if args.stage in {"prepare", "all"}:
        from .prepare import prepare
        print(prepare(cfg))
    if args.stage in {"features", "all"}:
        from .features import precompute
        print(precompute(cfg))
    if args.stage in {"memory", "all"}:
        from .memory import build_memory
        print(build_memory(cfg))
    return 0
