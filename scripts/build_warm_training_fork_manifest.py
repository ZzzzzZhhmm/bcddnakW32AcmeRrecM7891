#!/usr/bin/env python3
"""Build one immutable shared-WARM -> specialist fork manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from fastwam.models.warm.training_fork import (
    build_training_fork_manifest,
    publish_training_fork_manifest,
)


def _config(path: str) -> dict:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"resolved config must be a mapping: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--parent-config", required=True)
    parser.add_argument("--child-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fork-reason", required=True)
    args = parser.parse_args()
    manifest = build_training_fork_manifest(
        parent_checkpoint=args.parent_checkpoint,
        parent_config=_config(args.parent_config),
        child_config=_config(args.child_config),
        fork_reason=args.fork_reason,
    )
    output = publish_training_fork_manifest(args.output, manifest)
    print(f"training_fork_manifest={Path(output).resolve()}")
    print(f"parent_checkpoint_sha256={manifest['parent_checkpoint_sha256']}")
    print(f"parent_commit={manifest['parent_git_commit']}")
    print(f"parent_step={manifest['parent_checkpoint_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
