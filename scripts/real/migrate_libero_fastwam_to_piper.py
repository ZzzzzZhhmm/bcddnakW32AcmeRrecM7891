#!/usr/bin/env python3
"""Project official LIBERO FastWAM weights onto the Piper 7D proprio head."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import torch

from fastwam.memory.manifest import sha256_file
from fastwam.real.libero_to_piper import migrate_fastwam_checkpoint_payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("checkpoints/fastwam_release/libero_uncond_2cam224.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"LIBERO FastWAM checkpoint not found: {source}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"migrated checkpoint already exists: {output}")
    if output == source:
        raise ValueError("--output must not overwrite the LIBERO source checkpoint")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    migrated, report = migrate_fastwam_checkpoint_payload(payload)
    report["source"] = str(source)
    report["source_sha256"] = sha256_file(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    try:
        torch.save(migrated, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    report["output"] = str(output)
    report["output_sha256"] = sha256_file(output)
    sidecar = output.with_suffix(output.suffix + ".migration.json")
    sidecar.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
