#!/usr/bin/env python3
"""Read-only artifact discovery. Never invokes Git, downloads, or loads weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def inventory(roots: list[Path]) -> dict:
    records, errors = [], []
    for root in roots:
        if not root.is_dir():
            errors.append({"path": str(root), "reason": "missing_directory"})
            continue
        for directory, dirs, names in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d not in {
                ".git", "__pycache__", "code", "videos", "media", "state",
                "wandb", "features", "candidates", "third_party", "assets",
            })
            for name in sorted(names):
                p = Path(directory) / name
                kind = None
                if name.endswith(".training.json"):
                    kind = "training_attestation"
                elif p.suffix in {".pt", ".safetensors"}:
                    kind = "text_embedding" if "text" in p.parts else "checkpoint"
                elif name in {"config.yaml", "dataset_stats.json", "manifest.json"}:
                    kind = "configuration"
                elif p.suffix in {".json", ".jsonl", ".log"} and any(
                    s in name.lower() for s in ("result", "summary", "evidence", "episode", "stdout", "console")
                ):
                    kind = "possible_evaluation_evidence"
                if kind is None:
                    continue
                try:
                    st = p.stat()
                    row = {"path": str(p), "kind": kind, "bytes": st.st_size,
                           "mtime_ns": st.st_mtime_ns}
                    if p.suffix == ".json" and st.st_size < 2_000_000:
                        raw = p.read_bytes()
                        row["sha256"] = hashlib.sha256(raw).hexdigest()
                        data = json.loads(raw)
                        if isinstance(data, dict):
                            row["metadata"] = {k: data[k] for k in (
                                "schema", "version", "global_step", "step", "checkpoint_sha256",
                                "training_commit", "task", "successes", "episodes", "success_rate",
                                "status", "warm_retrospection", "training", "checkpoint",
                            ) if k in data}
                    # Weight filenames/size are discovery, not integrity attestation.
                    records.append(row)
                except (OSError, ValueError) as e:
                    errors.append({"path": str(p), "reason": str(e)})
    return {"schema": "warm.nonreal.inventory.v1", "roots": [str(p) for p in roots],
            "records": records, "errors": errors,
            "paper_result_linkage": None,
            "missing_reason": "discovery does not establish paper run identity"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "records": len(result["records"]),
                      "errors": len(result["errors"])}))
