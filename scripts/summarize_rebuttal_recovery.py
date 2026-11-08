#!/usr/bin/env python3
"""Summarize already-collected, independently labelled recovery branches."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastwam.research.recovery import summarize_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    result = summarize_recovery(rows, tasks=args.tasks, bootstrap=args.bootstrap, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "scope": result["scope"]}))


if __name__ == "__main__":
    main()
