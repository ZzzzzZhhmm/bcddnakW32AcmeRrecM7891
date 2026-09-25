#!/usr/bin/env python3
"""Report Table 4 from assemble_nonreal_candidates.py output (CPU only).

No simulator collection is performed. Missing evidence produces a blocked JSON
report and exit code 2, never draft numbers or a LaTeX table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastwam.research.table4 import latex_rows, summarize_table4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, config_raw = args.input.read_bytes(), args.config.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    config = json.loads(config_raw)
    report = summarize_table4(rows, config)
    report.update(input_sha256=hashlib.sha256(raw).hexdigest(),
                  config_sha256=hashlib.sha256(config_raw).hexdigest())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "table4.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if report["status"] == "measured":
        (args.output / "table4_rows.tex").write_text(latex_rows(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "blockers": report["blockers"]}))
    return 0 if report["status"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
