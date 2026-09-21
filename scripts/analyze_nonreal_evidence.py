#!/usr/bin/env python3
"""Analyze frozen episode outcomes or candidate branches; no GPU is needed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastwam.research.statistics import candidate_metrics, clustered_macro, paired_success, gate_clustered, donor_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("episodes", "candidates", "gates", "donors"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, config_raw = args.input.read_bytes(), args.config.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty evidence cannot produce results")
    config = json.loads(config_raw)
    common = {k: config[k] for k in ("tasks", "bootstrap", "seed")}
    if args.kind == "episodes":
        result = paired_success(rows, config["contrasts"], **common)
    elif args.kind == "gates":
        if len({r["cohort"] for r in rows}) != 1:
            raise ValueError("natural and donor cohorts must be analyzed separately")
        result = gate_clustered(rows, **common)
    elif args.kind == "donors":
        result = donor_coverage(rows)
    else:
        if len({r["cohort"] for r in rows}) != 1:
            raise ValueError("natural and donor cohorts must be analyzed separately")
        per_query = []
        for row in rows:
            metrics = candidate_metrics(row, **config["candidate_metrics"])
            out = {key: row[key] for key in ("task", "episode_id", "query_id")}
            out.update({k: v for k, v in metrics.items() if k != "references"})
            for reference, values in metrics["references"].items():
                for metric in ("effect_mse", "pairwise_agreement", "predicted_tie_rate", "top_choice_applicability"):
                    out[f"{reference}/{metric}"] = values[metric]
            per_query.append(out)
        metric_names = [k for k in per_query[0] if "/" in k]
        result = {"per_query": per_query,
                  "aggregates": {k: clustered_macro(per_query, k, **common) for k in metric_names}}
    result.update({"schema": "warm.nonreal.metrics.v1", "input_sha256": hashlib.sha256(raw).hexdigest(),
                   "config_sha256": hashlib.sha256(config_raw).hexdigest(), "config": config,
                   "input_records": len(rows)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({"output": str(args.output), "records": len(rows)}))


if __name__ == "__main__":
    main()
