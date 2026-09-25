#!/usr/bin/env python3
"""Verify complete disjoint LIBERO shards and produce the measured Table 4."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from fastwam.research.evidence import canonical
from fastwam.research.table4 import summarize_table4, latex_rows
from assemble_nonreal_candidates import assemble, read_jsonl


def load_shards(shards):
    index, branches, labels, plans, coverage = [], [], [], [], []
    declared = set()
    for directory in shards:
        p = json.loads((directory/"plan.json").read_text())
        execution = json.loads((directory/"execution.json").read_text())
        if p.get("qualification_only") or execution.get("completed") is not True:
            raise ValueError("only completed formal shards can be merged")
        if p["tasks"] != list(range(10)) or p["frames"] != [80, 160]:
            raise ValueError("formal shards must cover LIBERO-10 and fixed frames80/160")
        if (not p["episodes"] or len(set(p["episodes"])) != len(p["episodes"])
                or any(type(e) is not int or e not in range(10) for e in p["episodes"])):
            raise ValueError("formal shard has invalid or duplicate initial states")
        if plans:
            fields = ("cohort", "seed", "horizon", "top_k", "replan_steps", "outcome", "policy",
                      "effect_head", "effect_projection", "candidate_metrics", "bootstrap", "metrics_seed")
            if any(p[k] != plans[0][k] for k in fields):
                raise ValueError("shards disagree on frozen protocol")
        cells = {(t, ep, f) for t in p["tasks"] for ep in p["episodes"] for f in p["frames"]}
        if declared & cells:
            raise ValueError("overlapping source-episode shards")
        declared |= cells
        ix = read_jsonl(directory/"index.jsonl")
        if len(ix) != execution["complete_queries"]:
            raise ValueError("query index and completion receipt disagree")
        actual = {(int(r["task"].split("/")[1]), int(r["episode_id"]),
                   int(r["query_id"].split("frame")[1])) for r in ix}
        skipped = set()
        local_coverage = read_jsonl(directory/"coverage.jsonl") if (directory/"coverage.jsonl").exists() else []
        for record in local_coverage:
            for frame in record["skipped_queries"]:
                cell = (record["task"], record["episode"], frame)
                if cell in skipped or cell in actual:
                    raise ValueError("duplicate or contradictory skipped query")
                skipped.add(cell)
        if actual | skipped != cells or actual & skipped:
            raise ValueError("missing or undeclared factual queries")
        plans.append(p)
        index.extend(ix)
        branches.extend(read_jsonl(directory/"branches.jsonl"))
        labels.extend(read_jsonl(directory/"labels.jsonl"))
        coverage.extend(local_coverage)
    return index, branches, labels, plans, coverage, declared


def merge(shards):
    index, branches, labels, plans, coverage, declared = load_shards(shards)
    expected = {(t, ep, f) for t in range(10) for ep in range(10) for f in (80,160)}
    if declared != expected:
        raise ValueError("formal protocol requires all 10 tasks x initial states0..9 x two frames")
    rows = assemble(index, branches, labels)
    plan = plans[0]
    config = dict(tasks=[f"libero_10/{t}" for t in range(10)], bootstrap=plan["bootstrap"],
                  seed=plan["metrics_seed"], candidate_metrics=plan["candidate_metrics"])
    report = summarize_table4(rows, config)
    report.update(protocol={**plan, "episodes": list(range(10))}, shard_plans=plans,
                  shard_paths=[str(p) for p in shards],
                  planned_queries=len(expected), factual_early_termination=coverage,
                  planned_source_episodes=100,
                  label_scope=plan["outcome_scope"], effect_head=plan["effect_head"])
    return rows, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, report = merge(args.shards)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"assembled.jsonl").write_bytes(b"".join(canonical(r)+b"\n" for r in rows))
    (args.output/"table4.json").write_bytes(canonical(report))
    if report["status"] == "measured":
        (args.output/"table4_rows.tex").write_text(latex_rows(report), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], blockers=report["blockers"], output=str(args.output))))
    return 0 if report["status"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
