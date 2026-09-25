"""Matched-support Table 4 reporting from executed, assembled branch evidence.

This module does not collect simulator outcomes or qualify a restoration backend.
It cannot turn recorded demonstration futures into candidate execution evidence.
"""
from __future__ import annotations

from collections import Counter
import re

from .statistics import candidate_metrics, clustered_macro


REFERENCES = ("historical", "query", "predicted")
METRICS = ("effect_mse", "pairwise_agreement", "predicted_tie_rate", "top_choice_applicability")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def summarize_table4(rows, config):
    """Use identical queries for the three references for each table column.

    Query-only top choice ties over the complete valid candidate pool. Requiring
    its label support for *all* references prevents differential missing-label
    selection from changing which episodes contribute to the comparison.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("Table 4 requires actual assembled branch evidence")
    checkpoints, cohorts, queries, seen = set(), set(), [], set()
    coverage = Counter()
    task_coverage = {task: Counter() for task in config["tasks"]}
    endpoint_statuses = Counter()
    for row in rows:
        for name in ("checkpoint_sha256", "probe_metadata_sha256"):
            if not isinstance(row.get(name), str) or SHA256.fullmatch(row[name]) is None:
                raise ValueError(f"missing or invalid {name}")
        if not row.get("cohort") or not row.get("episode_id") or not row.get("query_id"):
            raise ValueError("cohort, episode_id and query_id are required")
        if row["task"] not in task_coverage:
            raise ValueError("unexpected task")
        if row["query_id"] in seen:
            raise ValueError("duplicate sealed query_id")
        seen.add(row["query_id"])
        checkpoints.add(row["checkpoint_sha256"])
        cohorts.add(row["cohort"])
        measured = candidate_metrics(row, **config["candidate_metrics"])
        q = {k: row[k] for k in ("task", "episode_id", "query_id")}
        local = Counter(queries=1, candidates=measured["candidate_count"],
                        complete_endpoints=measured["endpoint_count"],
                        known_labels=measured["known_label_count"],
                        comparable_pairs=measured["comparable_pairs"],
                        comparable_queries=int(measured["comparable_pairs"] > 0))
        for c in row["candidates"]:
            if not c["valid"]:
                continue
            status = c["endpoint_status"]
            endpoint_statuses[status] += 1
            local["unknown_labels" if c.get("applicable") is None else
                  ("positive_labels" if c["applicable"] == 1 else "negative_labels")] += 1
            if status in {"not_attempted", "missing_result", "attempted", "invalid_branch"}:
                local["unresolved_branches"] += 1
        for metric in METRICS:
            shared = all(measured["references"][ref][metric] is not None for ref in REFERENCES)
            local[f"shared_queries/{metric}"] = int(shared)
            for ref in REFERENCES:
                q[f"{ref}/{metric}"] = measured["references"][ref][metric] if shared else None
            for baseline in ("historical", "query"):
                q[f"predicted_minus_{baseline}/{metric}"] = (
                    q[f"predicted/{metric}"] - q[f"{baseline}/{metric}"] if shared else None)
        queries.append(q)
        coverage.update(local)
        task_coverage[row["task"]].update(local)
    if len(checkpoints) != 1 or len(cohorts) != 1:
        raise ValueError("Table 4 cannot mix checkpoints or cohorts")
    common = {k: config[k] for k in ("tasks", "bootstrap", "seed")}
    names = [key for key in queries[0] if "/" in key]
    aggregates = {name: clustered_macro(queries, name, **common) for name in names}
    blockers = []
    if coverage["unresolved_branches"]:
        blockers.append("some declared branches were not completed or remain invalid")
    for ref in REFERENCES:
        for metric in METRICS:
            result = aggregates[f"{ref}/{metric}"]
            if result["mean"] is None or result["ci95"] is None:
                blockers.append(f"{ref}/{metric}: {result['missing_reason']}")
    return {"schema": "warm.nonreal.table4.v1", "status": "blocked" if blockers else "measured",
            "blockers": blockers, "checkpoint_sha256": next(iter(checkpoints)),
            "cohort": next(iter(cohorts)), "coverage": dict(coverage),
            "task_coverage": {k: dict(v) for k, v in task_coverage.items()},
            "endpoint_statuses": dict(endpoint_statuses), "per_query": queries,
            "aggregates": aggregates, "config": config,
            "scope": "Executed candidate effects; equal-task, episode-clustered estimates. "
                     "This report does not qualify the simulator backend or verify label truth."}


def latex_rows(report):
    """Export measured values only; no presumed ranking or favorable boldface."""
    if report["status"] != "measured":
        raise ValueError("Table 4 is incomplete; read blockers and coverage")
    labels = {"historical": r"Historical $E_i$", "query": r"Query-only $r_t$",
              "predicted": r"Predicted $\widehat E_i$"}
    lines = ["% Measured branch evidence; see table4.json for scope, coverage and paired intervals."]
    for ref in REFERENCES:
        v = [report["aggregates"][f"{ref}/{metric}"]["mean"] for metric in METRICS]
        lines.append(f"{labels[ref]} & {v[0]:.4f} & {v[1]:.3f} & "
                     f"{100*v[2]:.1f}\\% & {100*v[3]:.1f}\\% " + r"\\")
    return "\n".join(lines) + "\n"
