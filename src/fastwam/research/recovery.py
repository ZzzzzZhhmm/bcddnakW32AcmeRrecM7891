"""Paired local-action recovery statistics, never inferred from episode success.

Inputs are independently labelled raw/refined/candidate-null branches assembled
by a collector. Metadata validation here does not certify simulator restoration
or the truth of a label; the collector's sealed traces remain required evidence.
"""
from __future__ import annotations

from collections import defaultdict
import re

import numpy as np


ROLES = ("raw", "refined", "candidate_null")
METRICS = ("repair_rate", "harm_rate", "net_applicability_change",
           "refined_minus_null_applicability")


def _validate(rows, tasks):
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("nonempty distinct fixed tasks required")
    seen, identities = set(), set()
    for row in rows:
        if row.get("schema") != "warm.recovery.query.v1":
            raise ValueError("unsupported recovery schema")
        key = (row["task"], str(row["episode_id"]), row["query_id"])
        if row["task"] not in tasks or key in seen or not key[2]:
            raise ValueError("unexpected task or duplicate/empty query identity")
        seen.add(key)
        for name in ("protocol_sha256", "checkpoint_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", row.get(name, "")) is None:
                raise ValueError(f"valid {name} required")
        if not row.get("cohort") or type(row.get("horizon")) is not int or row["horizon"] <= 0:
            raise ValueError("cohort and positive execution horizon required")
        identities.add(tuple(row[k] for k in ("protocol_sha256", "checkpoint_sha256", "cohort", "horizon")))
        gate = row.get("gate")
        if type(gate) not in (int, float) or not np.isfinite(gate) or not 0 <= gate <= 1:
            raise ValueError("finite natural gate in [0,1] required")
        if row.get("gate_intervention") != "natural":
            raise ValueError("forced gate controls must not enter natural recovery statistics")
        if set(row.get("branches", {})) != set(ROLES):
            raise ValueError("raw, refined and candidate_null branches required")
        for branch in row["branches"].values():
            label = branch.get("applicable")
            if label is not None and (type(label) is not int or label not in (0, 1)):
                raise ValueError("independent applicability must be 0, 1 or null")
            if label is not None and not branch.get("label_provenance"):
                raise ValueError("known applicability requires independent label_provenance")
    if len(identities) > 1:
        raise ValueError("do not mix checkpoints, protocols, cohorts or horizons")
    return next(iter(identities), None)


def _counts(rows):
    counts = {"n00": 0, "n01": 0, "n10": 0, "n11": 0,
              "queries": len(rows), "complete_queries": 0,
              "refined_minus_null_count": 0,
              "unknown_by_role": {role: 0 for role in ROLES}}
    for row in rows:
        labels = {k: row["branches"][k]["applicable"] for k in ROLES}
        for role, label in labels.items():
            counts["unknown_by_role"][role] += int(label is None)
        # All four metrics use the same three-branch, complete-case support.
        if any(v is None for v in labels.values()):
            continue
        counts["complete_queries"] += 1
        counts[f"n{labels['raw']}{labels['refined']}"] += 1
        counts["refined_minus_null_count"] += labels["refined"] - labels["candidate_null"]
    return counts


def _rates(counts):
    invalid_raw = counts["n00"] + counts["n01"]
    valid_raw = counts["n10"] + counts["n11"]
    n = counts["complete_queries"]
    return {
        "repair_rate": counts["n01"] / invalid_raw if invalid_raw else None,
        "harm_rate": counts["n10"] / valid_raw if valid_raw else None,
        "net_applicability_change": (counts["n01"] - counts["n10"]) / n if n else None,
        "refined_minus_null_applicability": counts["refined_minus_null_count"] / n if n else None,
    }


def summarize_recovery(rows, *, tasks, bootstrap=10000, seed=3407):
    """Task-specific count ratios and equal fixed-task macro with cluster CIs.

    Resample source episodes within task, retaining all their query records.
    Missing labels are reported and excluded on common complete-case support;
    no missing label is set to zero, and no successful resample is selected.
    Rates/differences are fractions, not percentage points. A single checkpoint
    supplies evaluation uncertainty, not training-seed uncertainty.
    """
    rows = list(rows)
    if type(bootstrap) is not int or bootstrap < 1:
        raise ValueError("positive bootstrap count required")
    identity = _validate(rows, tasks)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], str(row["episode_id"]))].append(row)
    task_rows = {t: [r for r in rows if r["task"] == t] for t in tasks}
    summaries = {}
    for task, rr in task_rows.items():
        c = _counts(rr)
        strata = {}
        for label, selected in (("zero_gate", [r for r in rr if r["gate"] == 0]),
                                ("positive_gate", [r for r in rr if r["gate"] > 0])):
            count = _counts(selected)
            strata[label] = {"counts": count, "rates": _rates(count)}
        summaries[task] = {"counts": c, "rates": _rates(c),
                           "source_episodes": sum(t == task for t, _ in groups),
                           "natural_gate_strata": strata}
    episode_rows = {t: [rr for (task, _), rr in sorted(groups.items()) if task == t] for t in tasks}
    estimates = {name: [] for name in METRICS}
    can_bootstrap = all(len(ep) >= 2 for ep in episode_rows.values())
    if can_bootstrap:
        rng = np.random.default_rng(seed)
        for _ in range(bootstrap):
            draw = []
            for ep in episode_rows.values():
                sampled = [r for i in rng.integers(len(ep), size=len(ep)) for r in ep[i]]
                draw.append(_rates(_counts(sampled)))
            for name in METRICS:
                values = [r[name] for r in draw]
                estimates[name].append(float(np.mean(values)) if all(v is not None for v in values) else np.nan)
    macro = {}
    for name in METRICS:
        values = [summaries[t]["rates"][name] for t in tasks]
        mean = float(np.mean(values)) if all(v is not None for v in values) else None
        missing = ("a fixed task has no eligible denominator" if mean is None else
                   "fewer than two source episodes in a fixed task" if not can_bootstrap else None)
        bad = int(np.isnan(estimates[name]).sum()) if can_bootstrap else None
        ci = None
        if can_bootstrap and mean is not None:
            if bad:
                missing = "zero denominators in bootstrap; do not condition on defined resamples"
            else:
                ci = np.quantile(estimates[name], [.025, .975]).tolist()
        macro[name] = {"mean": mean, "ci95": ci,
                       "undefined_bootstrap_replicates": bad, "missing_reason": missing}
    return {
        "schema": "warm.recovery.summary.v1", "identity": identity,
        "tasks": summaries, "macro": macro, "bootstrap_replicates": bootstrap,
        "bootstrap_seed": seed, "unit": "fraction", "cluster_unit": "source episode within fixed task",
        "support": "same three-branch complete cases for all metrics; unknowns retained in counts",
        "scope": "local independently labelled action branches; not episode success or causal SR contribution",
        "validation_scope": "semantic input checks only; sealed collector traces must separately certify restoration, action identity and independent labels",
    }
