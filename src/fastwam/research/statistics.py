"""Fixed-task, episode-clustered statistics for WARM's non-real evidence.

All rates are fractions, all effects/differences use the input scale. Unknown
measurements stay null. This module never manufactures episodes from tables.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import math
from typing import Iterable, Mapping

import numpy as np


def finite(value, name="value") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _vector(value, name):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return a


def kappa(effect, required, magnitude_weight: float, eps=1e-6):
    e, r = _vector(effect, "effect"), _vector(required, "required")
    if e.shape != r.shape:
        raise ValueError("effect dimensions differ")
    ne, nr = np.linalg.norm(e), np.linalg.norm(r)
    # Matches consequence_consistency's epsilon and zero-vector convention.
    return float(np.clip(np.dot(e, r) / (max(ne, eps) * max(nr, eps)), -1, 1)
                 - magnitude_weight * abs(math.log((ne + eps) / (nr + eps))))


def candidate_metrics(row: Mapping, *, delta_obs: float, delta_pred: float,
                      magnitude_weight: float, top_tolerance: float) -> dict:
    """One query; branch labels are independent of endpoint availability."""
    for name, value in (("delta_obs", delta_obs), ("delta_pred", delta_pred),
                        ("magnitude_weight", magnitude_weight), ("top_tolerance", top_tolerance)):
        if finite(value, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    required = _vector(row["required_effect"], "required_effect")
    candidates = row["candidates"]
    if any(type(c.get("valid")) is not bool for c in candidates):
        raise ValueError("candidate valid must be bool")
    if len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("duplicate candidate_id")
    candidates = [c for c in candidates if c["valid"] is True]
    names = ("predicted", "historical", "query")
    scores = {name: [] for name in names}
    errors = {name: [] for name in names}
    observed = {}
    labels = []
    for i, c in enumerate(candidates):
        label = c.get("applicable")
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise ValueError("applicable must be 0, 1 or null")
        if label is not None and not c.get("label_provenance"):
            raise ValueError("known applicability needs independent label_provenance")
        labels.append(label)
        effects = {"predicted": _vector(c["predicted_effect"], "predicted_effect"),
                   "historical": _vector(c["historical_effect"], "historical_effect"),
                   "query": required}
        for name, effect in effects.items():
            scores[name].append(kappa(effect, required, magnitude_weight))
        if c.get("endpoint_status") == "complete_horizon":
            if (type(c.get("executed_steps")) is not int or type(c.get("horizon")) is not int
                or c["executed_steps"] != c["horizon"] or c["horizon"] <= 0):
                raise ValueError("complete endpoint must have exactly H executed commands")
            obs = _vector(c["observed_effect"], "observed_effect")
            if obs.shape != required.shape:
                raise ValueError("observed effect dimensions differ")
            observed[i] = kappa(obs, required, magnitude_weight)
            for name, effect in effects.items():
                errors[name].append(float(np.mean((effect - obs) ** 2)))
        elif c.get("observed_effect") is not None:
            raise ValueError("incomplete endpoints must not contain imputed observed_effect")
    pairs = [(i, j) for i, j in combinations(observed, 2)
             if abs(observed[i] - observed[j]) > delta_obs]
    result = {"candidate_count": len(candidates), "endpoint_count": len(observed),
              "known_label_count": sum(x is not None for x in labels),
              "comparable_pairs": len(pairs), "references": {}}
    for name in names:
        agreements, ties = [], []
        for i, j in pairs:
            diff = scores[name][i] - scores[name][j]
            tie = abs(diff) <= delta_pred
            ties.append(float(tie))
            agreements.append(0.5 if tie else float(diff * (observed[i] - observed[j]) > 0))
        best = max(scores[name]) if candidates else None
        top = [i for i, s in enumerate(scores[name]) if best - s <= top_tolerance]
        complete_labels = bool(top) and all(labels[i] is not None for i in top)
        result["references"][name] = {
            "effect_mse": float(np.mean(errors[name])) if errors[name] else None,
            "pairwise_agreement": float(np.mean(agreements)) if pairs else None,
            "predicted_tie_rate": float(np.mean(ties)) if pairs else None,
            "top_choice_applicability": float(np.mean([labels[i] for i in top])) if complete_labels else None,
            "top_tie_count": len(top), "top_label_coverage": complete_labels,
        }
    # Paired differences use the same query support for both references.
    for baseline in ("historical", "query"):
        for metric in ("effect_mse", "pairwise_agreement", "top_choice_applicability"):
            a, b = result["references"]["predicted"][metric], result["references"][baseline][metric]
            result[f"predicted_minus_{baseline}/{metric}"] = None if a is None or b is None else a - b
    return result


def clustered_macro(rows: Iterable[Mapping], metric: str, *, tasks: list[str],
                    bootstrap=10000, seed=3407) -> dict:
    """Query mean -> source-episode mean -> equal fixed-task mean.

    Resamples source episodes WITHIN each fixed task. It never resamples queries
    as independent observations, nor silently drops a requested task.
    """
    if not tasks or len(set(tasks)) != len(tasks) or bootstrap < 1:
        raise ValueError("unique tasks and positive bootstrap count required")
    buckets = defaultdict(list)
    seen = set()
    total, missing = 0, 0
    for row in rows:
        key = (row["task"], row["episode_id"], row["query_id"])
        if key in seen:
            raise ValueError("duplicate query identity")
        seen.add(key)
        if row["task"] not in tasks:
            raise ValueError("unexpected task")
        total += 1
        value = row.get(metric)
        if value is None:
            missing += 1
        else:
            buckets[key[:2]].append(finite(value, metric))
    by_task = {task: np.array([np.mean(v) for (t, _), v in sorted(buckets.items()) if t == task])
               for task in tasks}
    means = {t: float(v.mean()) if len(v) else None for t, v in by_task.items()}
    result = {"metric": metric, "task_means": means,
              "episode_counts": {t: len(v) for t, v in by_task.items()},
              "queries": total, "missing_queries": missing, "mean": None,
              "ci95": None, "bootstrap_seed": seed, "bootstrap_replicates": bootstrap}
    if any(not len(v) for v in by_task.values()):
        result["missing_reason"] = "at least one required task has no scored episode"
        return result
    result["mean"] = float(np.mean(list(means.values())))
    if any(len(v) < 2 for v in by_task.values()):
        result["missing_reason"] = "CI needs at least two source episodes in every task"
        return result
    rng = np.random.default_rng(seed)
    draws = np.zeros(bootstrap)
    for values in by_task.values():
        # Bound memory even for a large legacy outcome archive.
        for begin in range(0, bootstrap, 256):
            n = min(256, bootstrap - begin)
            draws[begin:begin+n] += values[rng.integers(len(values), size=(n, len(values)))].mean(axis=1) / len(tasks)
    result["ci95"] = np.quantile(draws, [0.025, 0.975]).tolist()
    result["missing_reason"] = None
    return result


def paired_success(rows: Iterable[Mapping], contrasts: Mapping[str, Mapping[str, float]],
                   *, tasks: list[str], bootstrap=10000, seed=3407) -> dict:
    """Reject incomplete pairing instead of silently switching to unpaired CI.

    pairing_id must identify the same layout/reset block across conditions;
    checkpoint identity and seeds remain mandatory provenance in each row.
    """
    rows = list(rows)
    conditions = sorted({c for weights in contrasts.values() for c in weights})
    blocks, exclusions = defaultdict(dict), []
    for row in rows:
        for key in ("task", "pairing_id", "condition", "run_id", "checkpoint_sha256", "reset_seed", "eval_run"):
            if row.get(key) is None:
                raise ValueError(f"missing pairing provenance: {key}")
        if row["task"] not in tasks or row["condition"] not in conditions:
            raise ValueError("unexpected task or condition")
        if row.get("valid_trial") is not True:
            if not row.get("exclusion_reason"):
                raise ValueError("invalid trial needs exclusion_reason")
            exclusions.append(row)
            continue
        if type(row.get("success")) is not bool:
            raise ValueError("success must be bool")
        key = (row["task"], str(row["pairing_id"]))
        if row["condition"] in blocks[key]:
            raise ValueError("duplicate condition in pairing block")
        blocks[key][row["condition"]] = row
    incomplete = [list(key) for key, b in blocks.items() if set(b) != set(conditions)]
    if incomplete:
        raise ValueError(f"incomplete paired blocks: {incomplete[:5]}; explicitly repair or specify a separate unpaired analysis")
    samples = []
    checkpoints = defaultdict(set)
    for (task, pair), block in sorted(blocks.items()):
        if len({(r["reset_seed"], str(r["eval_run"])) for r in block.values()}) != 1:
            raise ValueError("paired conditions disagree on reset_seed/eval_run")
        if len({str(r.get("layout_id")) for r in block.values()}) != 1:
            raise ValueError("paired conditions disagree on layout_id")
        for condition, r in block.items():
            checkpoints[(task, condition)].add(r["checkpoint_sha256"])
        first = next(iter(block.values()))
        # Repeated evaluations of one layout/reset stay in one cluster.
        cluster = str(first.get("cluster_id", first["reset_seed"]))
        if any(str(r.get("cluster_id", r["reset_seed"])) != cluster for r in block.values()):
            raise ValueError("paired conditions disagree on cluster_id")
        sample = {"task": task, "episode_id": cluster, "query_id": pair}
        for name, weights in contrasts.items():
            sample[name] = sum(finite(w) * int(block[c]["success"]) for c, w in weights.items())
        samples.append(sample)
    if any(len(v) != 1 for v in checkpoints.values()):
        raise ValueError("multiple checkpoints within task/condition need a separate training-seed analysis")
    per_run = defaultdict(lambda: defaultdict(list))
    for (task, _), block in blocks.items():
        for condition, r in block.items():
            per_run[(condition, str(r["eval_run"]))][task].append(int(r["success"]))
    run_macros, run_details = defaultdict(list), []
    for (condition, eval_run), values in sorted(per_run.items()):
        task_rates = {t: float(np.mean(values[t])) if values.get(t) else None for t in tasks}
        macro = float(np.mean(list(task_rates.values()))) if all(v is not None for v in task_rates.values()) else None
        run_details.append({"condition": condition, "eval_run": eval_run, "task_rates": task_rates,
                            "counts": {t: {"successes": sum(values[t]), "episodes": len(values[t])} for t in tasks},
                            "macro": macro})
        run_macros[condition].append(macro)
    run_summary = {}
    for condition, values in run_macros.items():
        complete = all(v is not None for v in values)
        run_summary[condition] = {
            "evaluation_runs": len(values),
            "mean": float(np.mean(values)) if complete else None,
            "sample_std": float(np.std(values, ddof=1)) if complete and len(values) > 1 else None,
            "variation_source": "evaluation_runs_same_checkpoint_per_task",
            "missing_reason": None if complete and len(values) > 1 else "missing task coverage or fewer than two evaluation runs",
        }
    return {"paired_blocks": len(samples), "excluded_trials": len(exclusions),
            "excluded_records": exclusions,
            "evaluation_run_details": run_details, "evaluation_run_summary": run_summary,
            "contrasts": {name: clustered_macro(samples, name, tasks=tasks, bootstrap=bootstrap, seed=seed)
                          for name in contrasts}}


def gate_rates(rows: Iterable[Mapping]) -> dict:
    """Natural selected-candidate diagnostic; threshold is the deployed alpha threshold."""
    counts = {label: {"eligible": 0, "accepted": 0, "known": 0, "total": 0,
                      "hard_veto": 0, "raw_above_threshold": 0} for label in (0, 1, "unknown")}
    gate_values = {label: [] for label in counts}
    empty = 0
    for row in rows:
        if row.get("intervention", "none") != "none":
            raise ValueError("forced controls cannot be evidence of learned gate rejection")
        label = row.get("applicable")
        if label is not None and (type(label) is not int or label not in (0, 1) or not row.get("label_provenance")):
            raise ValueError("gate labels need independent provenance")
        if type(row.get("v_det")) is not bool:
            raise ValueError("v_det must exclude alpha threshold and be bool")
        alpha, threshold = finite(row["alpha"]), finite(row["threshold"])
        if not 0 <= alpha <= 1 or not 0 <= threshold <= 1:
            raise ValueError("gate values must be in [0,1]")
        if row.get("empty_pool", False):
            if label is not None or row["v_det"]:
                raise ValueError("empty pool cannot have a selected label or eligible candidate")
            empty += 1
            continue
        group = "unknown" if label is None else label
        count = counts[group]
        count["total"] += 1
        count["known"] += int(label is not None)
        count["hard_veto"] += int(not row["v_det"])
        count["raw_above_threshold"] += int(alpha >= threshold)
        count["eligible"] += int(row["v_det"])
        count["accepted"] += int(row["v_det"] and alpha >= threshold)
        if row.get("g") is not None:
            g = finite(row["g"], "g")
            if not 0 <= g <= 1:
                raise ValueError("g must be in [0,1]")
            gate_values[group].append(g)
    for group, count in counts.items():
        count["g_observations"] = len(gate_values[group])
        count["mean_g"] = float(np.mean(gate_values[group])) if gate_values[group] else None
        count["raw_acceptance_rate"] = count["raw_above_threshold"] / count["total"] if count["total"] else None
    return {"unknown_labels": counts["unknown"]["total"], "empty_queries": empty, "counts": counts,
            "false_acceptance": counts[0]["accepted"] / counts[0]["eligible"] if counts[0]["eligible"] else None,
            "true_acceptance": counts[1]["accepted"] / counts[1]["eligible"] if counts[1]["eligible"] else None}


def gate_clustered(rows: Iterable[Mapping], *, tasks: list[str], bootstrap=10000, seed=3407) -> dict:
    """Within-task episode bootstrap of count ratios; report sparse support."""
    if not tasks or len(set(tasks)) != len(tasks) or bootstrap < 1:
        raise ValueError("unique tasks and positive bootstrap count required")
    grouped, seen = defaultdict(list), set()
    for row in rows:
        identity = (row["task"], row["episode_id"], row["query_id"])
        if identity in seen or row["task"] not in tasks:
            raise ValueError("duplicate gate query or unexpected task")
        seen.add(identity)
        grouped[identity[:2]].append(row)
    counts = {key: gate_rates(value) for key, value in grouped.items()}
    task_summary = {task: gate_rates([r for (t, _), rr in grouped.items() if t == task for r in rr]) for task in tasks}
    result = {"tasks": task_summary, "bootstrap_seed": seed, "bootstrap_replicates": bootstrap, "macro": {}}
    for label, name in ((0, "false_acceptance"), (1, "true_acceptance")):
        values = {t: np.array([[v["counts"][label]["accepted"], v["counts"][label]["eligible"]]
                              for (task, _), v in sorted(counts.items()) if task == t], dtype=float).reshape(-1, 2)
                  for t in tasks}
        task_rates = [task_summary[t][name] for t in tasks]
        item = {"mean": float(np.mean(task_rates)) if all(v is not None for v in task_rates) else None,
                "ci95": None, "undefined_bootstrap_replicates": None, "missing_reason": None}
        if item["mean"] is None or any(len(v) < 2 for v in values.values()):
            item["missing_reason"] = "missing eligible class or fewer than two source episodes in a task"
        else:
            rng, draws = np.random.default_rng(seed), []
            for _ in range(bootstrap):
                rates = []
                for matrix in values.values():
                    num, den = matrix[rng.integers(len(matrix), size=len(matrix))].sum(axis=0)
                    if not den:
                        break
                    rates.append(num / den)
                draws.append(float(np.mean(rates)) if len(rates) == len(tasks) else np.nan)
            bad = int(np.isnan(draws).sum())
            item["undefined_bootstrap_replicates"] = bad
            if bad:
                item["missing_reason"] = "bootstrap has zero eligible denominators; do not condition CI on successful resamples"
            else:
                item["ci95"] = np.quantile(draws, [.025, .975]).tolist()
        result["macro"][name] = item
    return result


def donor_coverage(rows: Iterable[Mapping]) -> dict:
    """Coverage of the specified cohort, never retroactively of an old table."""
    rows, groups, seen = list(rows), defaultdict(list), set()
    for row in rows:
        key = (row["task"], row["condition"], row["episode_id"], row["query_id"])
        if key in seen:
            raise ValueError("duplicate donor query")
        seen.add(key)
        for field in ("valid_slots", "replaced_slots", "eligible_replaced_slots", "missing_donor_slots"):
            if type(row[field]) is not int or row[field] < 0:
                raise ValueError("donor counts must be nonnegative integers")
        if not 0 <= row["eligible_replaced_slots"] <= row["replaced_slots"] <= row["valid_slots"]:
            raise ValueError("inconsistent donor counts")
        if not row.get("protocol_sha256") or not row.get("cohort"):
            raise ValueError("donor protocol hash and cohort required")
        groups[(row["cohort"], row["protocol_sha256"], *key[:2])].append(row)
    output = []
    for key, group in sorted(groups.items()):
        totals = {f: sum(r[f] for r in group) for f in (
            "valid_slots", "replaced_slots", "eligible_replaced_slots", "missing_donor_slots")}
        output.append({"cohort": key[0], "protocol_sha256": key[1], "task": key[2], "condition": key[3],
                       **totals, "queries": len(group),
                       "replacement_rate": totals["replaced_slots"] / totals["valid_slots"] if totals["valid_slots"] else None,
                       "eligible_replacement_rate": totals["eligible_replaced_slots"] / totals["replaced_slots"] if totals["replaced_slots"] else None,
                       "query_distribution": group})
    return {"groups": output}
