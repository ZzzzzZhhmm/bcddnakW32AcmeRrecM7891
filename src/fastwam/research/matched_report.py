"""Fixed-cohort reporting for the independently trained Full/pooled study.

Episode-seed uncertainty is conditional on one training seed. This module never
treats missing/runtime outcomes as failures or performs a missing-skill NI test.
"""
from __future__ import annotations

from fastwam.research.paired_binary import paired_difference_bounds

TASKS = ("rearrange_blocks", "battery_try")
ROLES = ("full", "pooled_memory_prior")
KNOWN = ("complete_native", "identifiable_policy_failure")
IDENTITY_KEYS = ("design_sha256", "evaluation_source_sha256",
                 "full_checkpoint_sha256", "pooled_checkpoint_sha256",
                 "root_seed", "checkpoint_update")


def summarize_fixed_pairs(rows):
    """Summarize exactly the registered 2 x 50 cohort without replacing cases."""
    buckets = {task: {} for task in TASKS}
    identities = set()
    for row in rows:
        task, index = row.get("task"), row.get("episode_id")
        if task not in TASKS or type(index) is not int or not 0 <= index < 50:
            raise ValueError("Unexpected task/case outside the fixed cohort")
        if row.get("phase") != "formal":
            raise ValueError("Recorded DEV cannot provide native task success")
        if index in buckets[task]:
            raise ValueError("Duplicate pair; never choose by outcome")
        identity = row.get("study_identity", {})
        if any(identity.get(k) is None for k in IDENTITY_KEYS):
            raise ValueError("Complete fixed model/design identity required")
        if identity["root_seed"] != 3407 or identity["checkpoint_update"] != 3000:
            raise ValueError("Fixed training seed/final update differs")
        if identity["full_checkpoint_sha256"] == identity["pooled_checkpoint_sha256"]:
            raise ValueError("The two independently trained roles must differ")
        identities.add(tuple(identity[k] for k in IDENTITY_KEYS))
        roles = row.get("roles", {})
        if set(roles) - set(ROLES):
            raise ValueError("Foreign control role; candidate-null is a distinct study")
        known = []
        for role in ROLES:
            value = roles.get(role, {})
            status, outcome = value.get("status"), value.get("success")
            if status in KNOWN:
                if type(outcome) is not int or outcome not in (0, 1):
                    raise ValueError("Known native outcome requires an integer binary label")
                if status == "identifiable_policy_failure" and outcome != 0:
                    raise ValueError("Identified invalid policy output must be a failure")
                known.append(role)
            elif outcome is not None:
                raise ValueError("Runtime, pending and qualification outcomes must stay unknown")
        if len(known) == 2 and row.get("paired_initial_and_noise_verified") is not True:
            raise ValueError("Known pairs require actual initial-input and sampler matching")
        buckets[task][index] = row
    if len(identities) > 1:
        raise ValueError("Do not pool different checkpoints, training seeds or designs")
    result = dict(
        schema="warm.rebuttal.fixed-full-pooled-report.v1",
        planned_pairs=100, planned_policy_episodes=200,
        training_seeds=1, tasks={}, cohort_complete=False, primary=None,
        descriptive_macro_difference=None, descriptive_macro_ci95=None,
        missing_skill_noninferiority_test_performed=False,
        scope="Fixed two-task episode-seed uncertainty conditional on seed3407/final3000. "
              "Incomplete complete-case results are descriptive; not original specialist "
              "accuracy, named-baseline replication or a missing-skill NI result.")
    for task, values in buckets.items():
        cells = dict(both_success=0, both_failure=0, full_only=0, pooled_only=0)
        pending = []
        known_roles = {r: 0 for r in ROLES}
        policy_failures = {r: 0 for r in ROLES}
        for index in range(50):
            row = values.get(index)
            if row is None:
                pending.append(index)
                continue
            roles = row["roles"]
            for role in ROLES:
                if roles.get(role, {}).get("status") in KNOWN:
                    known_roles[role] += 1
                    policy_failures[role] += int(roles[role]["status"] == "identifiable_policy_failure")
            if not all(roles.get(r, {}).get("status") in KNOWN for r in ROLES):
                pending.append(index)
                continue
            a, b = (roles[r]["success"] for r in ROLES)
            key = "both_success" if a and b else "full_only" if a else "pooled_only" if b else "both_failure"
            cells[key] += 1
        n = sum(cells.values())
        bounds = paired_difference_bounds(cells["full_only"], cells["pooled_only"], n,
                                          simultaneous_tasks=2) if n else None
        result["tasks"][task] = dict(
            planned_pairs=50, observed_case_records=len(values), complete_pairs=n,
            pending_or_unknown_pairs=50-n, pending_or_unknown_case_ids=pending,
            known_role_outcomes=known_roles, identifiable_policy_failures=policy_failures,
            cells=cells,
            full_successes=cells["both_success"]+cells["full_only"],
            pooled_successes=cells["both_success"]+cells["pooled_only"],
            complete_case_bounds=bounds)
    result["cohort_complete"] = all(x["complete_pairs"] == 50 for x in result["tasks"].values())
    stats = [x["complete_case_bounds"] for x in result["tasks"].values()]
    if all(x is not None for x in stats):
        result["descriptive_macro_difference"] = sum(x["difference"] for x in stats)/2
        result["descriptive_macro_ci95"] = [sum(x["ci95_conservative"][i] for x in stats)/2 for i in (0, 1)]
    if result["cohort_complete"]:
        result["primary"] = dict(
            tasks=result["tasks"], equal_task_macro_difference=result["descriptive_macro_difference"],
            macro_ci95_conservative=result["descriptive_macro_ci95"],
            alpha=.05, simultaneous_tasks=2,
            method="Exact gain/harm marginal tails with Bonferroni union bound; "
                   "equal-task macro from simultaneous task bounds.")
    return result
