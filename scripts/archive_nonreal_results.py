#!/usr/bin/env python3
"""Snapshot existing LIBERO results and recompute counts without running a model.

This validates recorded counts and identities, not simulator truth or paper-row
equivalence. Raw bytes are stored as gzip outside the checkout with SHA-256.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_counts(value):
    total = value["total_episodes"]
    if type(total) is not int or total < 1:
        raise ValueError("invalid episode denominator")
    success, failure = value["success_episodes"], value["failure_episodes"]
    if any(type(i) is not int for i in success + failure):
        raise ValueError("episode IDs must be integers")
    if len(set(success + failure)) != total or sorted(success + failure) != list(range(total)):
        raise ValueError("success/failure lists must partition every episode exactly once")
    if type(value["successes"]) is not int or value["successes"] != len(success):
        raise ValueError("success count mismatch")
    episodes = value["warm_online_episodes"]
    if len(episodes) != total or sorted(e["episode_index"] for e in episodes) != list(range(total)):
        raise ValueError("missing or duplicate episode telemetry")
    for episode in episodes:
        outcome = episode["success"]
        if type(outcome) is not bool or outcome != (episode["episode_index"] in success):
            raise ValueError("episode outcome disagrees with partition")
        if episode["termination_reason"] != ("success" if outcome else "max_steps"):
            raise ValueError("unknown or inconsistent endpoint")
        if episode["replan_count"] != len(episode["replans"]):
            raise ValueError("replan count mismatch")
    return len(success), total


def archive(batches, allowed_root, output):
    allowed_root, output = allowed_root.resolve(), output.resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("runtime archive must be outside source checkout")
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    files, rows, issues, groups = [], [], [], []

    def snapshot(path):
        path = path.resolve()
        if not path.is_relative_to(allowed_root):
            raise ValueError("artifact escapes explicitly allowed root")
        raw = path.read_bytes()
        sha = digest(raw)
        target = output / "raw" / (sha + ".gz")
        if not target.exists():
            target.write_bytes(gzip.compress(raw, compresslevel=1, mtime=0))
        files.append(dict(path=str(path), sha256=sha, bytes=len(raw), archive=str(target.relative_to(output))))
        return raw

    for batch in batches:
        batch = batch.resolve()
        summary = json.loads(snapshot(batch / "summary.json"))
        if (batch / "plan.tsv").is_file():
            snapshot(batch / "plan.tsv")
        batch_rows = []
        for record in summary["results"]:
            path = Path(record["result_path"])
            try:
                raw = snapshot(path)
                if digest(raw) != record["result_sha256"]:
                    raise ValueError("result hash differs from original completion marker")
                value = json.loads(raw)
                k, n = validate_counts(value)
                if (k, n, value["task_suite"], value["task_id"]) != (record["successes"], record["total_episodes"], record["suite"], record["task_id"]):
                    raise ValueError("task identity/count mismatch against completion marker")
                header = value["warm_online_header"]
                contract = header["contract"]
                if contract["root_seed"] != record["root_seed"]:
                    raise ValueError("root seed mismatch")
                if contract["warm_checkpoint_sha256"] != header["runtime_attestation"]["model_loaded_checkpoint_sha256"]:
                    raise ValueError("checkpoint identity mismatch")
                run_root = path.parents[2]
                compat = snapshot(run_root / "evaluation_compatibility.json")
                if digest(compat) != record["evaluation_compatibility_sha256"]:
                    raise ValueError("compatibility hash mismatch")
                snapshot(run_root / "resolved_config.yaml")
                snapshot(run_root / "online_contract.json")
                row = dict(batch=batch.name, suite=value["task_suite"], task_id=value["task_id"],
                           root_seed=record["root_seed"], successes=k, episodes=n, success_rate=k/n,
                           result_sha256=digest(raw), checkpoint_sha256=contract["warm_checkpoint_sha256"],
                           code_commit=contract["git_commit"], comparison_kind=header["comparison_kind"],
                           source_policy=header["source_policy"], bank_sha256=contract["bank_content_sha256"],
                           compatibility_sha256=digest(compat), duration_seconds=record["duration_seconds"],
                           result_path=str(path))
                rows.append(row)
                batch_rows.append(row)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                issues.append(dict(path=str(path), error=f"{type(exc).__name__}: {exc}"))
        k, n = sum(r["successes"] for r in batch_rows), sum(r["episodes"] for r in batch_rows)
        identities = {(r["suite"], r["task_id"], r["root_seed"]) for r in batch_rows}
        valid = (len(identities) == len(batch_rows) == summary["completed_jobs"] == len(summary["results"])
                 and (k, n) == (summary["total_successes"], summary["total_episodes"])
                 and n > 0 and abs(k/n-summary["overall_success_rate"]) < 1e-12)
        groups.append(dict(batch=batch.name, count_audit_passed=valid, successes=k, episodes=n,
                           success_rate=k/n if n else None, tasks=len(batch_rows),
                           recorded_task_runtime_hours=sum(r["duration_seconds"] for r in batch_rows)/3600))
    report = dict(schema="warm.recovered-result-archive.v1", created_utc=datetime.now(timezone.utc).isoformat(),
                  status="counts_verified" if not issues and all(g["count_audit_passed"] for g in groups) else "issues_found",
                  paper_row_status="pending_method_and_protocol_attribution", groups=groups, tasks=rows, issues=issues,
                  limitation="Recorded counts and artifact hashes only; no new evaluation, simulator replay, paired comparison, or training-seed variance.")
    for name, payload in (("audit.json", report), ("files.json", files)):
        (output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status", "groups", "issues")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=Path, required=True)
    parser.add_argument("--allowed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = archive(args.batches, args.allowed_root, args.output)
    raise SystemExit(0 if report["status"] == "counts_verified" else 2)
