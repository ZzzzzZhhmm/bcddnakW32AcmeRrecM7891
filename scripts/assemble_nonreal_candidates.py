#!/usr/bin/env python3
"""Join sealed model probes, real branch results and independent labels.

Index JSONL fields: task, episode_id, query_id, cohort, probe_path,
probe_metadata_sha256. Candidate branch IDs are rank-0, rank-1, ... in the
sealed proposal. Missing endpoints and missing independent labels remain null.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastwam.research.evidence import canonical


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def assemble(index, branches, labels, *, kind="candidates"):
    if kind not in {"candidates", "gates"}:
        raise ValueError("unsupported assembly kind")
    attempts, results, annotations = {}, {}, {}
    for record in branches:
        key = (record["query_id"], record["candidate_id"])
        if record["kind"] not in {"branch_attempt", "branch_result"}:
            raise ValueError("unsupported branch record kind")
        bucket = attempts if record["kind"] == "branch_attempt" else results
        if key in bucket:
            raise ValueError("duplicate branch record; repeat branches need separate query IDs")
        bucket[key] = record
    for record in labels:
        key = (record["query_id"], record["candidate_id"])
        if key in annotations:
            raise ValueError("duplicate independent label")
        label = record.get("applicable")
        if label is not None and (type(label) is not int or label not in (0, 1) or not record.get("label_provenance")):
            raise ValueError("label must be 0/1/null with provenance when known")
        annotations[key] = record
    output, queries, used = [], set(), set()
    for row in index:
        query = row["query_id"]
        if query in queries:
            raise ValueError("duplicate query_id in frozen index")
        queries.add(query)
        path = Path(row["probe_path"])
        metadata_raw = (path / "proposal.json").read_bytes()
        digest = hashlib.sha256(metadata_raw).hexdigest()
        if row["probe_metadata_sha256"] != digest:
            raise ValueError("probe metadata identity mismatch")
        metadata = json.loads(metadata_raw)
        if metadata["query_id"] != query or metadata["capture_phase"] != "before_any_branch_truth":
            raise ValueError("proposal identity or capture phase mismatch")
        if hashlib.sha256((path / "proposal.npz").read_bytes()).hexdigest() != metadata["array_sha256"]:
            raise ValueError("proposal array identity mismatch")
        with np.load(path / "proposal.npz", allow_pickle=False) as arrays:
            if arrays["valid"].ndim != 2 or arrays["valid"].shape[0] != 1:
                raise ValueError("collection expects one factual query per probe")
            valid = arrays["valid"][0]
            if valid.dtype != np.bool_:
                raise ValueError("valid mask must be bool")
            horizon = arrays["adapted_actions"].shape[2]
            candidates = []
            for rank in np.flatnonzero(valid):
                candidate_id = f"rank-{rank}"
                key = (query, candidate_id)
                used.add(key)
                attempt, result = attempts.get(key), results.get(key)
                for branch in (attempt, result):
                    if branch is not None and branch.get("proposal_sha256") != digest:
                        raise ValueError("branch came from a different sealed proposal")
                if result is not None and (attempt is None or result.get("parent_restored") is not True):
                    raise ValueError("branch result lacks attempt or verified parent restoration")
                if result is not None and result["horizon"] != horizon:
                    raise ValueError("branch horizon differs from adapted action horizon")
                if result is not None and attempt.get("parent_fingerprint") != result.get("parent_fingerprint"):
                    raise ValueError("attempt and result disagree on parent state")
                label = annotations.get(key, {})
                candidates.append({
                    "candidate_id": candidate_id, "valid": True, "horizon": int(horizon),
                    "event_id": metadata["identity"].get("event_ids", [None] * len(valid))[rank],
                    "predicted_effect": arrays["predicted_effect"][0, rank].tolist(),
                    "historical_effect": arrays["historical_effect"][0, rank].tolist(),
                    "observed_effect": result.get("observed_effect") if result else None,
                    "endpoint_status": result["endpoint_status"] if result else ("missing_result" if attempt else "not_attempted"),
                    "executed_steps": result["executed_steps"] if result else None,
                    "termination": result.get("termination") if result else None,
                    "applicable": label.get("applicable"), "label_provenance": label.get("label_provenance"),
                })
            assembled = {**{k: row[k] for k in ("task", "episode_id", "query_id", "cohort")},
                           "probe_metadata_sha256": digest, "checkpoint_sha256": metadata["identity"].get("checkpoint_sha256"),
                           "required_effect": arrays["required_effect"][0].tolist(), "candidates": candidates}
            if kind == "gates":
                if metadata.get("source_mode") != "full" or metadata.get("comparison") is not True or metadata.get("force_null") is not False:
                    raise ValueError("learned gate analysis requires unmodified natural model controls")
                for key in ("selected_index", "alpha", "g", "v_det"):
                    if arrays[key].size != 1:
                        raise ValueError("gate assembly expects one selected query")
                if arrays["selected_index"].dtype.kind not in "iu" or arrays["v_det"].dtype != np.bool_:
                    raise ValueError("selected index must be integer and v_det boolean")
                if arrays["candidate_v_det"].shape != arrays["valid"].shape or arrays["candidate_v_det"].dtype != np.bool_:
                    raise ValueError("candidate eligibility mask must match candidate validity")
                eligible_count = int((arrays["candidate_v_det"][0] & valid).sum())
                selected = int(arrays["selected_index"].item())
                empty = not bool(valid.any())
                if not empty and (not 0 <= selected < len(valid) or not valid[selected]):
                    raise ValueError("selected candidate is not valid")
                label = annotations.get((query, f"rank-{selected}"), {}) if not empty else {}
                assembled = {**{k: assembled[k] for k in ("task", "episode_id", "query_id", "cohort", "probe_metadata_sha256", "checkpoint_sha256")},
                             "selected_candidate_id": None if empty else f"rank-{selected}",
                             "empty_pool": empty, "intervention": "none",
                             "alpha": float(arrays["alpha"].item()), "g": float(arrays["g"].item()),
                             "v_det": bool(arrays["v_det"].item()), "threshold": metadata["gate_threshold"],
                             "applicable": label.get("applicable"), "label_provenance": label.get("label_provenance"),
                             "candidate_count": len(candidates),
                             "eligible_candidate_count": eligible_count,
                             "known_candidate_labels": sum(c["applicable"] is not None for c in candidates),
                             "verified_all_inapplicable": bool(eligible_count) and all(c["applicable"] == 0 for c in candidates)}
            output.append(assembled)
    if (set(attempts) | set(results) | set(annotations)) - used:
        raise ValueError("unmatched branch or label identities")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("index", "branches", "labels", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--kind", choices=("candidates", "gates"), default="candidates")
    args = parser.parse_args()
    rows = assemble(read_jsonl(args.index), read_jsonl(args.branches), read_jsonl(args.labels), kind=args.kind)
    if not rows:
        raise ValueError("no frozen queries")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as f:
        for row in rows:
            f.write(canonical(row) + b"\n")
    print(json.dumps({"output": str(args.output), "queries": len(rows)}))


if __name__ == "__main__":
    main()
