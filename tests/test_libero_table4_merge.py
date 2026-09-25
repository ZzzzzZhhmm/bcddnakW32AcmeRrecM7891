import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from fastwam.research.evidence import append_record, canonical, write_probe
from fastwam.research.libero_table4 import make_plan

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
from merge_libero_table4 import merge
from plan_libero_table4_resume import plan_resume


def shard(root, episodes):
    root.mkdir()
    plan = make_plan(episodes=episodes)
    plan.update(qualification_only=False, cohort="synthetic-unit-test")
    (root/"plan.json").write_bytes(canonical(plan))
    n = 0
    for task in range(10):
        for ep in episodes:
            for frame in (80, 160):
                query = f"libero10-task{task:02d}-init{ep:02d}-frame{frame:04d}"
                effect = np.zeros((1, 32, 2))
                effect[0, 0, 0], effect[0, 1, 0] = 1, -1
                valid = np.zeros((1, 32), dtype=bool)
                valid[0, :2] = True
                probe = write_probe(root/"probes", query, dict(arrays=dict(
                    valid=valid, adapted_actions=np.zeros((1, 32, 32, 7)),
                    predicted_effect=effect, historical_effect=-effect,
                    required_effect=np.array([[1, 0]]))), dict(checkpoint_sha256="a"*64))
                append_record(root/"index.jsonl", dict(task=f"libero_10/{task}", episode_id=str(ep),
                    query_id=query, cohort=plan["cohort"], probe_path=probe["path"],
                    probe_metadata_sha256=probe["metadata_sha256"]))
                for rank in (0, 1):
                    common = dict(query_id=query, candidate_id=f"rank-{rank}",
                                  proposal_sha256=probe["metadata_sha256"], parent_fingerprint="synthetic",
                                  horizon=32, executed_steps=32, observed_effect=effect[0,rank].tolist(),
                                  endpoint_status="complete_horizon", parent_restored=True)
                    for kind in ("branch_attempt", "branch_result"):
                        append_record(root/"branches.jsonl", dict(kind=kind, **common))
                    append_record(root/"labels.jsonl", dict(query_id=query, candidate_id=f"rank-{rank}",
                        applicable=int(rank == 0), label_provenance="synthetic_unit_test"))
                n += 1
    (root/"execution.json").write_bytes(canonical(dict(completed=True, complete_queries=n, checkpoint_sha256="a"*64)))
    return root


def test_full_disjoint_collection_and_incomplete_or_duplicate_rejected(tmp_path):
    a = shard(tmp_path/"a", tuple(range(5)))
    b = shard(tmp_path/"b", tuple(range(5, 10)))
    rows, report = merge([a, b])
    assert len(rows) == 200
    assert report["planned_source_episodes"] == 100
    assert report["protocol"]["episodes"] == list(range(10))
    assert [p["episodes"] for p in report["shard_plans"]] == [list(range(5)),list(range(5,10))]
    assert report["status"] == "measured"
    assert report["aggregates"]["query/pairwise_agreement"]["mean"] == .5
    with pytest.raises(ValueError, match="overlapping"):
        merge([a, a])
    with pytest.raises(ValueError, match="requires all"):
        merge([a])
    (b/"execution.json").write_text(json.dumps(dict(completed=False, complete_queries=100)))
    with pytest.raises(ValueError, match="completed formal"):
        merge([a, b])


def test_resume_only_missing_initial_states(tmp_path):
    a = shard(tmp_path/"a", (0,4,8))
    plan = json.loads((a/"plan.json").read_text())
    plan['cohort'] = 'libero10-table4-s3407-v1'
    (a/"plan.json").write_bytes(canonical(plan))
    for t in range(10):
        query = f"libero10-task{t:02d}-init00-frame0080"
        append_record(a/"qualification.jsonl", dict(query_id=query,status='passed'))
        append_record(a/"qualification.jsonl", dict(query_id=query,dino_repeat_max_abs=0))
    # Synthetic fixture has two valid candidates, just as a retrieval pool may.
    report = plan_resume([a], 4)
    assert report['completed_initial_states'] == [0,4,8]
    assert report['episode_groups'] == [[1,6],[2,7],[3,9],[5]]
    assert report['new_source_episodes'] == 70
    assert report['checkpoint_sha256'] == 'a'*64
    with pytest.raises(ValueError, match='overlapping'):
        plan_resume([a,a], 4)
    (a/'qualification.jsonl').write_text('')
    with pytest.raises(ValueError, match='qualification'):
        plan_resume([a],4)


def test_fresh_plan_and_worker_validation():
    assert plan_resume([],4)['episode_groups'] == [[0,4,8],[1,5,9],[2,6],[3,7]]
    with pytest.raises(ValueError):
        plan_resume([],5)
