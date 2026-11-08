"""Synthetic fixtures test estimands and failure handling, not paper results."""
from copy import deepcopy

import pytest

from fastwam.research.recovery import summarize_recovery


def record(ep, query, raw, refined, null, *, task="a", gate=.4):
    return dict(schema="warm.recovery.query.v1", task=task, episode_id=ep,
                query_id=query, cohort="synthetic unit fixture", protocol_sha256="a"*64,
                checkpoint_sha256="b"*64, horizon=32, gate=gate, gate_intervention="natural",
                branches={role: dict(applicable=value, label_provenance="fixture predicate")
                          for role, value in zip(("raw", "refined", "candidate_null"), (raw, refined, null))})


def test_repairs_harms_and_null_increment_are_different_estimands():
    rows=[record("e1","q1",0,1,1),record("e1","q2",1,0,0),
          record("e2","q3",0,1,0),record("e2","q4",1,1,1)]
    r=summarize_recovery(rows,tasks=["a"],bootstrap=100)
    rates=r["tasks"]["a"]["rates"]
    assert rates==dict(repair_rate=1.,harm_rate=.5,net_applicability_change=.25,
                       refined_minus_null_applicability=.25)
    assert r["tasks"]["a"]["source_episodes"]==2


def test_unknown_is_not_zero_and_all_metrics_share_the_same_support():
    rows=[record("e1","q1",0,1,None),record("e2","q2",0,0,0)]
    r=summarize_recovery(rows,tasks=["a"],bootstrap=100)
    c=r["tasks"]["a"]["counts"]
    assert c["queries"]==2 and c["complete_queries"]==1
    assert c["unknown_by_role"]["candidate_null"]==1
    assert r["macro"]["repair_rate"]["mean"]==0
    assert r["macro"]["repair_rate"]["ci95"] is None
    assert r["macro"]["repair_rate"]["undefined_bootstrap_replicates"]>0


def test_fixed_tasks_are_macro_averaged_not_pooled_and_never_silently_dropped():
    rows=[record("e1",str(i),0,1,0) for i in range(10)]
    rows += [record("e2","q",0,1,0),record("e1","q",0,0,0,task="b")]
    r=summarize_recovery(rows,tasks=["a","b"],bootstrap=40)
    assert r["macro"]["repair_rate"]["mean"]==.5
    assert r["macro"]["repair_rate"]["ci95"] is None
    r=summarize_recovery(rows,tasks=["a","b","c"],bootstrap=40)
    assert r["macro"]["repair_rate"]["mean"] is None
    assert r["tasks"]["c"]["counts"]["queries"]==0


def test_queries_from_one_episode_do_not_become_independent_clusters():
    rows=[record("e1",str(i),i%2,1,0) for i in range(30)]
    r=summarize_recovery(rows,tasks=["a"],bootstrap=50)
    assert r["macro"]["net_applicability_change"]["ci95"] is None
    assert "two source episodes" in r["macro"]["net_applicability_change"]["missing_reason"]


def test_zero_gate_and_positive_gate_repairs_are_reported_separately():
    rows=[record("e1","q",0,1,1,gate=0),record("e2","q",0,0,1,gate=.2)]
    r=summarize_recovery(rows,tasks=["a"],bootstrap=20)
    strata=r["tasks"]["a"]["natural_gate_strata"]
    assert strata["zero_gate"]["rates"]["repair_rate"]==1
    assert strata["positive_gate"]["rates"]["repair_rate"]==0
    assert strata["zero_gate"]["rates"]["refined_minus_null_applicability"]==0


@pytest.mark.parametrize("mutate,match",[
    (lambda r:r.update(gate_intervention="forced_zero"),"forced gate"),
    (lambda r:r.update(checkpoint_sha256="not-a-hash"),"checkpoint_sha256"),
    (lambda r:r["branches"]["raw"].update(applicable=True),"0, 1 or null"),
    (lambda r:r["branches"]["raw"].update(label_provenance=""),"provenance"),
    (lambda r:r.update(gate=float("nan")),"finite natural gate"),
])
def test_invalid_or_unsupported_evidence_is_rejected(mutate,match):
    r=record("e","q",0,1,0);mutate(r)
    with pytest.raises(ValueError,match=match):summarize_recovery([r],tasks=["a"],bootstrap=10)


def test_mixed_identity_and_duplicate_queries_are_rejected():
    a=record("e1","q",0,1,0);b=record("e2","q",0,1,0)
    b["cohort"]="another cohort"
    with pytest.raises(ValueError,match="do not mix"):
        summarize_recovery([a,b],tasks=["a"],bootstrap=10)
    with pytest.raises(ValueError,match="duplicate"):
        summarize_recovery([a,deepcopy(a)],tasks=["a"],bootstrap=10)
