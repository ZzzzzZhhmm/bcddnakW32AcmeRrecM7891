from copy import deepcopy

import numpy as np
import pytest

from fastwam.research.statistics import candidate_metrics, clustered_macro, gate_rates, gate_clustered, donor_coverage, paired_success


KW = dict(delta_obs=1e-6, delta_pred=1e-6, magnitude_weight=0.25, top_tolerance=1e-6)


def query():
    return {"required_effect": [1., 0.], "candidates": [
        dict(candidate_id="a", valid=True, predicted_effect=[1., 0.], historical_effect=[-1., 0.],
             observed_effect=[1., 0.], endpoint_status="complete_horizon", executed_steps=32, horizon=32,
             applicable=1, label_provenance="task_predicate:v1"),
        dict(candidate_id="b", valid=True, predicted_effect=[-1., 0.], historical_effect=[1., 0.],
             observed_effect=[-1., 0.], endpoint_status="complete_horizon", executed_steps=32, horizon=32,
             applicable=0, label_provenance="task_predicate:v1")
    ]}


def test_query_constant_is_half_and_all_tied_applicability_is_pool_rate():
    r = candidate_metrics(query(), **KW)
    assert r["references"]["query"]["pairwise_agreement"] == 0.5
    assert r["references"]["query"]["predicted_tie_rate"] == 1
    assert r["references"]["query"]["top_choice_applicability"] == 0.5
    assert r["predicted_minus_historical/pairwise_agreement"] == 1
    assert r["references"]["predicted"]["effect_mse"] == 0


def test_incomplete_endpoints_never_pad_and_labels_still_count():
    q = query()
    q["candidates"][0].update(endpoint_status="early_success", executed_steps=17, observed_effect=None)
    r = candidate_metrics(q, **KW)
    assert r["endpoint_count"] == 1
    assert r["references"]["predicted"]["pairwise_agreement"] is None
    assert r["references"]["predicted"]["top_choice_applicability"] == 1
    q["candidates"][0]["observed_effect"] = [1., 0.]
    with pytest.raises(ValueError, match="imputed"):
        candidate_metrics(q, **KW)


def test_unknown_label_in_tie_set_removes_only_that_metric_common_support():
    q = query()
    q["candidates"][1]["applicable"] = None
    r = candidate_metrics(q, **KW)
    assert r["references"]["query"]["top_choice_applicability"] is None
    assert r["predicted_minus_query/top_choice_applicability"] is None
    assert r["predicted_minus_query/pairwise_agreement"] == .5


def test_clustering_does_not_weight_episodes_by_query_count():
    rows = [dict(task="a", episode_id="x", query_id=str(i), score=1) for i in range(100)]
    rows += [dict(task="a", episode_id="y", query_id="0", score=0)]
    out = clustered_macro(rows, "score", tasks=["a"], bootstrap=200, seed=7)
    assert out["mean"] == .5
    assert out == clustered_macro(rows, "score", tasks=["a"], bootstrap=200, seed=7)
    assert clustered_macro(rows, "score", tasks=["a", "b"])["mean"] is None
    with pytest.raises(ValueError, match="duplicate"):
        clustered_macro(rows + [rows[0]], "score", tasks=["a"])


def episodes():
    return [dict(task="a", pairing_id=str(i), condition=c, run_id=c, checkpoint_sha256="a"*64,
                 reset_seed=i, eval_run=0, valid_trial=True, success=bool(v))
            for i in range(4) for c, v in (("normal_full", 1), ("donor_full", i%2),
                                         ("normal_no_cmp", 1), ("donor_no_cmp", 0))]


CONTRAST = {"gamma": {"normal_no_cmp": 1, "donor_no_cmp": -1, "normal_full": -1, "donor_full": 1}}


def test_paired_gamma_is_a_joint_contrast_and_partial_pairs_rejected():
    out = paired_success(episodes(), CONTRAST, tasks=["a"], bootstrap=100)
    assert out["contrasts"]["gamma"]["mean"] == .5
    assert out["evaluation_run_summary"]["normal_full"]["sample_std"] is None
    with pytest.raises(ValueError, match="incomplete"):
        paired_success(episodes()[:-1], CONTRAST, tasks=["a"])
    bad = episodes()
    bad[0]["reset_seed"] = 500
    with pytest.raises(ValueError, match="disagree"):
        paired_success(bad, CONTRAST, tasks=["a"])


def test_gate_eligibility_excludes_alpha_and_unknown_is_not_negative():
    rows = [dict(alpha=.9, threshold=.15, v_det=False, applicable=0, label_provenance="blind:r1"),
            dict(alpha=.1, threshold=.15, v_det=True, applicable=0, label_provenance="blind:r1"),
            dict(alpha=.9, threshold=.15, v_det=True, applicable=None)]
    r = gate_rates(rows)
    assert r["false_acceptance"] == 0
    assert r["true_acceptance"] is None
    assert r["counts"][0]["raw_above_threshold"] == 1
    assert r["unknown_labels"] == 1
    assert r["counts"]["unknown"]["eligible"] == 1
    assert r["counts"]["unknown"]["accepted"] == 1
    assert r["counts"][0]["hard_veto"] == 1
    rows[0]["intervention"] = "forced_null"
    with pytest.raises(ValueError, match="forced"):
        gate_rates(rows)


def test_empty_gate_query_is_not_unknown_candidate_or_all_inapplicable():
    row = dict(alpha=0., g=0., threshold=.15, v_det=False, applicable=None, empty_pool=True)
    result = gate_rates([row])
    assert result["empty_queries"] == 1
    assert result["unknown_labels"] == 0
    assert result["false_acceptance"] is None
    with pytest.raises(ValueError, match="empty pool"):
        gate_rates([{**row, "v_det": True}])


def test_nonfinite_unknown_and_false_completion_are_rejected():
    for field, value in (("observed_effect", [float("nan"), 0]), ("executed_steps", 31)):
        q = query()
        q["candidates"][0][field] = value
        with pytest.raises(ValueError):
            candidate_metrics(q, **KW)


def test_gate_bootstrap_is_clustered_and_sparse_class_ci_is_unknown():
    rows = [dict(task="a", episode_id=str(i), query_id="q", applicable=0,
                 alpha=.1, threshold=.15, v_det=True, label_provenance="predicate:v1") for i in range(4)]
    result = gate_clustered(rows, tasks=["a"], bootstrap=100)
    assert result["macro"]["false_acceptance"]["ci95"] == [0., 0.]
    assert result["macro"]["true_acceptance"]["mean"] is None
    rows[1]["applicable"] = rows[2]["applicable"] = rows[3]["applicable"] = None
    sparse = gate_clustered(rows, tasks=["a"], bootstrap=100)
    assert sparse["macro"]["false_acceptance"]["ci95"] is None
    assert sparse["macro"]["false_acceptance"]["undefined_bootstrap_replicates"] > 0


def test_donor_coverage_counts_actual_slots_and_keeps_cohorts_separate():
    row = dict(task="a", condition="random", episode_id="e", query_id="q", cohort="new",
               protocol_sha256="a"*64, valid_slots=5, replaced_slots=2,
               eligible_replaced_slots=0, missing_donor_slots=0)
    result = donor_coverage([row])["groups"][0]
    assert result["replacement_rate"] == .4
    assert result["eligible_replacement_rate"] == 0
    with pytest.raises(ValueError, match="inconsistent"):
        donor_coverage([{**row, "eligible_replaced_slots": 3}])


def test_repeated_reset_is_not_counted_as_independent_cluster():
    rows = episodes()
    for r in rows:
        r["reset_seed"] = 17
    result = paired_success(rows, CONTRAST, tasks=["a"], bootstrap=100)
    assert result["contrasts"]["gamma"]["episode_counts"] == {"a": 1}
    assert result["contrasts"]["gamma"]["ci95"] is None


def test_five_run_sample_std_uses_task_macro_then_ddof_one():
    rows = []
    for run in range(5):
        for row in episodes():
            r = {**row, "pairing_id": f"{run}-{row['pairing_id']}", "eval_run": run}
            if r["condition"] == "donor_full":
                r["success"] = int(row["pairing_id"]) < run
            rows.append(r)
    out = paired_success(rows, CONTRAST, tasks=["a"], bootstrap=100)
    summary = out["evaluation_run_summary"]["donor_full"]
    assert summary["evaluation_runs"] == 5
    assert summary["mean"] == .5
    assert summary["sample_std"] == pytest.approx(np.std([0, .25, .5, .75, 1], ddof=1))
