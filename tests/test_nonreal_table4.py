import json
from pathlib import Path
import subprocess
import sys

import pytest

from fastwam.research.table4 import latex_rows, summarize_table4


def config():
    return dict(tasks=["press_button", "put_back_block"], bootstrap=100, seed=3407,
                candidate_metrics=dict(delta_obs=.001, delta_pred=.001,
                                       magnitude_weight=.25, top_tolerance=.001))


def rows():
    return [dict(task=task, episode_id=f"e{ep}", query_id=f"{task}-e{ep}",
                 checkpoint_sha256="a"*64, probe_metadata_sha256="b"*64,
                 cohort="synthetic_unit_test", required_effect=[1., 0.], candidates=[
                     dict(candidate_id="a", valid=True, predicted_effect=[1., 0.],
                          historical_effect=[-1., 0.], observed_effect=[1., 0.],
                          endpoint_status="complete_horizon", executed_steps=32, horizon=32,
                          applicable=1, label_provenance="synthetic_unit_test"),
                     dict(candidate_id="b", valid=True, predicted_effect=[-1., 0.],
                          historical_effect=[1., 0.], observed_effect=[-1., 0.],
                          endpoint_status="complete_horizon", executed_steps=32, horizon=32,
                          applicable=0, label_provenance="synthetic_unit_test")])
            for task in config()["tasks"] for ep in range(3)]


def test_table4_matched_labels_and_analytical_ties():
    evidence = rows()
    evidence[0]["candidates"][1]["applicable"] = None
    report = summarize_table4(evidence, config())
    assert report["status"] == "measured"
    assert report["coverage"]["unknown_labels"] == 1
    assert report["coverage"]["shared_queries/top_choice_applicability"] == 5
    for ref in ("historical", "query", "predicted"):
        assert report["per_query"][0][f"{ref}/top_choice_applicability"] is None
        assert report["aggregates"][f"{ref}/top_choice_applicability"]["missing_queries"] == 1
    assert report["aggregates"]["query/pairwise_agreement"]["mean"] == .5
    assert report["aggregates"]["query/predicted_tie_rate"]["mean"] == 1
    assert report["aggregates"]["predicted_minus_historical/top_choice_applicability"]["mean"] == 1
    assert "100.0\\%" in latex_rows(report)


@pytest.mark.parametrize("field,value", [("checkpoint_sha256", "f"*64), ("cohort", "other")])
def test_mixed_model_and_cohort_rejected(field, value):
    evidence = rows()
    evidence[0][field] = value
    with pytest.raises(ValueError, match="cannot mix"):
        summarize_table4(evidence, config())


def test_missing_branch_and_missing_task_do_not_export_latex():
    evidence = rows()
    evidence[0]["candidates"][0].update(endpoint_status="not_attempted", observed_effect=None,
                                        executed_steps=None, applicable=None)
    report = summarize_table4(evidence, config())
    assert report["status"] == "blocked"
    with pytest.raises(ValueError, match="incomplete"):
        latex_rows(report)
    report = summarize_table4(rows()[:3], config())
    assert report["status"] == "blocked"


def test_early_terminal_keeps_label_and_coverage():
    evidence = rows()
    evidence[0]["candidates"][0].update(endpoint_status="incomplete_horizon", observed_effect=None,
                                        executed_steps=9, termination="success")
    report = summarize_table4(evidence, config())
    assert report["status"] == "measured"
    assert report["coverage"]["complete_endpoints"] == 11
    assert report["coverage"]["known_labels"] == 12
    assert report["coverage"]["comparable_queries"] == 5


def test_unknown_labels_never_become_applicable_percentages():
    evidence = rows()
    for row in evidence:
        for c in row["candidates"]:
            c["applicable"] = None
    report = summarize_table4(evidence, config())
    assert report["status"] == "blocked"
    assert report["aggregates"]["predicted/effect_mse"]["mean"] == 0
    assert report["aggregates"]["predicted/top_choice_applicability"]["mean"] is None


def test_cli_does_not_overwrite_and_blocks_empty_evidence(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/report_nonreal_table4.py"
    source, settings, output = tmp_path/"rows.jsonl", tmp_path/"config.json", tmp_path/"output"
    source.write_text("\n".join(json.dumps(r) for r in rows()))
    settings.write_text(json.dumps(config()))
    command = [sys.executable, str(script), "--input", str(source), "--config", str(settings),
               "--output", str(output)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    assert (output/"table4_rows.tex").is_file()
    before = (output/"table4.json").read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert (output/"table4.json").read_bytes() == before
    source.write_text("")
    command[-1] = str(tmp_path/"empty")
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert not (tmp_path/"empty").exists()
