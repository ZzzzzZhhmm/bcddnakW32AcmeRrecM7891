import importlib.util
from pathlib import Path
import numpy as np
import pytest

from fastwam.research.evidence import write_probe
from fastwam.research.statistics import candidate_metrics

path = Path(__file__).resolve().parents[1] / "scripts/assemble_nonreal_candidates.py"
spec = importlib.util.spec_from_file_location("nonreal_assembly", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def bundle(tmp_path):
    probe = {"source_mode": "full", "comparison": True, "force_null": False, "gate_threshold": .15,
             "arrays": {"valid": np.array([[True, True, False]]),
                        "candidate_v_det": np.array([[True, True, False]]),
                        "selected_index": np.array([1]), "alpha": np.array([.2]),
                        "g": np.array([.1]), "v_det": np.array([True]),
                        "adapted_actions": np.zeros((1, 3, 32, 14), dtype=np.float32),
                        "predicted_effect": np.array([[[1., 0.], [-1., 0.], [0., 0.]]]),
                        "historical_effect": np.array([[[-1., 0.], [1., 0.], [0., 0.]]]),
                        "required_effect": np.array([[1., 0.]])}}
    reference = write_probe(tmp_path, "q0", probe, {"checkpoint_sha256": "a"*64})
    index = [dict(task="press_button", episode_id="episode0", query_id="q0", cohort="natural",
                  probe_path=reference["path"], probe_metadata_sha256=reference["metadata_sha256"])]
    attempt = dict(query_id="q0", candidate_id="rank-0", proposal_sha256=reference["metadata_sha256"], kind="branch_attempt")
    result = {**attempt, "kind": "branch_result", "parent_restored": True, "horizon": 32,
              "executed_steps": 32, "endpoint_status": "complete_horizon", "observed_effect": [1., 0.]}
    return index, [attempt, result]


def test_missing_branches_and_labels_remain_unknown_and_analyzable(tmp_path):
    index, branches = bundle(tmp_path)
    rows = module.assemble(index, branches, [])
    assert rows[0]["candidates"][1]["endpoint_status"] == "not_attempted"
    assert rows[0]["candidates"][1]["executed_steps"] is None
    assert rows[0]["candidates"][0]["applicable"] is None
    result = candidate_metrics(rows[0], delta_obs=.001, delta_pred=.001, magnitude_weight=.25, top_tolerance=.001)
    assert result["endpoint_count"] == 1 and result["known_label_count"] == 0


def test_rejects_branch_from_different_proposal_or_unrestored_parent(tmp_path):
    index, branches = bundle(tmp_path)
    branches[-1]["proposal_sha256"] = "f"*64
    with pytest.raises(ValueError, match="different sealed"):
        module.assemble(index, branches, [])
    branches[-1]["proposal_sha256"] = branches[0]["proposal_sha256"]
    branches[-1]["parent_restored"] = False
    with pytest.raises(ValueError, match="restoration"):
        module.assemble(index, branches, [])


def test_gate_join_uses_natural_selection_and_independent_label(tmp_path):
    from fastwam.research.statistics import gate_rates
    index, branches = bundle(tmp_path)
    labels = [dict(query_id="q0", candidate_id="rank-1", applicable=0, label_provenance="blind:v1")]
    row = module.assemble(index, branches, labels, kind="gates")[0]
    assert row["selected_candidate_id"] == "rank-1"
    assert row["known_candidate_labels"] == 1
    assert not row["verified_all_inapplicable"]
    assert gate_rates([row])["false_acceptance"] == 1
    assert gate_rates([row])["counts"][0]["mean_g"] == .1


def test_rejects_intervened_gate_even_with_matching_metadata_hash(tmp_path):
    import hashlib
    import json
    index, _ = bundle(tmp_path)
    meta_path = Path(index[0]["probe_path"]) / "proposal.json"
    metadata = json.loads(meta_path.read_bytes())
    metadata["force_null"] = True
    raw = module.canonical(metadata)
    meta_path.write_bytes(raw)
    index[0]["probe_metadata_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="unmodified"):
        module.assemble(index, [], [], kind="gates")
