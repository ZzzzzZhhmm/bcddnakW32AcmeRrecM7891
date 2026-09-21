import json
from pathlib import Path
import numpy as np
import pytest

from fastwam.research.evidence import append_record, write_probe, probe_query_id


def test_query_identity_separates_tasks_and_interventions():
    kwargs = dict(experiment_id="nonreal-full", task="press_button", bound_step_sha256="a"*64)
    q = probe_query_id(**kwargs)
    assert probe_query_id(**kwargs) == q
    assert probe_query_id(**{**kwargs, "experiment_id": "nonreal-null"}) != q
    assert probe_query_id(**{**kwargs, "task": "put_back_block"}) != q


def test_probe_is_immutable_numeric_and_hash_bound(tmp_path):
    p = {"schema": "test", "arrays": {"mu": np.ones((1, 2, 32, 14), dtype=np.float32)}}
    r = write_probe(tmp_path, "episode-0-frame-0", p, {"checkpoint_sha256": "f"*64})
    assert len(r["array_sha256"]) == 64
    with np.load(Path(r["path"]) / "proposal.npz", allow_pickle=False) as arrays:
        assert arrays["mu"].shape == (1, 2, 32, 14)
    with pytest.raises(FileExistsError):
        write_probe(tmp_path, "episode-0-frame-0", p, {})
    with pytest.raises(ValueError):
        write_probe(tmp_path, "../unsafe", p, {})


def test_nonfinite_and_objects_are_rejected_before_writing(tmp_path):
    for value in (np.array([float("nan")]), np.array([{}], dtype=object)):
        with pytest.raises(ValueError):
            write_probe(tmp_path, "q", {"arrays": {"bad": value}}, {})
    assert not (tmp_path / "q").exists()


def test_append_records_are_complete_json_lines(tmp_path):
    p = tmp_path / "out.jsonl"
    append_record(p, {"attempt": 1})
    append_record(p, {"complete": False})
    assert [json.loads(x) for x in p.read_text().splitlines()] == [{"attempt": 1}, {"complete": False}]
