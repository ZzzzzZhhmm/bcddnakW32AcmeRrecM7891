import importlib.util
import json
from pathlib import Path
import sys

import pytest

path = Path(__file__).resolve().parents[1] / "scripts/nonreal_job.py"
spec = importlib.util.spec_from_file_location("nonreal_job", path)
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


def recipe(command):
    return dict(schema="warm.nonreal.job.v1", readiness="ready", max_seconds=10,
                argv=command, evidence_type="test", claim="runner verification")


def test_prepare_blocks_unqualified_jobs_and_wrong_input_hash(tmp_path):
    blocked = recipe(["{python}", "-V"])
    blocked["readiness"] = "blocked"
    with pytest.raises(ValueError, match="not ready"):
        job.prepare(blocked, root=tmp_path, python=sys.executable)
    bad = recipe(["{python}", "-V"])
    p = tmp_path / "input.json"
    p.write_text("{}")
    bad["inputs"] = [{"path": str(p), "sha256": "0"*64}]
    with pytest.raises(ValueError, match="identity"):
        job.prepare(bad, root=tmp_path, python=sys.executable)


def test_prepare_preserves_literal_code_and_rejects_string_argv(tmp_path):
    command = ["{python}", "-c", "print({'status': 'ok'})"]
    assert job.prepare(recipe(command), root=tmp_path, python=sys.executable)[2] == command[2]
    with pytest.raises(ValueError, match="list of strings"):
        job.prepare(recipe("python -V"), root=tmp_path, python=sys.executable)


@pytest.mark.parametrize("code", [0, 7])
def test_runner_preserves_exit_code_logs_and_provenance(tmp_path, code):
    command = [sys.executable, "-c", f"print('traceable'); raise SystemExit({code})"]
    source = job.source_identity(job.ROOT)
    directory = tmp_path / "run"
    result = job.run(recipe(command), command, directory, source=source)
    assert result == code
    assert b"traceable" in (directory / "console.log").read_bytes()
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["exit_code"] == code
    assert manifest["status"] == ("complete" if code == 0 else "failed")
    assert manifest["source_sha256"] == manifest["source_sha256_after"]
    assert json.loads((directory / "events.jsonl").read_text().splitlines()[-1])["kind"] == "exit"


def test_wall_time_limit_is_not_reported_as_success(tmp_path):
    command = [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(20)"]
    config = recipe(command)
    config["max_seconds"] = 1
    directory = tmp_path / "deadline"
    assert job.run(config, command, directory, source=job.source_identity(job.ROOT)) != 0
    result = json.loads((directory / "run_manifest.json").read_text())
    assert result["status"] == "failed"
    assert result["reason"] == "wall_time_limit"
