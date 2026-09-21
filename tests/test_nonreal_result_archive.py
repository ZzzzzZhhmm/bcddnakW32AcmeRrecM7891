import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("archive_results", Path(__file__).resolve().parents[1] / "scripts/archive_nonreal_results.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def payload():
    return dict(total_episodes=2, successes=1, success_episodes=[0], failure_episodes=[1], warm_online_episodes=[
        dict(episode_index=0, success=True, termination_reason="success", replan_count=1, replans=[{}]),
        dict(episode_index=1, success=False, termination_reason="max_steps", replan_count=0, replans=[])])


def test_counts_include_failures():
    assert module.validate_counts(payload()) == (1, 2)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "unknown", "wrong_outcome"])
def test_invalid_episode_evidence_is_rejected(fault):
    value = payload()
    if fault == "duplicate":
        value["failure_episodes"] = [0]
    elif fault == "missing":
        value["warm_online_episodes"].pop()
    elif fault == "unknown":
        value["warm_online_episodes"][1]["termination_reason"] = "exception"
    else:
        value["warm_online_episodes"][1]["success"] = True
    with pytest.raises(ValueError):
        module.validate_counts(value)


@pytest.mark.parametrize('tamper', [False, True])
def test_archive_checks_completion_hash_and_keeps_raw_bytes(tmp_path, tamper):
    run = tmp_path / 'inputs' / 'run'
    result_dir = run / 'results' / 'suite'
    result_dir.mkdir(parents=True)
    path = result_dir / 'result.json'
    value = payload()
    value.update(task_suite='suite', task_id=0, warm_online_header=dict(
        contract=dict(root_seed=1, warm_checkpoint_sha256='ckpt', git_commit='commit', bank_content_sha256='bank'),
        runtime_attestation=dict(model_loaded_checkpoint_sha256='ckpt'), comparison_kind='historical', source_policy='fixed'))
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    for name in ('evaluation_compatibility.json', 'online_contract.json', 'resolved_config.yaml'):
        (run / name).write_bytes(b'{}')
    summary = dict(completed_jobs=1, total_successes=1, total_episodes=2, overall_success_rate=.5, results=[dict(
        result_path=str(path), result_sha256=hashlib.sha256(raw).hexdigest(), successes=1, total_episodes=2,
        suite='suite', task_id=0, root_seed=1, duration_seconds=5,
        evaluation_compatibility_sha256=hashlib.sha256(b'{}').hexdigest())])
    batch = tmp_path / 'inputs' / 'batch'
    batch.mkdir()
    (batch / 'summary.json').write_text(json.dumps(summary))
    if tamper:
        path.write_bytes(raw + b' ')
    output = tmp_path / 'archive'
    report = module.archive([batch], tmp_path / 'inputs', output)
    assert report['status'] == ('issues_found' if tamper else 'counts_verified')
    assert report['groups'][0]['count_audit_passed'] is (not tamper)
    assert (output / 'files.json').is_file()
