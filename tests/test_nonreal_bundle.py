import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/nonreal_bundle.py'
    spec = importlib.util.spec_from_file_location('nonreal_bundle_under_test', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def bundle(tmp_path, monkeypatch):
    m = module()
    plan = dict(output=str(tmp_path / 'out'), max_seconds=7200, source_sha256='source')
    path = tmp_path / 'plan.json'
    write(path, plan)
    b = m.Bundle(plan, {}, path)
    monkeypatch.setattr(m, 'source_identity', lambda _: dict(sha256='source'))
    return m, b


def test_success_without_artifact_is_rejected(tmp_path, monkeypatch):
    m, b = bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(m, 'run_logged', lambda *a, **kw: 0)
    with pytest.raises(ValueError, match='publish evidence'):
        b.stage('fake_success', ['fake'], b.output / 'data')
    assert not (b.output / 'fake_success.receipt.json').exists()
    assert m.read(b.output / 'bundle_summary.json')['paper_evidence_complete'] is False


@pytest.mark.parametrize('exit_code,optional', [(0, False), (1, True)])
def test_stages_reuse_verified_success_or_optional_failure(tmp_path, monkeypatch, exit_code, optional):
    m, b = bundle(tmp_path, monkeypatch)
    data = b.output / 'data'
    calls = []
    def run(spec, argv, output, **kw):
        calls.append(argv)
        write(data / 'artifact.json', {'value': 7})
        write(output / 'run_manifest.json', dict(status='complete' if exit_code == 0 else 'failed',
              exit_code=exit_code, source_sha256_after='source'))
        return exit_code
    monkeypatch.setattr(m, 'run_logged', run)
    assert b.stage('probe', ['fake'], data, optional=optional) == exit_code
    assert b.stage('probe', ['fake'], data, optional=optional) == exit_code
    assert len(calls) == 1
    write(data / 'artifact.json', {'value': 999})
    with pytest.raises(ValueError, match='evidence changed'):
        b.stage('probe', ['fake'], data, optional=optional)


def test_partial_output_never_overwritten(tmp_path, monkeypatch):
    m, b = bundle(tmp_path, monkeypatch)
    data = b.output / 'data'
    write(data / 'unfinished.json', {'partial': True})
    monkeypatch.setattr(m, 'run_logged', lambda *a, **kw: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='unfinished stage output preserved'):
        b.stage('probe', ['fake'], data)


def test_timeout_before_launch_preserves_report(tmp_path, monkeypatch):
    m, b = bundle(tmp_path, monkeypatch)
    b.deadline = 0
    monkeypatch.setattr(m, 'run_logged', lambda *a, **kw: pytest.fail('must not launch'))
    with pytest.raises(TimeoutError):
        b.stage('probe', ['fake'], b.output / 'data')


def full_fixture(tmp_path, m, nonzero=False):
    prefixes, rows = [], []
    for i in range(50):
        prefix = dict(task='press_button' if i < 25 else 'put_back_block', episode_index=i // 5, frame_index=i % 5)
        prefixes.append(prefix)
        query = f'prefix-{i:04d}'
        directory = tmp_path / 'probes' / (query + '-full')
        directory.mkdir(parents=True)
        gate = .3 if nonzero and i == 0 else 0.
        np.savez(directory / 'proposal.npz', alpha=np.array([.2]), g=np.array([gate]),
                 v_det=np.array([True]), source_noise_scale=np.array([1.]))
        write(directory / 'proposal.json', dict(force_null=False, source_mode='full'))
        np.savez(tmp_path / (query + '-source-actions.npz'), full=np.zeros((32, 14)), repeat=np.zeros((32, 14)))
        rows.append(dict(prefix, query_id=query, mode='full', nonzero_gate=bool(gate), mean_g=gate,
                         probe=dict(array_sha256=m.file_hash(directory / 'proposal.npz'), metadata_sha256=m.file_hash(directory / 'proposal.json'))))
    write(tmp_path / 'frozen_prefixes.json', prefixes)
    expected = dict(checkpoint_sha256='new-checkpoint', prefix_manifest_sha256=m.file_hash(tmp_path / 'frozen_prefixes.json'))
    write(tmp_path / 'manifest.json', expected)
    write(tmp_path / 'summary.json', dict(status='complete', queries=50, nonzero_gate_queries=int(nonzero)))
    (tmp_path / 'source_results.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    return expected


@pytest.mark.parametrize('nonzero', [False, True])
def test_gate_decision_uses_verified_arrays_and_all_queries(tmp_path, nonzero):
    m = module()
    expected = full_fixture(tmp_path, m, nonzero)
    result = m.full_gate_report(tmp_path, expected)
    assert result['queries'] == 50 and result['episodes'] == 10
    assert result['nonzero_g'] == int(nonzero)
    (tmp_path / 'probes/prefix-0000-full/proposal.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        m.full_gate_report(tmp_path, expected)


def test_gate_rejects_wrong_checkpoint_and_incomplete_rows(tmp_path):
    m = module()
    expected = full_fixture(tmp_path, m)
    with pytest.raises(ValueError, match='identity mismatch'):
        m.full_gate_report(tmp_path, dict(expected, checkpoint_sha256='old-smoke'))
    path = tmp_path / 'source_results.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match='all 50'):
        m.full_gate_report(tmp_path, expected)


def test_unfinished_training_never_restarts(tmp_path, monkeypatch):
    m, b = bundle(tmp_path, monkeypatch)
    parent = tmp_path / 'training_plan'
    write(parent / 'started.json', {'started': True})
    b.plan['resume_plan'] = str(parent / 'plan.json')
    monkeypatch.setattr(m, 'run_logged', lambda *a, **kw: pytest.fail('must not restart'))
    with pytest.raises(ValueError, match='do not restart'):
        b.execute()


def test_checkpoint_rejects_success_summary_with_failed_job(tmp_path):
    m = module()
    write(tmp_path / 'summary.json', dict(status='complete', exit_code=0, reason=None))
    write(tmp_path / 'job/run_manifest.json', dict(status='failed', exit_code=1))
    with pytest.raises(ValueError, match='process did not complete'):
        m.verified_target(dict(resume_plan=str(tmp_path / 'plan.json')), {})


@pytest.mark.parametrize('nonzero,controls', [(0, False), (1, True)])
def test_new_checkpoint_pipeline_collects_full_once_and_controls_conditionally(tmp_path, monkeypatch, nonzero, controls):
    m, b = bundle(tmp_path, monkeypatch)
    write(tmp_path / 'summary.json', dict(status='complete'))
    b.plan.update(resume_plan=str(tmp_path / 'plan.json'), asset_root='assets', dev_identity={})
    b.resume.update(config='new-config', config_sha256='config-hash')
    target = dict(weights='new-weights', checkpoint_sha256='new-hash', training_summary='summary')
    monkeypatch.setattr(m, 'verified_target', lambda *args: target)
    monkeypatch.setattr(m, 'full_gate_report', lambda *args: dict(nonzero_g=nonzero))
    stages = []
    def stage(name, argv, data, **kwargs):
        stages.append(name)
        if name == 'paired_source':
            write(data / 'summary.json', dict(status='complete', queries=50))
        return 0
    monkeypatch.setattr(b, 'stage', stage)
    b.execute()
    assert stages.count('source_full') == 1
    assert ('source_gaussian' in stages) == controls
    assert ('source_scale_only' in stages) == controls
    assert ('paired_source' in stages) == controls
    assert b.report['status'] == 'completed_available_stages'
    assert b.report['paper_evidence_complete'] is False


def test_legacy_crlf_shell_is_never_executed(tmp_path, monkeypatch):
    m = module()
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    # This reproduces the real failure: a valid entrypoint sources a CRLF helper.
    (scripts / 'warm_server_common.sh').write_bytes(b'#!/bin/bash\r\n\r\nfalse\r\n')
    (scripts / 'acp_nonreal_resume.sh').write_text('exit 127\n')
    (scripts / 'nonreal_resume.py').write_bytes(
        b'import argparse\r\np=argparse.ArgumentParser()\r\n'
        b'p.add_argument("command")\r\np.add_argument("--plan")\r\n'
        b'p.add_argument("--port")\r\np.parse_args()\r\n')
    plan = dict(resume_code=str(tmp_path), resume_plan=str(tmp_path / 'plan.json'))
    monkeypatch.setenv('MASTER_PORT', '29577')
    command = m.training_command(plan)
    assert command[0] == sys.executable and command[-1] == '29577'
    assert all(not token.endswith('.sh') for token in command)
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_preflight_executes_frozen_python_and_rejects_crlf_active_shell(tmp_path, monkeypatch):
    m = module()
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('acp_nonreal_bundle.sh', 'warm_server_common.sh'):
        (scripts / name).write_bytes(b'#!/bin/bash\ntrue\n')
    (scripts / 'nonreal_resume.py').write_text('import argparse\np=argparse.ArgumentParser()\np.add_argument("--plan")\np.parse_args()\n')
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    plan = dict(resume_code=str(tmp_path), resume_plan='plan.json')
    assert m.check_training_entrypoint(plan)['status'] == 'passed'
    (scripts / 'warm_server_common.sh').write_bytes(b'#!/bin/bash\r\n')
    with pytest.raises(ValueError, match='LF line endings'):
        m.check_training_entrypoint(plan)


def test_bundle_preflight_runs_actual_shell_dependency_chain(tmp_path):
    if sys.platform == 'win32':
        pytest.skip('Run actual Bash dependency-chain integration on CCI/Linux')
    m = module()
    scripts = tmp_path / 'code/scripts'
    scripts.mkdir(parents=True)
    original = Path(__file__).resolve().parents[1] / 'scripts'
    for name in ('acp_nonreal_bundle.sh', 'warm_server_common.sh'):
        (scripts / name).write_bytes((original / name).read_bytes())
    write(tmp_path / 'bundle_plan.json', {})
    (scripts / 'nonreal_bundle.py').write_text(
        'import os,sys\nassert "--validate" in sys.argv\n'
        'assert os.environ["HF_HUB_OFFLINE"] == "1"\n'
        'assert os.environ["DIFFSYNTH_MODEL_BASE_PATH"]\n'
        'assert os.environ["OMP_NUM_THREADS"] == "4"\nprint("shell_chain_passed")\n')
    env = dict(m.os.environ, WARM_PYTHON=sys.executable)
    result = subprocess.run(['bash', str(scripts / 'acp_nonreal_bundle.sh'), '--preflight'],
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'shell_chain_passed' in result.stdout
