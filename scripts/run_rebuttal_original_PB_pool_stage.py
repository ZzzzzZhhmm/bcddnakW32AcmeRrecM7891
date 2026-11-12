"""Bounded original-PB physical pools after an independently completed CPU seal.

The stage never accesses Git or generates policy actions. Failed qualifications
remain archived while other fixed queries continue; no failed query is replaced.
"""
from pathlib import Path
import argparse
import datetime
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import time
import traceback

_path = Path(__file__).with_name('run_rebuttal_original_PB_stage.py')
_spec = importlib.util.spec_from_file_location('PB_archive_helpers', _path)
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def validate_plan(p):
    expected = dict(schema='warm.original-PB-physical-pool-stage.v1',
        queries=[8,48,96], phases=['qualify','pool'], maximum_seconds=5400,
        original_windows_unchanged=True, source_policy_forwards=0,
        qualification_failure_does_not_replace_fixed_query=True)
    if any(type(p.get(k)) is not type(v) or p.get(k) != v for k,v in expected.items()):
        raise ValueError('Fixed PB physical stage scope or resource limit differs')
    return p


def fixed_deadline(first, latest, maximum):
    if maximum != 5400 or first.tzinfo is None or latest.tzinfo is None:
        raise ValueError('A single timezone-aware physical admission is required')
    end = min(first + datetime.timedelta(seconds=maximum), latest)
    if (end-first).total_seconds() < 600:
        raise ValueError('Insufficient remaining time for safe physical admission')
    return end


def main():
    import fcntl
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--expected-plan-sha256', required=True)
    cli = ap.parse_args()
    assert H.sha(cli.plan) == cli.expected_plan_sha256
    p = validate_plan(H.read(cli.plan))
    root = Path(p['stage_root'])
    root.mkdir(exist_ok=True)
    lock = (root/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    op = root/'operator.json'
    assert not op.exists(), 'An existing stage is never automatically relaunched'
    state = dict(status='waiting_for_CPU_seals', started_utc=H.now().isoformat(),
        plan_sha256=cli.expected_plan_sha256, source_policy_forwards=0, phases={})
    child = None

    def save():
        state['updated_utc'] = H.now().isoformat()
        H.write(op,state)

    save()
    try:
        for row in p['inputs']:
            assert H.sha(row['path']) == row['sha256'], row['path']
        source_stage = H.read(p['source_stage_operator'])
        assert source_stage['status'] == 'complete_fixed_PB_fixture_raw_archived'
        assert H.read(p['source_trace_exit'])['exit_code'] == 0
        latest_admission = datetime.datetime.fromisoformat(p['latest_admission_utc'])
        cpu = Path(p['CPU_stage_root'])
        while True:
            if H.now() >= latest_admission:
                raise TimeoutError('PB physical admission expired; all old clocks remain unchanged')
            manifests = list((cpu/'job_logs').glob('*/run_manifest.json'))
            assert len(manifests) <= 1, 'Unknown or duplicate CPU producer'
            complete = cpu/'CPU_seal_complete.json'
            if complete.exists():
                assert H.read(complete)['all_completed'], 'CPU producer failed; preserve it'
                if manifests and H.read(manifests[0]).get('status') == 'complete' and H.read(manifests[0]).get('exit_code') == 0:
                    break
            save()
            time.sleep(5)
        for row in p['query_plans']:
            q = H.read(row['path'])
            proof = H.read(Path(q['output'])/'preflight.json')
            assert proof['status'] == 'complete_PB_CPU_input_seal'
            assert proof['plan_sha256'] == row['sha256'] and proof['CUDA_initialized'] is False
        gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                              '--format=csv,noheader'],capture_output=True,text=True)
        assert gpu.returncode == 0 and not gpu.stdout.strip(), 'GPU already owned'
        assert shutil.disk_usage(root).free > 4*1024**3
        with Path(p['physical_worker_lease']).open('a') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        first = H.now()
        end = fixed_deadline(first,datetime.datetime.fromisoformat(p['latest_end_utc']),p['maximum_seconds'])
        admission = dict(schema='warm.original-PB-physical-pool-admission.v1',
            first_admitted_utc=first.isoformat(), absolute_deadline_utc=end.isoformat(),
            plan_sha256s={r['path']:r['sha256'] for r in p['query_plans']},
            source_policy_forwards=0, original_windows_unchanged=True,
            source_stage_absolute_deadline_utc=source_stage['absolute_deadline_utc'])
        af = root/'admission.json'
        assert not af.exists()
        H.write(af,admission)
        state.update(status='physical_stage_admitted',first_admitted_utc=first.isoformat(),
                     absolute_deadline_utc=end.isoformat())
        save()
        qualified = set()
        for phase in p['phases']:
            for row in p['query_plans']:
                query = H.read(row['path'])
                q = query['query_replan']
                key = f'q{q}_{phase}'
                if phase == 'pool' and q not in qualified:
                    state['phases'][key] = dict(status='not_run_unqualified_fixed_query')
                    save()
                    continue
                if (end-H.now()).total_seconds() < 120:
                    state['phases'][key] = dict(status='not_run_fixed_window_insufficient')
                    save()
                    continue
                assert shutil.disk_usage(root).free > 4*1024**3
                argv = [p['python'],'-B',p['collector'],'--plan',row['path'],
                    '--expected-plan-sha256',row['sha256'],'--admission',str(af),
                    '--expected-admission-sha256',H.sha(af)]
                argv += ['--qualify-only'] if phase == 'qualify' else ['--resume-known-terminal']
                with (root/(key+'_console.log')).open('x') as log:
                    child = subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,
                        start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
                        PYTHONUNBUFFERED='1'))
                    state.update(status='running_'+key,actual_child_pid=child.pid,active_argv=argv)
                    save()
                    bounded = False
                    while child.poll() is None:
                        if H.now() >= end:
                            bounded = True
                            os.killpg(child.pid,signal.SIGINT)
                            try:
                                child.wait(timeout=120)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid,signal.SIGKILL)
                                child.wait()
                            break
                        save()
                        time.sleep(5)
                    receipt = dict(exit_code=child.returncode,bounded_termination=bounded,
                        actual_pid=child.pid,ended_utc=H.now().isoformat(),argv=argv)
                    H.write(root/(key+'_exit.json'),receipt)
                    child = None
                if Path(query['output']).is_dir():
                    receipt['archive'] = H.archive(query['output'],root/'archives',key)
                state['phases'][key] = receipt
                if phase == 'qualify' and receipt['exit_code'] == 0:
                    assert H.read(Path(query['output'])/'operator.json')['status'] == 'qualified_PB_query_AABA_pool_pending'
                    qualified.add(q)
                save()
        state.update(status='terminal_fixed_PB_stage_all_existing_output_archived',
            ended_utc=H.now().isoformat(),all_old_windows_unchanged=True)
        save()
    except BaseException as exc:
        state.update(status='failed_or_interrupted_preserve_all',error=repr(exc),
            traceback=traceback.format_exc(),ended_utc=H.now().isoformat())
        save()
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    main()
