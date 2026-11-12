"""Run only remaining original PB branches in a separately bounded ACP stage."""
from pathlib import Path
import argparse
import datetime
import importlib.util
import os
import shutil
import signal
import subprocess
import time
import traceback

_spec=importlib.util.spec_from_file_location('PB_archives',Path(__file__).with_name('run_rebuttal_original_PB_stage.py'))
H=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def validate_plan(p):
    expected=dict(schema='warm.original-PB-remaining-resource-stage.v1',
        original_queries=[8,48,96],maximum_seconds=28800,GPUs=1,
        original_windows_unchanged=True,source_policy_forwards=0,
        interrupted_branches_never_retried=True)
    if any(type(p.get(k)) is not type(v) or p.get(k)!=v for k,v in expected.items()):
        raise ValueError('Wrong remaining-only PB resource stage')
    rows=p['query_plans'];queries=[r['query'] for r in rows]
    if not queries or queries!=[q for q in [8,48,96] if q in queries]:
        raise ValueError('Eligible queries retain their original order without duplicates')
    omitted=p.get('omitted_query_accounting',{})
    if set(map(int,omitted))!=set([8,48,96])-set(queries):
        raise ValueError('Every omitted fixed query requires explicit accounting')
    if any(v.get('reason') not in ('original_AABA_not_complete','no_remaining_new_work') for v in omitted.values()):
        raise ValueError('Queries cannot be omitted because of their scientific outcomes')
    if Path(p['stage_root']).resolve()==Path(p['old_stage_root']).resolve():
        raise ValueError('The original resource stage must be preserved')
    return p


def main():
    import fcntl
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--expected-plan-sha256',required=True)
    cli=ap.parse_args();assert H.sha(cli.plan)==cli.expected_plan_sha256
    p=validate_plan(H.read(cli.plan));R=Path(p['stage_root']);R.mkdir(exist_ok=True)
    lock=(R/'operator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    op=R/'operator.json';assert not op.exists(),'Never relaunch an unknown continuation'
    state=dict(status='checking_terminal_inputs',started_utc=H.now().isoformat(),
        plan_sha256=cli.expected_plan_sha256,source_policy_forwards=0,queries={})
    child=None;interrupted=False

    def save():
        state['updated_utc']=H.now().isoformat();H.write(op,state)

    def stop(_signum,_frame):
        nonlocal interrupted
        interrupted=True

    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);save()
    try:
        for row in p['inputs']:
            assert H.sha(row['path'])==row['sha256'],row['path']
        old=H.read(Path(p['old_stage_root'])/'operator.json')
        assert old['status'] in ('terminal_fixed_PB_stage_all_existing_output_archived',
                                  'failed_or_interrupted_preserve_all')
        with Path(p['physical_worker_lease']).open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(R).free>4*1024**3
        first=H.now();end=min(first+datetime.timedelta(seconds=28800),
            datetime.datetime.fromisoformat(p['latest_end_utc']))
        assert (end-first).total_seconds()>=600
        af=R/'admission.json';assert not af.exists()
        H.write(af,dict(schema='warm.original-PB-physical-pool-admission.v1',
            first_admitted_utc=first.isoformat(),absolute_deadline_utc=end.isoformat(),
            plan_sha256s={r['path']:r['sha256'] for r in p['query_plans']},
            old_stage_first_admitted_utc=old['first_admitted_utc'],
            old_stage_absolute_deadline_utc=old['absolute_deadline_utc'],
            source_policy_forwards=0,original_windows_unchanged=True,
            new_work='Remaining original branches only; no resampling of old unknown attempts'))
        state.update(status='new_remaining_only_stage_admitted',
            first_admitted_utc=first.isoformat(),absolute_deadline_utc=end.isoformat());save()
        for item in p['query_plans']:
            q=item['query']
            assert H.read(item['path'])['query_replan']==q
            if interrupted or (end-H.now()).total_seconds()<120:
                state['queries'][str(q)]=dict(status='not_started_fixed_resource_limit');save();continue
            argv=[p['python'],'-B',p['collector'],'--plan',item['path'],
                '--expected-plan-sha256',item['sha256'],'--admission',str(af),
                '--expected-admission-sha256',H.sha(af),'--continuation-manifest',
                item['manifest'],'--expected-continuation-sha256',item['manifest_sha256']]
            with (R/f'q{q}_console.log').open('x') as log:
                child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
                state.update(status=f'running_remaining_q{q}',active_child_pid=child.pid);save()
                bounded=False
                while child.poll() is None:
                    if interrupted or H.now()>=end:
                        bounded=True;os.killpg(child.pid,signal.SIGINT)
                        try:child.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait()
                        break
                    save();time.sleep(5)
                receipt=dict(exit_code=child.returncode,bounded_termination=bounded,
                    actual_pid=child.pid,ended_utc=H.now().isoformat(),argv=argv)
                H.write(R/f'q{q}_exit.json',receipt);child=None
            output=Path(H.read(item['path'])['output'])
            if output.exists():receipt['archive']=H.archive(output,R/'archives',f'q{q}')
            state['queries'][str(q)]=receipt;save()
        state.update(status='terminal_fixed_PB_stage_all_existing_output_archived',
            ended_utc=H.now().isoformat(),policy_updates=0,old_phase_clock_reset=False);save()
    except BaseException as exc:
        state.update(status='failed_or_interrupted_preserve_all',error=repr(exc),
            traceback=traceback.format_exc(),ended_utc=H.now().isoformat());save();raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
