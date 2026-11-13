"""Explicitly recover two certified pre-branch layout errors; never reset their clock."""
from pathlib import Path
import argparse,datetime,fcntl,importlib.util,json,os,shutil,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fastwam.research import pb_namespace_null_recovery as B

def load(path):
    spec=importlib.util.spec_from_file_location('fixed_null_recovery_base',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def validate(p):
    assert p['schema']==B.SCHEMA and p['case_ids']==[0,1] and p['new_GPU_hours_reserved']==0 and p['automatic_restarts']==0
    assert p['cumulative_reserved_GPU_hours']==143.8054115725 and p['lifetime_cap_GPU_hours']==144
    for r in p['inputs']:B.reference(r)
    original=B.read(B.reference(p['original_plan']));base=load(B.reference(p['patched_runner']));base.BIND.validate_plan(original)
    parent=B.read(B.reference(p['old_parent']));runner=B.read(B.reference(p['old_runner']))
    cases={r['case_id']:B.read(B.reference(r['operator'])) for r in p['failed_cases']}
    first,end=B.validate_failures(parent,runner,cases,base.now())
    assert p['original_first_admitted_utc']==parent['first_admitted_utc'] and p['absolute_deadline_utc']==parent['absolute_deadline_utc']
    for row in p['failed_cases']:
        case=row['case_id'];oldout=Path(original['cases'][case]['output']);manifest=B.read(B.reference(row['manifest']))
        assert B.sha(B.reference(row['archive']))==parent['cases'][str(case)]['archive']['sha256']
        assert len(manifest)==13 and all((oldout/n).is_file() and B.sha(oldout/n)==v['sha256'] for n,v in manifest.items())
        for q in range(9):assert B.read(oldout/f'replay-q{q:04d}.json')['original_H32_exact'] is True
        assert not (oldout/'action_seal.json').exists()
    return base,original,first,end

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--expected-plan-sha256',required=True);ap.add_argument('--validate-only',action='store_true');ap.add_argument('--case',type=int,choices=[0,1]);a=ap.parse_args()
    assert B.sha(a.plan)==a.expected_plan_sha256;p=B.read(a.plan);base,original,first,end=validate(p);root=Path(p['root']);new=B.execution_plan(original,root)
    if a.validate_only:print(json.dumps(dict(status='known_prebranch_failures_CPU_validated',cases=[0,1],new_policy_queries=0,new_physical_commands=0,original_deadline_unchanged=True)));return
    if a.case is not None:
        admission=B.read(root/f'case{a.case:02d}.admission.json');assert admission['original_first_admitted_utc']==first.isoformat() and datetime.datetime.fromisoformat(admission['absolute_deadline_utc'])<=end-datetime.timedelta(seconds=180)
        base.run_case(new,a.case,admission);return
    locks=[]
    for f in [Path(original['root'])/'supervisor.lock',root/'supervisor.lock']:
        lock=f.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    op=root/'operator.json';assert not op.exists() and shutil.disk_usage(root).free>=4*1024**3
    state=dict(status='admitted_known_prebranch_layout_recovery',first_admitted_utc=first.isoformat(),recovery_admitted_utc=base.now().isoformat(),absolute_deadline_utc=end.isoformat(),case_ids=[0,1],cases={},original_windows_unchanged=True,automatic_restarts=0,old_prebranch_failures_preserved=True,extra_reserved_GPU_hours=0)
    def save():state['updated_utc']=base.now().isoformat();base.write(op,state)
    child=None;interrupted=False
    def stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);save()
    def stop_child():
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=90)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    try:
        for case in [0,1]:
            if interrupted:raise InterruptedError('Preserve interrupted recovery without retry')
            try:deadline=base.BIND.case_deadline(base.now(),end)
            except ValueError:state['cases'][str(case)]=dict(status='not_started_original_window_insufficient');save();continue
            admission=dict(original_first_admitted_utc=first.isoformat(),absolute_deadline_utc=deadline.isoformat(),original_windows_unchanged=True)
            base.write(root/f'case{case:02d}.admission.json',admission);bounded=False
            with (root/f'case{case:02d}.console.log').open('x') as stream:
                child=subprocess.Popen([p['python'],'-B',str(Path(__file__).resolve()),'--plan',str(a.plan),'--expected-plan-sha256',a.expected_plan_sha256,'--case',str(case)],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                state.update(status=f'running_fixed_case{case}',actual_child_pid=child.pid);save()
                while child.poll() is None:
                    if interrupted or base.now()>=deadline or shutil.disk_usage(root).free<512*1024**2:bounded=True;stop_child();break
                    save();time.sleep(5)
            result=dict(exit_code=child.returncode,bounded_termination=bounded,ended_utc=base.now().isoformat());child=None
            out=Path(new['cases'][case]['output'])
            if out.is_dir():result['archive']=base.ARCH.archive(out,root/'archives',f'case{case:02d}')
            state['cases'][str(case)]=result;save()
        state.update(status='terminal_fixed_recovery_all_existing_output_archived',ended_utc=base.now().isoformat());save()
    except BaseException as exc:
        stop_child();state.update(status='failed_or_interrupted_recovery_preserve_all',error=repr(exc),traceback=traceback.format_exc(),ended_utc=base.now().isoformat());save();raise
    finally:stop_child()

if __name__=='__main__':main()
