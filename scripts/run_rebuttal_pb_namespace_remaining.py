"""Finish one unstarted, fixed q96 using the original frozen physical collector.

The old factual/q8/q48 data, parent operator and absolute source deadline stay
unchanged. A separate platform-terminal gate prevents replay of an unknown owner.
"""
from pathlib import Path
import argparse,datetime,importlib.util,json,os,shutil,signal,subprocess,sys,tempfile,time,traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fastwam.research import pb_namespace_remaining as B


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--expected-plan-sha256',required=True);ap.add_argument('--validate-only',action='store_true');ap.add_argument('--audit-only',action='store_true');a=ap.parse_args()
    assert B.sha(a.plan)==a.expected_plan_sha256;p=B.validate_plan(B.read(a.plan))
    for row in p['inputs']:assert B.sha(row['path'])==row['sha256'],row['path']
    old=B.read(p['old_operator']['path']);runner=B.read(p['old_runner']['path']);oldfirst,oldend,stopped=B.require_old_terminal(old,runner)
    pool=B.read(p['pool_plan']['path']);base=B.read(p['base_q48_plan']['path']);B.require_q96_plan(base,pool)
    assert pool['namespace_case_id']==p['case_id'] and pool['query_replan']==96
    assert B.sha(p['collector'])==p['collector_sha256']
    out=Path(p['stage_root']);physical=Path(pool['output'])
    Hspec=importlib.util.spec_from_file_location('PB_remaining_archive',Path(p['frozen_controller'])/'scripts/run_rebuttal_original_PB_stage.py');H=importlib.util.module_from_spec(Hspec);Hspec.loader.exec_module(H)
    if a.validate_only:
        assert not a.audit_only;print(json.dumps(dict(status='unstarted_fixed_namespace_q96_CPU_validated',case_id=p['case_id'],source_policy_forwards=0,new_physical_commands=0,old_platform_gate_required_before_dispatch=True)));return
    if a.audit_only:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        from fastwam.research.pb_pool_audit import audit_pool,verify_archive
        archive=B.read(out/'pool_archive.json');bound=verify_archive(physical,archive);report=audit_pool(physical,query=96,plan_sha256=p['pool_plan']['sha256'],namespace_plan=pool)
        report.update(namespace_case_id=p['case_id'],physical_seed=pool['physical_seed'],archive=bound,source_policy_forwards=0,GPUs=0)
        assert not (out/'CPU_audit.json').exists();H.write(out/'CPU_audit.json',report);print(json.dumps(dict(status='fixed_q96_CPU_audit_verified',case_id=p['case_id'],GPUs=0)));return
    import fcntl
    B.validate_cloud_gate(p)
    lock=(out/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    operator=out/'operator.json';assert not operator.exists() and not (physical/'operator.json').exists(),'Never relaunch an active or unknown recovery'
    with Path(pool['native_worker_lease']).open('a') as lease:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert shutil.disk_usage(out).free>=p['minimum_admission_free_bytes']
    first=H.now();end=B.bounded_deadline(first,oldfirst,oldend,stopped);active_end=end-datetime.timedelta(seconds=p['shutdown_margin_seconds'])
    state=dict(status='admitted_unstarted_fixed_q96',case_id=p['case_id'],query=96,first_admitted_utc=first.isoformat(),absolute_deadline_utc=end.isoformat(),original_source_first_admitted_utc=oldfirst.isoformat(),original_source_absolute_deadline_utc=oldend.isoformat(),original_runner_stopped_utc=stopped.isoformat(),plan_sha256=a.expected_plan_sha256,source_policy_forwards=0,model_updates=0,automatic_restarts=0,old_parent_operator_unchanged=True,q48_interrupted_branches_never_retried=True)
    child=None;interrupted=False
    def save():state['updated_utc']=H.now().isoformat();H.write(operator,state)
    def stop(_signum,_frame):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);save()
    def stop_child():
        nonlocal child
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=90)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    def run(label,argv,*,CPU=False,maximum=None,limit=active_end):
        nonlocal child
        if interrupted or H.now()>=limit:raise TimeoutError('Original-window compute portion ended')
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        if CPU:env['CUDA_VISIBLE_DEVICES']=''
        if maximum is not None:limit=min(limit,H.now()+datetime.timedelta(seconds=maximum))
        with (out/(label+'_console.log')).open('x') as stream:
            child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
            state.update(status='running_'+label,actual_child_pid=child.pid);save();bounded=False
            while child.poll() is None:
                if interrupted or H.now()>=limit or shutil.disk_usage(out).free<p['runtime_free_space_guard_bytes']:
                    bounded=True;stop_child();break
                save();time.sleep(5)
            result=dict(exit_code=child.returncode,bounded_termination=bounded,ended_utc=H.now().isoformat(),actual_pid=child.pid,CUDA_visible_empty=CPU,argv=argv)
        H.write(out/(label+'_exit.json'),result);child=None;return result
    def archive():
        target=out/'pool_archive.json'
        if physical.exists() and not target.exists():H.write(target,H.archive(physical,out/'archives','q96_pool'))
    try:
        cache=Path(tempfile.mkdtemp(prefix=f"warm-pb-q96-case{p['case_id']}-"));os.environ.update(TMPDIR=str(cache),TORCH_EXTENSIONS_DIR=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),CUDA_CACHE_PATH=str(cache/'cuda'));state['node_local_private_cache']=str(cache);save()
        cmd=[p['python'],'-B',p['collector'],'--plan',p['pool_plan']['path'],'--expected-plan-sha256',p['pool_plan']['sha256']]
        if p.get('prepared_CPU_seal'):
            ref=p['prepared_CPU_seal'];assert B.sha(ref['path'])==ref['sha256']
            B.require_prepared_CPU_seal(B.read(ref['path']),p['pool_plan']['sha256'])
            seal=dict(exit_code=0,reused_prepared_CPU_seal=True,prepared_ref=ref,new_policy_generations=0,new_physical_commands=0)
            H.write(out/'q96_CPU_seal_exit.json',seal)
        else:seal=run('q96_CPU_seal',cmd+['--preflight-only'],CPU=True,maximum=900)
        if seal['exit_code']!=0:raise RuntimeError('Fixed q96 input seal failed;query not replaced')
        admission=out/'pool.admission.json';assert not admission.exists();H.write(admission,dict(schema='warm.original-PB-physical-pool-admission.v1',first_admitted_utc=first.isoformat(),absolute_deadline_utc=active_end.isoformat(),plan_sha256s={p['pool_plan']['path']:p['pool_plan']['sha256']},source_policy_forwards=0,original_windows_unchanged=True))
        result=run('q96_pool',cmd+['--admission',str(admission),'--expected-admission-sha256',B.sha(admission)])
        archive()
        if result['exit_code']!=0:raise RuntimeError('Fixed q96 interrupted;raw evidence retained without automatic retry')
        audit=run('q96_CPU_audit',[p['python'],'-B',str(Path(__file__).resolve()),'--plan',str(a.plan),'--expected-plan-sha256',a.expected_plan_sha256,'--audit-only'],CPU=True,limit=end-datetime.timedelta(seconds=15))
        if audit['exit_code']!=0:raise RuntimeError('q96 physical pool ended but CPU verification is not released')
        state.update(status='terminal_fixed_namespace_q96_archived_and_audited',ended_utc=H.now().isoformat(),pool_archive=B.read(out/'pool_archive.json'),CPU_audit_sha256=B.sha(out/'CPU_audit.json'));save()
    except BaseException as exc:
        stop_child();archive();state.update(status='failed_or_interrupted_fixed_q96_preserve_all',error=repr(exc),traceback=traceback.format_exc(),ended_utc=H.now().isoformat());save();raise
    finally:stop_child()


if __name__=='__main__':main()
