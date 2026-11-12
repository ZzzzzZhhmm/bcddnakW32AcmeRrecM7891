"""Run one prospectively bound PB namespace source and its three fixed pools.

Each ACP allocation owns a private simulator/cache/lease. Old physical0 studies
and their clocks are untouched. Failed, censored and unreached queries remain
in the records; no learned-policy outcome can select another source or query.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,os,shutil,signal,subprocess,tempfile,time,traceback

ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
N=load('PB_namespace_stage_binding',ROOT/'src/fastwam/research/pb_namespace.py')
H=load('PB_namespace_archive',ROOT/'scripts/run_rebuttal_original_PB_stage.py')


def validate_plan(plan):
    expected=dict(schema='warm.PB-namespace2-source-stage.v1',queries=[8,48,96],
        maximum_seconds=36000,qualification_maximum_seconds=1800,factual_maximum_seconds=7200,
        source_checkpoint_step=12000,original_windows_unchanged=True,automatic_restarts=0,
        recorded_prefix_TOPP_policy='exact_original_executor_and_error_history_twice',
        minimum_admission_free_bytes=4*1024**3,runtime_free_space_guard_bytes=512*1024**2)
    if any(type(plan.get(k)) is not type(v) or plan.get(k)!=v for k,v in expected.items()):
        raise ValueError('Prospective fixed namespace scope,resource window or storage guard differs')
    qualification=H.read(plan['qualification_plan']);registration,entry=N.validate_execution_binding(qualification,'qualification')
    if (plan['case_id']!=entry['case_id'] or plan['physical_seed']!=entry['seed'] or type(plan['case_id']) is not int
            or Path(plan['stage_root'])!=Path(registration['phase_root'])/f"case{entry['case_id']:02d}"):
        raise ValueError('Source stage is not the assigned expert-roster case')
    if plan['latest_end_utc']!=registration['latest_end_utc']:raise ValueError('Hard writing deadline changed')
    return plan


def fixed_deadline(first,latest):
    if first.tzinfo is None or latest.tzinfo is None:raise ValueError('Timezone-aware admission is required')
    end=min(first+datetime.timedelta(seconds=36000),latest)
    if (end-first).total_seconds()<1800:raise ValueError('Insufficient time before the original hard end')
    return end


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--expected-plan-sha256',required=True);parser.add_argument('--validate-only',action='store_true');cli=parser.parse_args()
    assert H.sha(cli.plan)==cli.expected_plan_sha256;p=validate_plan(H.read(cli.plan));root=Path(p['stage_root'])
    for row in p['frozen_inputs']:assert H.sha(row['path'])==row['sha256'],row['path']
    if cli.validate_only:
        print(json.dumps(dict(status='namespace_source_stage_CPU_validated',case=p['case_id'],physical_seed=p['physical_seed'],model_queries=0,physical_commands=0)));return
    import fcntl
    lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    op=root/'operator.json';assert not op.exists(),'Existing or unknown dispatch must be adjudicated;no automatic restart'
    assert shutil.disk_usage(root).free>=p['minimum_admission_free_bytes']
    first=H.now();end=fixed_deadline(first,datetime.datetime.fromisoformat(p['latest_end_utc']))
    qualification=H.read(p['qualification_plan']);_,entry=N.bind_case(qualification)
    state=dict(status='admitted_bound_namespace_source',started_utc=first.isoformat(),first_admitted_utc=first.isoformat(),absolute_deadline_utc=end.isoformat(),plan_sha256=cli.expected_plan_sha256,namespace_case_id=p['case_id'],physical_seed=p['physical_seed'],qualification_started=False,trace_started=False,queries={},original_windows_unchanged=True,automatic_restarts=0,scope='Two fixed independent original-PB namespace sources;queries within one source are dependent,not population repair or benchmark reproduction.')
    child=None
    def save():state['updated_utc']=H.now().isoformat();H.write(op,state)
    def budget():
        if H.now()>=end:raise TimeoutError('Namespace source hard window ended;do not reset it')
        if shutil.disk_usage(root).free<p['runtime_free_space_guard_bytes']:raise OSError('AFS runtime guard;preserve all partial evidence')
    def run(label,argv,maximum,*,CPU=False):
        nonlocal child
        budget();target=root/(label+'_exit.json');assert not target.exists()
        limit=min(end,H.now()+datetime.timedelta(seconds=maximum));env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        if CPU:env['CUDA_VISIBLE_DEVICES']=''
        with (root/(label+'_console.log')).open('x') as stream:
            child=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running_'+label,actual_child_pid=child.pid,active_argv=argv);save();bounded=False
            while child.poll() is None:
                if H.now()>=limit or shutil.disk_usage(root).free<p['runtime_free_space_guard_bytes']:
                    bounded=True;os.killpg(child.pid,signal.SIGINT)
                    try:child.wait(timeout=120)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                save();time.sleep(5)
            result=dict(exit_code=child.returncode,bounded_termination=bounded,actual_pid=child.pid,ended_utc=H.now().isoformat(),argv=argv,CUDA_visible_empty=CPU)
        H.write(target,result);child=None;return result
    def archive_if_present(path,label):
        return H.archive(Path(path),root/'archives',label) if Path(path).is_dir() else dict(output_created=False)
    def immutable(path,value):
        assert not path.exists();H.write(path,value);return dict(path=str(path),sha256=H.sha(path))
    def command(entry,plan):return [p['python'],'-B',str(ROOT/'scripts'/entry),'--plan',str(plan),'--expected-plan-sha256',H.sha(plan)]
    save()
    try:
        cache=Path(tempfile.mkdtemp(prefix=f"warm-pb-ns2-case{p['case_id']}-"));os.environ.update(TMPDIR=str(cache),TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'),TRITON_CACHE_DIR=str(cache/'triton'),CUDA_CACHE_PATH=str(cache/'cuda'))
        state['node_local_private_cache']=str(cache);save()
        qa=dict(schema='warm.original-PB-new-stage-admission.v1',first_admitted_utc=first.isoformat(),absolute_deadline_utc=min(end,first+datetime.timedelta(seconds=1800)).isoformat(),qualification_plan_sha256=H.sha(p['qualification_plan']),original_windows_unchanged=True)
        qaf=root/'qualification.admission.json';immutable(qaf,qa);state['qualification_started']=True;save()
        result=run('qualification',command('qualify_rebuttal_original_PB.py',p['qualification_plan'])+['--admission',str(qaf),'--expected-admission-sha256',H.sha(qaf)],1800)
        state['qualification_archive']=archive_if_present(qualification['output'],'qualification');save()
        if result['exit_code']!=0:raise RuntimeError('Fixed namespace qualification failed;source not replaced')
        qualified=H.read(Path(qualification['output'])/'operator.json')
        assert qualified['status']=='complete_one_original_replan_four_native_commands' and qualified['namespace_case_id']==entry['case_id']
        trace=H.read(p['factual_template']);trace.update(first_admitted_utc=first.isoformat(),absolute_deadline_utc=min(end,H.now()+datetime.timedelta(seconds=7200)).isoformat(),qualification_operator_sha256=H.sha(Path(qualification['output'])/'operator.json'))
        tf=root/'factual.plan.json';tref=immutable(tf,trace);state['factual_plan']=tref;state['trace_started']=True;save()
        result=run('factual',command('collect_rebuttal_original_specialist_trace.py',tf),7200)
        state['factual_archive']=archive_if_present(trace['output'],'factual');save()
        if result['exit_code']!=0:raise RuntimeError('Fixed namespace factual episode incomplete;no query or source replacement')
        factual=Path(trace['output']);fop=H.read(factual/'operator.json');assert fop['status']=='complete_fixed_original_factual_episode'
        state.update(source_native_commands=fop['native_commands'],source_replans=fop['inference_queries'],source_success_flag=fop['success_flag']);save()
        for query in p['queries']:
            if fop['probe_status'].get(str(query))!='sealed_before_execution' or fop['native_commands']<query*4+4:
                state['queries'][str(query)]=dict(status='not_reached_no_replacement');save();continue
            if (end-H.now()).total_seconds()<180:
                state['queries'][str(query)]=dict(status='not_started_original_hard_window_insufficient');save();continue
            # Torch only reads a pre-execution CPU tensor snapshot here. No model
            # is constructed or CUDA initialized by the supervisor.
            import torch
            model=torch.load(factual/f'probes/q{query:04d}/model_output.pt',map_location='cpu',weights_only=False)
            pool=H.read(p['pool_template']);pool.update(schema=N.SCHEMAS['pool'],namespace2_registration=qualification['namespace2_registration'],namespace2_roster=qualification['namespace2_roster'],namespace_case_id=p['case_id'],physical_seed=p['physical_seed'],rmbench_runtime=qualification['rmbench_runtime'],native_worker_lease=qualification['native_worker_lease'],namespace2_factual_plan=tref,factual_source_plan_sha256=tref['sha256'],query_replan=query,selected_rank=int(model['warm_retrospection']['selected_candidate_index']),factual_source=str(factual),output=str(root/f'pool-q{query:04d}'),source_archive=state['factual_archive']['archive'],source_archive_sha256=state['factual_archive']['sha256'],recorded_prefix_TOPP_policy=p['recorded_prefix_TOPP_policy'],native_TOPP_observer_source=trace['native_TOPP_observer_source'],native_TOPP_observer_source_sha256=trace['native_TOPP_observer_source_sha256'])
            pool['inputs']=[dict(path=str(f),sha256=H.sha(f)) for f in sorted(factual.rglob('*')) if f.is_file()]
            pf=root/f'pool-q{query:04d}.plan.json';pref=immutable(pf,pool);state['queries'][str(query)]=dict(status='CPU_seal_pending',plan=pref);save()
            seal=run(f'q{query}_CPU_seal',command('collect_rebuttal_original_PB_pool.py',pf)+['--preflight-only'],900,CPU=True)
            state['queries'][str(query)]['CPU_seal_exit']=seal['exit_code'];save()
            if seal['exit_code']!=0:
                state['queries'][str(query)].update(status='CPU_seal_failed_fixed_query_retained',archive=archive_if_present(pool['output'],f'q{query}_CPU_failed'));save();continue
            pa=dict(schema='warm.original-PB-physical-pool-admission.v1',first_admitted_utc=H.now().isoformat(),absolute_deadline_utc=end.isoformat(),plan_sha256s={str(pf):H.sha(pf)},source_policy_forwards=0,original_windows_unchanged=True)
            af=root/f'pool-q{query:04d}.admission.json';immutable(af,pa)
            result=run(f'q{query}_pool',command('collect_rebuttal_original_PB_pool.py',pf)+['--admission',str(af),'--expected-admission-sha256',H.sha(af)],max(1,(end-H.now()).total_seconds()-120))
            state['queries'][str(query)].update(status='terminal_pool_archived' if result['exit_code']==0 else 'incomplete_pool_archived_no_replacement',exit_code=result['exit_code'],archive=archive_if_present(pool['output'],f'q{query}_pool'));save()
        state.update(status='terminal_namespace_source_all_existing_output_archived',ended_utc=H.now().isoformat(),full_protocol_pools=sum(v['status']=='terminal_pool_archived' for v in state['queries'].values()),automatic_restarts=0);save()
    except BaseException as exc:
        state.update(status='failed_or_interrupted_namespace_source_preserve_all',error=repr(exc),traceback=traceback.format_exc(),ended_utc=H.now().isoformat());save();raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

if __name__=='__main__':main()
