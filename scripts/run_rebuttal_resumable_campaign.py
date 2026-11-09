"""Resume fixed official role trials after preemption without changing samples.

The scientific worker remains the bound, immutable original core. This controller
only assigns work, verifies durable evidence and exports counts. No Git or installs.
"""
from __future__ import annotations
import argparse,csv,json,os,signal,subprocess,sys,time,uuid
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.nonreal_job import source_identity
from fastwam.research.resumable_pairs import file_sha,read_json,read_roles,unique_roles,paired_rows,next_role,summarize


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def campaign_deadline(identity,expected_hash,max_seconds,now):
    if identity['plan_sha256']!=expected_hash:raise ValueError('existing campaign identity differs')
    start=datetime.fromisoformat(identity['first_started_utc']).timestamp()
    if start>now+60:raise ValueError('host clock precedes the durable campaign start')
    end=start+max_seconds
    if now>=end:raise TimeoutError('original persistent campaign budget expired')
    return end


def load_plan(path,expected):
    if file_sha(path)!=expected:raise ValueError('frozen campaign plan differs')
    p=read_json(path)
    if p.get('schema')!='warm.rebuttal.resumable-pairs.v1' or p['task'] not in ('rearrange_blocks','battery_try'):
        raise ValueError('unsupported resumable campaign')
    if type(p['max_seconds']) is not int or not 1<=p['max_seconds']<=12*3600 or not 60<=p['worker_max_seconds']<=4500:
        raise ValueError('invalid bounded campaign/worker time')
    if not 0<p['memory_fraction']<=.7 or type(p['max_attempts_per_role']) is not int or not 1<=p['max_attempts_per_role']<=3:raise ValueError('invalid resource/retry budget')
    if p['official_step_limit']!=({'rearrange_blocks':700,'battery_try':1000}[p['task']]):raise ValueError('official limit changed')
    for key in ('core_code','runtime','run_root','roster','bundle_args','evidence_plan'):
        if not Path(p[key]).is_absolute():raise ValueError('absolute paths required')
    if Path(p['run_root']).resolve().is_relative_to(ROOT.resolve()) or Path(p['run_root']).resolve().is_relative_to(Path(p['core_code']).resolve()):
        raise ValueError('evidence must be outside source')
    for item in p['inputs']:
        if file_sha(item['path'])!=item['sha256']:raise ValueError('changed input: '+item['path'])
    evidence=read_json(p['evidence_plan']);roster=read_json(p['roster'])
    if file_sha(p['evidence_plan'])!=p['evidence_plan_sha256'] or evidence['checkpoint_sha256']!=p['checkpoint_sha256']:
        raise ValueError('original evidence plan or author checkpoint differs')
    if roster['status']!='qualified_reset_roster_only' or roster['task']!=p['task'] or roster['core_source_sha256']!=p['core_source_sha256']:
        raise ValueError('unqualified or differently bound reset roster')
    if roster['runtime_source_sha256']!=p['runtime_source_sha256'] or roster['seed_start']!=evidence['closed_loop']['reset_seed_start']:
        raise ValueError('original runtime or reset namespace differs')
    if source_identity(Path(p['core_code']))['sha256']!=p['core_source_sha256']:raise ValueError('qualified scientific core changed')
    seeds={r['episode_id']:r['seed'] for r in roster['entries']}
    if len(seeds)!=len(roster['entries']) or any(type(k) is not int or type(v) is not int for k,v in seeds.items()) or list(seeds)!=list(range(len(seeds))) or list(seeds.values())!=sorted(set(seeds.values())):
        raise ValueError('reset roster must retain ordered expert admissions')
    if not p['case_ids'] or any(type(k) is not int for k in p['case_ids']) or p['case_ids']!=sorted(set(p['case_ids'])) or not set(p['case_ids']).issubset(seeds):raise ValueError('invalid fixed cases')
    if p['cohort_case_ids']!=list(range(len(seeds))):raise ValueError('cohort reporting must retain every fixed roster case')
    return p,seeds


def owned_records(root,plan,seeds):
    records=[];attempts=[]
    for manifest in sorted((root/'attempts').glob('*/attempt.json')):
        a=read_json(manifest)
        if a['plan_sha256']!=plan['_plan_sha256']:raise ValueError('attempt belongs to a different frozen campaign')
        if a['episode_id'] not in plan['case_ids'] or a['seed']!=seeds[a['episode_id']]:raise ValueError('attempt changed fixed reset')
        rows=read_roles(manifest.parent/'worker',plan,seeds)
        if any(r['row']['episode_id']!=a['episode_id'] or r['role']!=a['role'] for r in rows):raise ValueError('worker returned unrequested role/case')
        records.extend(rows);attempts.append((manifest.parent,a,rows))
    unique_roles(records)
    return records,attempts


def export(root,plan,seeds,records):
    all_records=list(records)
    for external in plan.get('external_workers',[]):
        if Path(external).exists():all_records.extend(read_roles(external,plan,seeds))
    rows=paired_rows(all_records,seeds,plan['cohort_case_ids']);stats=summarize(rows)
    atomic_json(root/'verified_pairs.json',dict(task=plan['task'],plan_sha256=plan['_plan_sha256'],pairs=rows,summary=stats))
    with (root/'paired_counts.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['task','episode_id','seed','status','full_success','null_success','full_minus_null'])
        for r in rows:writer.writerow([plan['task'],r['episode_id'],r['seed'],r['status'],
            r['roles']['full']['success'] if r['roles']['full'] else '',r['roles']['candidate_null']['success'] if r['roles']['candidate_null'] else '',r['full_minus_null']])
    lines=['% Verified fixed-cohort counts only; pending roles are not failures.',r'\begin{tabular}{rrrrr}',r'\toprule',
        r'Complete / planned & Full successes & Null successes & Full-only & Null-only \\',r'\midrule',
        f"{stats['complete_pairs']} / {stats['requested_pairs']} & {stats['full_successes']} & {stats['null_successes']} & {stats['full_only']} & {stats['null_only']} "+r'\\',r'\bottomrule',r'\end{tabular}']
    (root/'verified_counts.tex').write_text('\n'.join(lines)+'\n')
    return stats


def worker_environment(plan,scratch):
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',DIFFSYNTH_SKIP_DOWNLOAD='true',TOKENIZERS_PARALLELISM='false',
        DIFFSYNTH_MODEL_BASE_PATH=plan['model_root'],WARM_IMMUTABLE_ASSET_HASH_CACHE='1',PYTHONPATH=plan['curobo_source']+':'+plan['core_code']+'/src:'+plan['core_code'],
        PYTORCH_KERNEL_CACHE_PATH=str(scratch/'kernels'),TRITON_CACHE_DIR=str(scratch/'triton'),XDG_CACHE_HOME=str(scratch/'xdg'),CUDA_CACHE_PATH=str(scratch/'cuda'),
        LD_LIBRARY_PATH=plan['graphics_libraries']+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),
        VK_ICD_FILENAMES=plan['vulkan_icd'],__EGL_VENDOR_LIBRARY_FILENAMES=plan['egl_vendor'])
    return env


def role_wrapper(plan, *, qualification_plan=None, qualification_sha=None, receipt=None):
    prefix='import torch,runpy\ntorch.cuda.set_per_process_memory_fraction('+repr(plan['memory_fraction'])+')\n'
    if qualification_plan is not None:
        module=ROOT/'src/fastwam/research/encoder_provenance.py'
        prefix=('import importlib.util,runpy\n'+
                'spec=importlib.util.spec_from_file_location("warm_encoder_provenance",'+repr(str(module))+')\n'+
                'module=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(module)\n'+
                'module.install('+repr(str(qualification_plan))+','+repr(qualification_sha)+','+repr(str(receipt))+',memory_fraction='+repr(plan['memory_fraction'])+')\n')
    return prefix+'runpy.run_path('+repr(str(Path(plan['core_code'])/'scripts/run_rebuttal_rmbench_pairs.py'))+",run_name='__main__')\n"


def main():
    import fcntl
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--expected-plan-sha256',required=True);parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--encoder-qualification-plan',type=Path)
    parser.add_argument('--expected-encoder-qualification-plan-sha256')
    args=parser.parse_args();plan,seeds=load_plan(args.plan,args.expected_plan_sha256);plan['_plan_sha256']=args.expected_plan_sha256
    if bool(args.encoder_qualification_plan)!=bool(args.expected_encoder_qualification_plan_sha256):raise ValueError('both encoder qualification path and SHA required')
    if args.encoder_qualification_plan is not None:
        if file_sha(args.encoder_qualification_plan)!=args.expected_encoder_qualification_plan_sha256:raise ValueError('encoder qualification plan changed')
        qualification=read_json(args.encoder_qualification_plan)
        if qualification['core_code']!=plan['core_code'] or qualification['core_source_sha256']!=plan['core_source_sha256']:raise ValueError('qualification uses a different bound core')
    if args.validate_only:print(json.dumps(dict(status='validated_not_executed',cases=plan['case_ids'])));return
    root=Path(plan['run_root']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity_path=root/'campaign_identity.json';source=source_identity(ROOT)
    if identity_path.exists():
        identity=read_json(identity_path)
        if identity['plan_sha256']!=args.expected_plan_sha256:raise ValueError('existing campaign identity differs')
    else:
        identity=dict(plan_sha256=args.expected_plan_sha256,first_started_utc=datetime.now(timezone.utc).isoformat(),scientific_core_sha256=plan['core_source_sha256'])
        atomic_json(identity_path,identity)
    # The durable wall-clock budget does not reset when ACP restarts a container.
    deadline=campaign_deadline(identity,args.expected_plan_sha256,plan['max_seconds'],time.time())
    child=None;stop_reason=None
    state=dict(status='preflight',controller_source_sha256=source['sha256'],plan_sha256=args.expected_plan_sha256,
        first_started_utc=identity['first_started_utc'],deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat())
    def save():atomic_json(root/'campaign_state.json',state)
    def stop(signum,_frame):
        nonlocal stop_reason
        stop_reason='scheduler_signal_'+str(signum)
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);save()
    try:
        import torch
        if torch.cuda.device_count()!=1:raise RuntimeError('this plan requires exactly one scheduler-assigned visible GPU')
        prop=torch.cuda.get_device_properties(0)
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip().splitlines()
        hardware=dict(name=prop.name,capability=list(torch.cuda.get_device_capability(0)),bytes=prop.total_memory,drivers=driver,
            torch=str(torch.__version__),cuda=torch.version.cuda,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        if hardware['capability']!=[9,0] or hardware['bytes']<75*1024**3 or not driver or set(driver)!={plan['graphics_driver_version']}:
            raise RuntimeError('unqualified native GPU or graphics client')
        hardware_path=root/('hardware-'+uuid.uuid4().hex+'.json');atomic_json(hardware_path,hardware)
        while True:
            records,attempts=owned_records(root,plan,seeds);accepted=unique_roles(records);stats=export(root,plan,seeds,records)
            request=next_role(accepted,seeds,plan['case_ids'])
            state.update(status='running',**stats,accepted_role_count=len(accepted),updated_utc=datetime.now(timezone.utc).isoformat());save()
            if request is None:state['status']='complete_verified_owned_cases';save();break
            if stop_reason or time.time()>=deadline:raise TimeoutError(stop_reason or 'original persistent campaign budget expired')
            index,role,seed=request
            if sum(a['episode_id']==index and a['role']==role for _,a,_ in attempts)>=plan['max_attempts_per_role']:
                raise RuntimeError('bounded retry limit reached for the same fixed role; diagnose before another campaign amendment')
            if __import__('shutil').disk_usage(root).free<512*1024**2:raise OSError('AFS reserve below512MiB')
            attempt=root/'attempts'/(f'{index:04d}-{role}-'+uuid.uuid4().hex);attempt.mkdir(parents=True,exist_ok=False)
            wrapper=attempt/'run_role.py';wrapper.write_text(role_wrapper(plan,qualification_plan=args.encoder_qualification_plan,
                qualification_sha=args.expected_encoder_qualification_plan_sha256,receipt=attempt/'encoder_qualification.json'))
            argv=[sys.executable,str(wrapper),'--core-code',plan['core_code'],'--bundle-args',plan['bundle_args'],'--rmbench-runtime',plan['runtime'],
                '--output',str(attempt/'worker'),'--phase','formal','--plan',plan['evidence_plan'],'--seed-start',str(seed),'--candidate-limit','1',
                '--cases','1','--case-index-offset',str(index),'--roles',role,'--search-condition','full_bank']
            meta=dict(episode_id=index,role=role,seed=seed,plan_sha256=args.expected_plan_sha256,controller_source_sha256=source['sha256'],
                argv=argv,wrapper_sha256=file_sha(wrapper),started_utc=datetime.now(timezone.utc).isoformat(),hardware_receipt=str(hardware_path))
            if args.encoder_qualification_plan is not None:meta['encoder_qualification_plan_sha256']=args.expected_encoder_qualification_plan_sha256
            atomic_json(attempt/'attempt.json',meta)
            scratch=Path('/tmp')/('warm-resumable-'+attempt.name);scratch.mkdir(exist_ok=False)
            log=(attempt/'console.log').open('xb')
            # The scientific worker inherits the lease: a killed controller can
            # never admit a second worker while the original child is still alive.
            child=subprocess.Popen(argv,env=worker_environment(plan,scratch),stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(lock.fileno(),))
            state.update(active_case=index,active_role=role,active_seed=seed,active_pid=child.pid,active_attempt=str(attempt));save()
            start=time.monotonic();terminating=None
            while child.poll() is None:
                if (time.time()>=deadline or time.monotonic()-start>=plan['worker_max_seconds']) and stop_reason is None:
                    stop(signal.SIGTERM,None);stop_reason='bounded_worker_or_campaign_time'
                if stop_reason:
                    terminating=terminating or time.monotonic()
                    if time.monotonic()-terminating>120:os.killpg(child.pid,signal.SIGKILL)
                state['heartbeat_utc']=datetime.now(timezone.utc).isoformat();save();time.sleep(5)
            log.close();code=child.returncode
            atomic_json(attempt/'exit.json',dict(exit_code=code,reason=stop_reason,ended_utc=datetime.now(timezone.utc).isoformat()))
            if code or stop_reason:raise RuntimeError('role interrupted or failed; retain attempt and diagnose exit'+str(code)+' '+str(stop_reason))
            evidence=read_roles(attempt/'worker',plan,seeds)
            if len(evidence)!=1:raise RuntimeError('worker exited without exactly one verified complete role')
            from fastwam.research.resumable_pairs import archive_verified_role
            atomic_json(attempt/'verified_role.json',evidence[0]);archive_verified_role(attempt/'worker');child=None
            print(json.dumps(dict(stage='verified_role',episode_id=index,role=role,seed=seed,success=evidence[0]['row']['roles'][role]['success'])),flush=True)
    except BaseException as exc:
        state.update(status='failed_or_interrupted',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(updated_utc=datetime.now(timezone.utc).isoformat(),controller_source_sha256_after=source_identity(ROOT)['sha256']);save()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
