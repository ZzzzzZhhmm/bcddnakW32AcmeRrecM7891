"""Bounded independent-role ACP evaluation using an already qualified core."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.nonreal_job import source_identity


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(8*1024*1024),b''):h.update(value)
    return h.hexdigest()


def assigned_devices(count,declared,layout='two_gpu'):
    if layout not in ('two_gpu','shared_single_gpu'):raise ValueError('unknown execution layout')
    expected=1 if layout=='shared_single_gpu' else 2
    if type(count) is not int or count!=expected:raise ValueError('visible GPU count differs from the frozen execution layout')
    values=[v.strip() for v in declared.split(',')] if declared else [str(i) for i in range(count)]
    if len(values)!=count or len(set(values))!=count or any(not v or v in ('-1','all') for v in values):
        raise ValueError('ambiguous scheduler device assignment')
    return values*2 if layout=='shared_single_gpu' else values


def validate_plan(plan):
    if plan.get('schema')!='warm.rebuttal.acp-pairs.v1':raise ValueError('unsupported ACP plan')
    if plan.get('roles')!=['full','candidate_null'] or plan.get('cases')!=10 or plan.get('case_index_offset')!=0:
        raise ValueError('fixed first ten Full/null pairs required')
    if plan.get('task')!='rearrange_blocks' or plan.get('search_condition')!='full_bank':raise ValueError('unexpected primary cohort')
    layout=plan.get('execution_layout','two_gpu')
    if layout not in ('two_gpu','shared_single_gpu'):raise ValueError('unknown execution layout')
    if not 1<=plan.get('max_seconds',0)<=(12 if layout=='shared_single_gpu' else 6)*3600:
        raise ValueError('allocation exceeds the bounded layout budget')
    for key in ('core_code','bundle_args','runtime','evidence_plan','run_root'):
        if not Path(plan[key]).is_absolute():raise ValueError('absolute paths required')
    evidence=json.loads(Path(plan['evidence_plan']).read_text())
    if evidence['checkpoint_sha256']!=plan['checkpoint_sha256'] or evidence['closed_loop']['accepted_episodes_per_task_per_condition']<10:
        raise ValueError('author model or primary cohort differs')
    for item in plan['inputs']:
        if digest(item['path'])!=item['sha256']:raise ValueError('changed input: '+item['path'])
    if source_identity(Path(plan['core_code']))['sha256']!=plan['core_source_sha256']:raise ValueError('qualified immutable core differs')
    if Path(plan['run_root']).resolve().is_relative_to(ROOT.resolve()):raise ValueError('runtime outputs must be outside code')
    return evidence


def worker_argv(plan,role):
    if role not in plan['roles']:raise ValueError('unknown worker role')
    return [sys.executable,str(Path(plan['core_code'])/'scripts/run_rebuttal_rmbench_pairs.py'),
        '--core-code',plan['core_code'],'--bundle-args',plan['bundle_args'],'--rmbench-runtime',plan['runtime'],
        '--output',str(Path(plan['run_root'])/role),'--phase','formal','--plan',plan['evidence_plan'],
        '--seed-start','340800000','--candidate-limit','2000','--cases','10','--case-index-offset','0',
        '--roles',role,'--search-condition','full_bank']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--expected-plan-sha256',required=True);parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if digest(args.plan)!=args.expected_plan_sha256:raise ValueError('ACP plan hash differs')
    plan=json.loads(args.plan.read_text());validate_plan(plan)
    if args.validate_only:print(json.dumps(dict(status='validated_not_executed',roles=plan['roles'],cases=10)));return
    output=Path(plan['run_root']);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'active.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    summary_path=output/'acp_summary.json'
    if summary_path.exists():raise FileExistsError('existing batch receipt: inspect it before any retry')
    source=source_identity(ROOT);workers={};started=time.monotonic();reason=None
    summary=dict(status='preflight',controller_source_sha256=source['sha256'],core_source_sha256=plan['core_source_sha256'],
        checkpoint_sha256=plan['checkpoint_sha256'],plan_sha256=digest(args.plan),task=plan['task'],
        execution_layout=plan.get('execution_layout','two_gpu'),
        requested_pairs=10,complete_pairs=0,scope='Fixed primary closed-loop success trials; no timing benchmark or automatic efficacy conclusion.')
    def save():
        temp=summary_path.with_suffix('.partial');temp.write_text(json.dumps(summary,indent=2));temp.replace(summary_path)
    def stop(signum,_frame):
        nonlocal reason
        reason='scheduler_signal_'+str(signum)
        for worker in workers.values():
            if worker['process'].poll() is None:os.killpg(worker['process'].pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);save()
    try:
        import torch
        count=torch.cuda.device_count()
        devices=assigned_devices(count,os.environ.get('CUDA_VISIBLE_DEVICES'),summary['execution_layout'])
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip().splitlines()
        hardware=dict(devices=[dict(name=torch.cuda.get_device_name(i),capability=list(torch.cuda.get_device_capability(i)),
            bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(count)],drivers=driver,
            torch_version=str(torch.__version__),torch_cuda=torch.version.cuda,scheduler_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        (output/'hardware.json').write_text(json.dumps(hardware,indent=2))
        if any(row['capability']!=[9,0] or row['bytes']<75*1024**3 for row in hardware['devices']):
            raise RuntimeError('H10080GB qualification requires SM90 and at least75GiB per assigned GPU')
        if not driver or set(driver)!={plan['graphics_driver_version']}:raise RuntimeError('unqualified graphics driver; preserve preflight and qualify its native client before simulation')
        if __import__('shutil').disk_usage(output).free<512*1024**2:raise OSError('AFS reserve below512MiB')
        for role,device in zip(plan['roles'],devices,strict=True):
            env=os.environ.copy();scratch=Path('/tmp')/(plan['name']+'-'+role);scratch.mkdir(exist_ok=True)
            env.update(CUDA_VISIBLE_DEVICES=device,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',DIFFSYNTH_SKIP_DOWNLOAD='true',TOKENIZERS_PARALLELISM='false',
                DIFFSYNTH_MODEL_BASE_PATH=plan['model_root'],WARM_IMMUTABLE_ASSET_HASH_CACHE='1',
                PYTHONPATH=plan['curobo_source']+':'+plan['core_code']+'/src:'+plan['core_code'],
                PYTORCH_KERNEL_CACHE_PATH=str(scratch/'kernels'),TRITON_CACHE_DIR=str(scratch/'triton'),
                XDG_CACHE_HOME=str(scratch/'xdg'),CUDA_CACHE_PATH=str(scratch/'cuda'),
                LD_LIBRARY_PATH=plan['graphics_libraries']+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''),
                VK_ICD_FILENAMES=plan['vulkan_icd'],__EGL_VENDOR_LIBRARY_FILENAMES=plan['egl_vendor'])
            spec=dict(schema='warm.nonreal.job.v1',readiness='ready',max_seconds=plan['max_seconds']-180,
                argv=worker_argv(plan,role),evidence_type='fixed_primary_RB_ten_official_episodes_on_one_assigned_GPU',
                claim='Original author checkpoint, official700 steps, H32/prefix4/NFE10/K32, fresh-reset role and unchanged factual recollection.',
                inputs=plan['inputs'],qualification=[])
            spec_path=output/(role+'.job.json');spec_path.write_text(json.dumps(spec,indent=2))
            log=(output/(role+'-launcher.log')).open('xb')
            proc=subprocess.Popen([sys.executable,str(Path(plan['core_code'])/'scripts/nonreal_job.py'),'--spec',str(spec_path),
                '--output-root',str(output/'jobs')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers[role]=dict(process=proc,log=log,device=device)
        summary['status']='running';save()
        stopping_at=None
        while any(w['process'].poll() is None for w in workers.values()):
            if time.monotonic()-started>plan['max_seconds'] and reason is None:stop(signal.SIGTERM,None);reason='bounded_wall_time'
            if reason:
                stopping_at=stopping_at or time.monotonic()
                if time.monotonic()-stopping_at>120:
                    for w in workers.values():
                        if w['process'].poll() is None:os.killpg(w['process'].pid,signal.SIGKILL)
            summary['workers']={role:dict(pid=w['process'].pid,device=w['device'],exit_code=w['process'].poll()) for role,w in workers.items()}
            summary['elapsed_seconds']=time.monotonic()-started;save();time.sleep(10)
        codes={role:w['process'].returncode for role,w in workers.items()};summary['worker_exit_codes']=codes
        if reason or any(codes.values()):raise RuntimeError('incomplete ACP workers: '+str(codes)+' '+str(reason))
        merge=subprocess.run([sys.executable,str(Path(plan['core_code'])/'scripts/merge_rebuttal_role_trials.py'),'--inputs',
            str(output/'full'),str(output/'candidate_null'),'--output',str(output/'pairs.jsonl')],capture_output=True,text=True)
        summary.update(merge_exit_code=merge.returncode,merge_stdout=merge.stdout,merge_stderr=merge.stderr)
        if merge.returncode:raise RuntimeError('pair/reset identity validation failed')
        rows=[json.loads(line) for line in (output/'pairs.jsonl').read_text().splitlines()]
        summary['complete_pairs']=sum(row['status']=='complete' for row in rows)
        if len(rows)!=10 or summary['complete_pairs']!=10:raise RuntimeError('fixed cohort incomplete')
        summary.update(status='complete',full_successes=sum(row['roles']['full']['success'] for row in rows),
            null_successes=sum(row['roles']['candidate_null']['success'] for row in rows))
    except BaseException as exc:
        summary.update(status='failed_or_interrupted',error=repr(exc));raise
    finally:
        if any(w['process'].poll() is None for w in workers.values()):stop(signal.SIGTERM,None)
        for w in workers.values():
            try:w['process'].wait(timeout=125)
            except subprocess.TimeoutExpired:os.killpg(w['process'].pid,signal.SIGKILL);w['process'].wait()
            w['log'].close()
        summary.update(elapsed_seconds=time.monotonic()-started,controller_source_sha256_after=source_identity(ROOT)['sha256']);save()
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


if __name__=='__main__':main()
