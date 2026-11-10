#!/usr/bin/env python3
"""Bounded, leased orchestration of explicitly typed native/recorded-DEV roles.

No Git access, implicit model promotion, outcome-based sample selection or
automatic retry of an unknown invocation. A shard keeps its first-start clock.
"""
from pathlib import Path
import argparse,datetime,fcntl,importlib.util,json,os,signal,subprocess,sys,time,uuid

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('matched_native_helpers',ROOT/'src/fastwam/research/matched_native.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def verified_result(worker,plan,plan_sha,case_id,role):
    """Admit complete native or identified bad model output, never generic exit1."""
    result=M.read_json(worker/'result.json')
    expected=dict(schema='warm.rebuttal.matched-native-role.v1',task=plan['task'],phase=plan['phase'],episode_id=case_id,role=role,
        plan_sha256=plan_sha,checkpoint_sha256=plan['roles'][role]['checkpoint_sha256'],
        evaluation_source_sha256=plan['evaluation_source_sha256'],controller_source_sha256=plan['controller_source_sha256'])
    if any(result.get(k)!=v for k,v in expected.items()):raise ValueError('Worker result differs from the fixed typed role')
    if result['status']=='complete_recorded_DEV':
        if plan['phase']!='recorded_dev' or result['success'] is not None or [x['frame'] for x in result['queries']]!=plan['query_frames']:
            raise ValueError('Recorded DEV must never supply a task-success label')
    elif result['status']=='complete_native':
        import numpy as np
        count=result['executed_commands']
        if plan['phase']!='formal' or type(result['success']) is not int or result['success'] not in (0,1):raise ValueError('Native outcome label unavailable')
        if not 0<count<=plan['official_step_limit'] or (not result['success'] and count!=plan['official_step_limit']):
            raise ValueError('Unsuccessful role did not reach its native horizon')
        with np.load(worker/'qpos_trace.npz',allow_pickle=False) as trace:
            if any(trace[k].shape!=(count,14) or not np.isfinite(trace[k]).all() for k in ('commands','before','after')):raise ValueError('Incomplete native trace')
            initial_state=trace['before'][0].copy()
        from fastwam.memory.manifest import sha256_array
        with np.load(worker/'initial_observation.npz',allow_pickle=False) as obs:
            if not np.array_equal(obs['native_qpos'],initial_state) or {k:sha256_array(obs[k]) for k in obs.files}!=result['initial_identity']['observation_arrays']:
                raise ValueError('Native first command differs from the sealed initial observation')
        if len(result['queries'])!=(count+3)//4 or [r['frame'] for r in result['queries']]!=list(range(0,count,4)):
            raise ValueError('Native replan trace is incomplete')
        if not (worker/'final_observation.npz').exists():raise ValueError('Final native observation missing')
    elif result['status']=='identifiable_policy_failure':
        if plan['phase']!='formal' or result['success']!=0 or not M.invalid_action_proof(worker):
            raise ValueError('Policy failure requires sealed returned invalid action evidence')
    elif result['success'] is not None:raise ValueError('Infrastructure/interruption outcomes must remain unknown')
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--expected-plan-sha256',required=True);ap.add_argument('--max-new-roles',type=int)
    ap.add_argument('--resume-known-terminal',action='store_true');ap.add_argument('--validate-only',action='store_true')
    cli=ap.parse_args()
    if cli.max_new_roles is not None and cli.max_new_roles<1:raise ValueError('Positive new-role invocation bound required')
    if M.file_sha(cli.plan)!=cli.expected_plan_sha256:raise ValueError('Frozen campaign plan differs')
    plan=M.validate_plan(M.read_json(cli.plan));phase_end=M.require_deadline(plan)
    if cli.validate_only:print(json.dumps(dict(status='plan_validated_not_executed',phase=plan['phase'])));return
    root=Path(plan['run_root']).resolve()
    if root.is_relative_to(ROOT.resolve()) or root.is_relative_to(Path(plan['evaluation_code']).resolve()):raise ValueError('Evidence cannot live in code')
    root.mkdir(parents=True,exist_ok=True);lock=(root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity_file=root/'campaign_identity.json'
    if identity_file.exists():
        identity=M.read_json(identity_file)
        if identity['plan_sha256']!=cli.expected_plan_sha256:raise ValueError('Existing shard scientific identity differs')
    else:
        identity=dict(plan_sha256=cli.expected_plan_sha256,first_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        M.atomic_json(identity_file,identity)
    first=datetime.datetime.fromisoformat(identity['first_started_utc']).timestamp()
    shard_end=min(phase_end,first+plan.get('shard_max_seconds',5400))
    state=dict(status='preflight',plan_sha256=cli.expected_plan_sha256,first_started_utc=identity['first_started_utc'],
        deadline_utc=datetime.datetime.fromtimestamp(shard_end,datetime.timezone.utc).isoformat(),new_roles_this_invocation=0)
    child=None;interrupted=False
    def save():state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();M.atomic_json(root/'campaign_state.json',state)
    def stop(sig,_):
        nonlocal interrupted
        interrupted=True
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);save()
    try:
        while True:
            records=[];known_terminal=[];attempts=[]
            for p in sorted((root/'attempts').glob('*/attempt.json')):
                meta=M.read_json(p);attempts.append(meta)
                if meta['plan_sha256']!=cli.expected_plan_sha256:raise ValueError('Foreign campaign attempt')
                exitfile=p.parent/'exit.json';worker=p.parent/'worker'
                if not exitfile.exists():raise RuntimeError('An attempt has unknown exit;do not duplicate it')
                if (worker/'result.json').exists():
                    row=verified_result(worker,plan,cli.expected_plan_sha256,meta['case_id'],meta['role']);row['attempt_id']=p.parent.name
                else:row=dict(episode_id=meta['case_id'],role=meta['role'],status='unknown_before_result',success=None,attempt_id=p.parent.name)
                records.append(row)
                if row['status'].startswith('unknown'):
                    if not cli.resume_known_terminal:raise RuntimeError('Known terminal unknown outcome preserved;explicit inspected resume required')
                    known_terminal.append(p.parent.name)
            indices=[0] if plan['phase']=='recorded_dev' else plan['case_ids']
            request=M.next_request(records,indices,known_terminal_attempts=known_terminal)
            pairs=[]
            for index in indices:
                selected=[r for r in records if r['episode_id']==index and r['status'] in ('complete_recorded_DEV','complete_native','identifiable_policy_failure')]
                if len(selected)==2:
                    byrole={r['role']:r for r in selected};pairs.append(M.pair_receipts(byrole['full'],byrole['pooled_memory_prior'],recorded_dev=plan['phase']=='recorded_dev'))
            M.atomic_json(root/'paired_verification.json',dict(phase=plan['phase'],planned_pairs=len(indices),verified_pairs=pairs,
                complete_pairs=len(pairs),unknown_attempts=sum(r['status'].startswith('unknown') for r in records),formal_efficacy_claimed=False))
            if request is None:state['status']='complete_all_fixed_owned_roles';save();break
            if cli.max_new_roles is not None and state['new_roles_this_invocation']>=cli.max_new_roles:
                state['status']='partial_verified_invocation_limit';save();break
            if interrupted or time.time()>=shard_end:raise TimeoutError('Persistent phase/shard window expired or interrupted')
            index,role=request
            prior=[a for a in attempts if (a['case_id'],a['role'])==(index,role)]
            if len(prior)>=plan.get('max_attempts_per_role',2):raise RuntimeError('Bounded same-role retry limit reached;diagnose without changing the cohort')
            attempt=root/'attempts'/f'{index:04d}-{role}-{uuid.uuid4().hex}';attempt.mkdir(parents=True,exist_ok=False)
            argv=[sys.executable,'-B',str(ROOT/'scripts/run_rebuttal_matched_native.py'),'--plan',str(cli.plan),
                '--expected-plan-sha256',cli.expected_plan_sha256,'--role',role,'--case-id',str(index),'--output',str(attempt/'worker')]
            M.atomic_json(attempt/'attempt.json',dict(plan_sha256=cli.expected_plan_sha256,role=role,case_id=index,argv=argv,
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            log=(attempt/'console.log').open('xb')
            env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='true',DIFFSYNTH_MODEL_BASE_PATH=plan['model_root'],TOKENIZERS_PARALLELISM='false',
                TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=(lock.fileno(),))
            state.update(status='running',active_pid=child.pid,active_case=index,active_role=role,active_attempt=str(attempt));save()
            started=time.monotonic();termination=None
            while child.poll() is None:
                if interrupted or time.time()>=shard_end or time.monotonic()-started>=plan.get('worker_max_seconds',2400):
                    if termination is None:termination=time.monotonic();os.killpg(child.pid,signal.SIGTERM)
                    elif time.monotonic()-termination>60:os.killpg(child.pid,signal.SIGKILL)
                state['worker_heartbeat_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();time.sleep(5)
            code=child.returncode;log.close();child=None
            M.atomic_json(attempt/'exit.json',dict(exit_code=code,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),bounded_termination=termination is not None))
            # Preserve EVERY terminal output, including unknown/failed attempts.
            sys.path[:0]=[str(Path(plan['evaluation_code'])/'src')]
            from fastwam.research.resumable_pairs import archive_verified_role
            if (attempt/'worker').exists():archive_verified_role(attempt/'worker')
            state['new_roles_this_invocation']+=1
            if code!=0:
                worker=attempt/'worker'
                row=verified_result(worker,plan,cli.expected_plan_sha256,index,role) if (worker/'result.json').exists() else None
                if row is None or row['status']!='identifiable_policy_failure':raise RuntimeError('Role exited without a scientifically known outcome;all raw evidence retained')
    except BaseException as exc:
        state.update(status='failed_or_interrupted',error=repr(exc));save();raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        state['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save();lock.close()
    print(json.dumps(state),flush=True)

if __name__=='__main__':main()
