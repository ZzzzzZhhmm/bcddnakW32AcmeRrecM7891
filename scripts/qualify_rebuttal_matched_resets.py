#!/usr/bin/env python3
"""Freeze first-fifty expert resets, then qualify private native workers.

No policy is loaded. Admission uses only the official expert. A reset error is
never a reason to substitute another seed. All evidence lives outside Git.
"""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,importlib,importlib.util,json,os,random,sys,time,traceback

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('matched_helpers',ROOT/'src/fastwam/research/matched_native.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def require_window(plan):
    now=datetime.datetime.now(datetime.timezone.utc)
    first=datetime.datetime.fromisoformat(plan['first_admitted_utc']);end=datetime.datetime.fromisoformat(plan['absolute_deadline_utc'])
    if first.tzinfo is None or end.tzinfo is None or not 0<(end-first).total_seconds()<=7200:
        raise ValueError('One persistent qualification window of at most two hours is required')
    if now<first or now>=end:raise TimeoutError('Original reset qualification window expired;do not reset it')


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if hasattr(value,'tolist'):return plain(value.tolist())
    if value is None or type(value) in (str,int,float,bool):return value
    raise TypeError('Unspecified expert metadata type: '+type(value).__name__)


def task_values(env):
    import numpy as np
    if env.task_name=='rearrange_blocks':
        return dict(block1=env.block1.get_pose().p.tolist(),block2=env.block2.get_pose().p.tolist(),
            target1=env.target_pose1.p.tolist(),target2=env.target_pose2.p.tolist(),stage_id=int(env.stage_id),press_cnt=int(env.press_cnt))
    if env.task_name=='battery_try':
        return dict(battery1_p=env.battery1.get_pose().p.tolist(),battery1_q=env.battery1.get_pose().q.tolist(),
            battery2_p=env.battery2.get_pose().p.tolist(),battery2_q=env.battery2.get_pose().q.tolist(),
            target1=np.asarray(env.target_pose1_p).tolist(),target2=np.asarray(env.target_pose2_p).tolist(),correct_combination=list(env.correct_combination))
    raise ValueError('No independently declared task state reader')


def validate_prefix(entries,old):
    expected=[dict(episode_id=x['episode_id'],seed=x['seed']) for x in old]
    actual=[dict(episode_id=x['episode_id'],seed=x['seed']) for x in entries[:len(old)]]
    if actual!=expected[:len(actual)]:raise ValueError('Existing RB10 prefix changed;retain the failed admission and do not replace seeds')


def load_native(plan):
    core=Path(plan['evaluation_code']).resolve(strict=True)
    sys.path[:0]=[str(core),str(core/'src')]
    from scripts.nonreal_job import source_identity
    from scripts.qualify_rebuttal_rmbench_task import resolved_config,source_identity as runtime_identity
    if source_identity(core)['sha256']!=plan['evaluation_source_sha256']:raise ValueError('Frozen common model/deployment source changed')
    runtime=Path(plan['native_runtime']).resolve(strict=True)
    if runtime_identity(runtime)['sha256']!=plan['runtime_source_sha256']:raise ValueError('Official native source changed')
    if M.read_json(runtime/'MATCHED_WORKER_ISOLATION.json')!=plan['runtime_isolation']:raise ValueError('Private native paths differ')
    sys.path[:0]=[str(runtime),str(runtime/'description/utils'),plan['curobo_source']];os.chdir(runtime)
    return resolved_config,runtime_identity


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def observe_initial(env,task,entry,output):
    import numpy as np
    from fastwam.memory.manifest import sha256_array
    from generate_episode_instructions import generate_episode_descriptions
    instruction=str(np.random.choice(generate_episode_descriptions(task,[entry['expert_info']['info']],100)[0]['unseen']))
    env.set_instruction(instruction=instruction);obs=env.get_obs()
    arrays={k:np.asarray(v['rgb']) for k,v in obs['observation'].items() if 'rgb' in v}
    arrays['native_qpos']=np.asarray(obs['joint_action']['vector'])
    expected={'native_qpos','left_camera','right_camera'}|set(env.cameras.static_camera_name)
    if set(arrays)!=expected or not {'head_camera','left_camera','right_camera'}.issubset(arrays):
        raise ValueError('Official initial camera/state inventory differs: '+repr((sorted(arrays),sorted(expected))))
    if arrays['native_qpos'].shape!=(14,) or not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError('Nonfinite or incorrectly shaped initial native state')
    with output.open('xb') as stream:np.savez_compressed(stream,**arrays)
    return dict(observation_arrays={k:sha256_array(v) for k,v in arrays.items()},task_state=task_values(env),instruction=instruction)


def python_write_audit(allowed):
    """Observe Python-level writes; block writes outside the private workers.

    C++ filesystem calls are outside this hook. Concurrent exact-reset/hold
    checks and source review are reported separately, not as a syscall audit.
    """
    allowed=tuple(Path(p).resolve() for p in allowed);observed=set()
    def audit(event,args):
        path=None
        if event=='open' and not isinstance(args[0],int):
            flags=args[2] or 0
            if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):path=args[0]
        elif event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.rename') and args and not isinstance(args[0],int):path=args[0]
        if path is None:return
        path=Path(os.fsdecode(path)).resolve()
        if str(path).startswith(('/dev/','/proc/','/sys/')):return
        if not any(path==p or path.is_relative_to(p) for p in allowed):raise PermissionError('Native Python write outside qualified private roots: '+str(path))
        observed.add(str(path))
    sys.addaudithook(audit)
    return observed



def wait_native_probe_barrier(plan,index,identity):
    barrier=plan.get('concurrency_barrier')
    if barrier is None:return None
    if plan['mode']!='probe':raise ValueError('Native barrier is only an isolation qualification,never expert admission or policy TEST')
    names=barrier['worker_ids'];worker=plan['runtime_isolation']['worker_id'];root=Path(barrier['directory']).resolve(strict=True)
    if worker not in names or len(names)!=len(set(names)) or not 2<=len(names)<=6:raise ValueError('Unplanned native concurrency cohort')
    if not 0<barrier['timeout_seconds']<=300:raise ValueError('Bounded native barrier wait required')
    if not root.is_relative_to(Path(plan['barrier_parent']).resolve(strict=True)):raise ValueError('Barrier path escapes its declared batch')
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload=dict(worker_id=worker,pid=os.getpid(),reset_index=index,task=plan['task'],utc=now,initial_identity=identity)
    ready=root/f'{index}-{worker}.json'
    if ready.exists():raise RuntimeError('Unknown prior native barrier attempt;do not overwrite or duplicate it')
    M.atomic_json(ready,payload);start=time.monotonic()
    while True:
        require_window(plan);rows=[]
        for name in names:
            path=root/f'{index}-{name}.json'
            if not path.exists():break
            row=M.read_json(path)
            if row['worker_id']!=name or row['reset_index']!=index:raise ValueError('Native barrier contains a foreign worker')
            rows.append(row)
        if len(rows)==len(names):return dict(ready_utc=now,all_ready_observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),ready_records={name:M.file_sha(root/f'{index}-{name}.json') for name in names},simultaneously_initialized_native_scenes=len(names))
        if time.monotonic()-start>=barrier['timeout_seconds']:raise TimeoutError('Actual initialized native scenes did not meet at the barrier')
        time.sleep(.1)


def reset_once(env,plan,entry,out,index,resolved_config,hold=False):
    import numpy as np
    from fastwam.research.native_replay import physical_joint_state
    config=resolved_config(Path(plan['native_runtime']),plan['task'],entry['seed']);config['now_ep_num']=entry['episode_id']
    seed_all(entry['seed']);env.setup_demo(**config)
    if env.step_lim!={'rearrange_blocks':700,'battery_try':1000}[plan['task']]:raise ValueError('Official native horizon changed')
    identity=observe_initial(env,plan['task'],entry,out/f'initial-{index}.npz')
    result=dict(initial_identity=identity,physical_initial=plain(physical_joint_state(env)),pid=os.getpid(),
        planner_ipc=dict(kind='anonymous_multiprocessing_Pipe',left_fd=env.robot.left_conn.fileno(),right_fd=env.robot.right_conn.fileno(),left_process=env.robot.left_proc.pid,right_process=env.robot.right_proc.pid,left_endpoint_inode=os.fstat(env.robot.left_conn.fileno()).st_ino,right_endpoint_inode=os.fstat(env.robot.right_conn.fileno()).st_ino),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if hold:
        result['concurrency_barrier']=wait_native_probe_barrier(plan,index,identity)
        commands=[];state=np.asarray(env.get_obs()['joint_action']['vector']).copy()
        for i in range(4):
            before=plain(physical_joint_state(env));previous=int(env.take_action_cnt)
            env.take_action(state.copy(),action_type='qpos')
            commands.append(dict(index=i,command=state.tolist(),take_action_cnt_before=previous,take_action_cnt_after=int(env.take_action_cnt),physical_before=before,physical_after=plain(physical_joint_state(env))))
        result['hold_commands']=commands
    env.close_env(clear_cache=True);result['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    M.atomic_json(out/f'reset-{index}.json',result)
    return result


def known_expert_geometry_rejection(detail):
    return (detail.get('stage')=='play_once' and detail.get('error_type')=='AssertionError'
        and detail.get('error_message')=='target_pose cannot be None for move action.'
        and any(x['name']=='grasp_actor' and x['filename'].endswith('/envs/_base_task.py') for x in detail.get('frames',[]))
        and bool(detail.get('frames')) and detail['frames'][-1]['name']=='__init__'
        and detail['frames'][-1]['filename'].endswith('/envs/utils/action.py'))


def prepare_resume(plan,out):
    import shutil
    ref=plan.get('resume_proof')
    if ref is None:return {}
    if M.file_sha(ref['path'])!=ref['sha256']:raise ValueError('Inspected reset resume proof changed')
    proof=M.read_json(ref['path']);source=Path(proof['source_root']).resolve(strict=True)
    if (proof.get('schema')!='warm.rebuttal.reset-resume-reconciliation.v1' or proof['task']!=plan['task']
        or proof['native_source_sha256']!=plan['runtime_source_sha256'] or proof['known_exit'] not in (1,-15)
        or proof['original_window_reset'] is not False or proof['policy_outcomes_used'] is not False):
        raise ValueError('Unknown/uninspected reset exit must not be resumed')
    for name,record in proof['source_manifest'].items():
        file=source/name
        if not file.resolve(strict=True).is_relative_to(source) or M.file_sha(file)!=record['sha256']:raise ValueError('Archived reset prefix changed')
    shutil.copytree(source/'candidates',out/'candidates')
    for name,record in proof['source_manifest'].items():
        if M.file_sha(out/name)!=record['sha256']:raise ValueError('Reset prefix copy differs')
    M.atomic_json(out/'resume_proof.json',proof)
    return proof.get('admission_overrides',{})


def roster(plan,out,resolved_config):
    from envs.utils.create_actor import UnStableError
    env=getattr(importlib.import_module('envs.'+plan['task']),plan['task'])();entries=[]
    old=M.read_json(plan['legacy_prefix'])['entries'] if plan['task']=='rearrange_blocks' else []
    overrides=prepare_resume(plan,out)
    for offset in range(2000):
        require_window(plan);seed=340800000+offset;episode=len(entries)
        case=out/'candidates'/f'{offset:04d}';existing=(case/'expert_result.json').exists()
        row=dict(seed=seed,candidate_offset=offset,episode_id_if_admitted=episode,status='expert_started',utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        if existing:
            row=M.read_json(case/'expert_result.json')
            if row['seed']!=seed or row['episode_id_if_admitted']!=episode:raise ValueError('Preserved seed/episode order differs')
            if row['status']=='unknown_runtime':
                override=overrides.get(str(offset))
                if (override is None or override['raw_expert_result_sha256']!=M.file_sha(case/'expert_result.json')
                    or override['status']!='expert_rejected_undefined_grasp'):raise RuntimeError('Unresolved expert runtime failure retained;do not skip')
                row=row|dict(status=override['status'],reconciliation=override)
                M.atomic_json(case/'expert_result_reconciled.json',row)
            if row['status'] not in ('accepted','expert_rejected','unstable_seed','expert_rejected_undefined_grasp'):raise ValueError('Unknown preserved admission status')
            ok=row['status']=='accepted';info=row.get('expert_info')
        else:
            case.mkdir(parents=True,exist_ok=False);M.atomic_json(case/'started.json',row)
            config=resolved_config(Path(plan['native_runtime']),plan['task'],seed);config['now_ep_num']=episode;stage='setup'
            try:
                seed_all(seed);env.setup_demo(**config);stage='play_once';info=plain(env.play_once());stage='close';env.close_env()
                ok=bool(env.plan_success and env.check_success());row.update(status='accepted' if ok else 'expert_rejected',expert_info=info,plan_success=bool(env.plan_success))
            except UnStableError as exc:
                row.update(status='unstable_seed',error=repr(exc));ok=False;env.close_env(clear_cache=True)
            except BaseException as exc:
                detail=dict(stage=stage,error_type=type(exc).__name__,error_message=str(exc),frames=[dict(filename=x.filename,name=x.name,lineno=x.lineno) for x in traceback.extract_tb(exc.__traceback__)])
                row.update(status='unknown_runtime',error=repr(exc),traceback=traceback.format_exc(),failure_detail=detail)
                if known_expert_geometry_rejection(detail):
                    row['status']='expert_rejected_undefined_grasp';ok=False;env.close_env(clear_cache=True)
                else:M.atomic_json(case/'expert_result.json',row);raise
            M.atomic_json(case/'expert_result.json',row)
        if ok:
            entry=dict(episode_id=episode,seed=seed,expert_info=info);validate_prefix(entries+[entry],old)
            rows=[]
            for index in range(2):
                record=case/f'reset-{index}.json';image=case/f'initial-{index}.npz'
                if record.exists():
                    if not image.exists():raise ValueError('Preserved complete reset lacks its raw image')
                    value=M.read_json(record)
                else:
                    if image.exists():image.rename(case/f'interrupted_initial-{index}.npz')
                    value=reset_once(env,plan,entry,case,index,resolved_config)
                rows.append(value)
            first,second=rows
            if first['initial_identity']!=second['initial_identity'] or first['physical_initial']!=second['physical_initial']:
                raise ValueError('Expert-admitted initial reset does not reproduce;retain this seed and diagnose without substitution')
            entry.update(initial_identity=first['initial_identity'],physical_initial=first['physical_initial'],two_resets_exact=True,candidate_offset=offset)
            if (case/'admitted_entry.json').exists():
                if M.read_json(case/'admitted_entry.json')!=entry:raise ValueError('Preserved complete reset entry differs')
            else:M.atomic_json(case/'admitted_entry.json',entry)
            entries.append(entry)
        with (out/'admission.jsonl').open('ab') as stream:
            stream.write((json.dumps(row,sort_keys=True,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
        M.atomic_json(out/'progress.json',dict(status='expert_admission',accepted=len(entries),candidate_offset=offset,seed=seed,last_status=row['status'],prior_completed_candidate_reused=existing))
        print(json.dumps(dict(stage='expert_admission',task=plan['task'],seed=seed,status=row['status'],accepted=len(entries),reused_preserved_expert=existing)),flush=True)
        if len(entries)==50:break
    if len(entries)!=50:raise RuntimeError('Original 2000-candidate seed budget exhausted')
    validate_prefix(entries,old)
    final=dict(schema='warm.rebuttal.matched-reset-roster.v1',status='qualified_first50_expert_admitted_reset_roster',task=plan['task'],entries=entries,
        seed_start=340800000,candidate_limit=2000,runtime_source_sha256=plan['runtime_source_sha256'],evaluation_source_sha256=plan['evaluation_source_sha256'],
        plan_sha256=M.file_sha(plan['_path']),legacy_prefix_sha256=M.file_sha(plan['legacy_prefix']) if old else None,two_initial_resets_exact_per_entry=True,
        resume_proof=plan.get('resume_proof'),policy_loaded=False,policy_outcomes_used_for_admission=False,scope='Outcome-independent reset qualification, not task success-rate or model competence evidence.')
    M.atomic_json(out/'roster.json',final)


def probe(plan,out,resolved_config):
    roster=M.read_json(plan['roster']);entry=roster['entries'][0]
    if roster['status']!='qualified_first50_expert_admitted_reset_roster' or roster['task']!=plan['task']:raise ValueError('Complete fixed roster required')
    if M.file_sha(plan['roster'])!=plan['roster_sha256']:raise ValueError('Frozen roster changed')
    env=getattr(importlib.import_module('envs.'+plan['task']),plan['task'])()
    rows=[reset_once(env,plan,entry,out,i,resolved_config,hold=True) for i in range(2)]
    for row in rows:
        if row['initial_identity']!=entry['initial_identity'] or row['physical_initial']!=entry['physical_initial']:raise ValueError('Concurrent fresh reset differs from the fixed cohort')
        if [x['take_action_cnt_after'] for x in row['hold_commands']]!=[1,2,3,4]:raise ValueError('Native hold counter differs')
    if rows[0]['hold_commands']!=rows[1]['hold_commands']:raise ValueError('Concurrent native hold physical trace does not reproduce')
    M.atomic_json(out/'probe.json',dict(status='qualified_two_exact_resets_and_four_native_holds',task=plan['task'],isolation=plan['runtime_isolation'],entry_id=0,seed=entry['seed'],reset_count=2,hold_commands_each=4,policy_loaded=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--expected-plan-sha256',required=True)
    a=p.parse_args()
    if M.file_sha(a.plan)!=a.expected_plan_sha256:raise ValueError('Frozen reset qualification plan changed')
    plan=M.read_json(a.plan);require_window(plan)
    if plan.get('schema')!='warm.rebuttal.matched-reset-qualification.v1' or plan['mode'] not in ('roster','probe') or plan['task'] not in ('rearrange_blocks','battery_try'):raise ValueError('Unspecified qualification')
    out=Path(plan['output']).resolve()
    if out.is_relative_to(ROOT.resolve()) or out.is_relative_to(Path(plan['evaluation_code']).resolve()):raise ValueError('Evidence must be outside frozen code')
    lease=M.activate_native_isolation(plan)
    out.mkdir(parents=True,exist_ok=False);started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    M.atomic_json(out/'operator.json',dict(status='running',pid=os.getpid(),started_utc=started,plan_sha256=a.expected_plan_sha256))
    writes=python_write_audit([out,plan['runtime_isolation']['writable_root'],str(Path(plan['native_runtime'])/'data'),str(Path(plan['native_runtime'])/'camera_images')]+([plan['concurrency_barrier']['directory']] if plan.get('concurrency_barrier') is not None else []))
    try:
        resolved_config,runtime_identity=load_native(plan)
        (roster if plan['mode']=='roster' else probe)(plan|{'_path':str(a.plan)},out,resolved_config)
        if runtime_identity(Path(plan['native_runtime']))['sha256']!=plan['runtime_source_sha256']:raise ValueError('Native source changed')
        status='complete'
    except BaseException as exc:
        status='failed_or_interrupted';M.atomic_json(out/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        M.atomic_json(out/'python_writes.json',dict(paths=sorted(writes),scope='Python audit only;not a native syscall audit'))
        M.atomic_json(out/'operator.json',dict(status=locals().get('status','failed_or_interrupted'),pid=os.getpid(),started_utc=started,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=a.expected_plan_sha256,policy_loaded=False))
        fcntl.flock(lease,fcntl.LOCK_UN);os.close(lease)


if __name__=='__main__':main()
