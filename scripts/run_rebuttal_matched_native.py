#!/usr/bin/env python3
"""Run one explicitly typed Full/pooled role on fixed DEV or a native reset.

Model/deployment code is independently frozen. This collector enables no Full
probe, uses no physical unpack, and never contacts Git or installs packages.
"""
from pathlib import Path
import argparse,datetime,importlib,importlib.util,json,os,random,sys,traceback

HELPER=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('matched_native_helpers',HELPER/'src/fastwam/research/matched_native.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def install_core(plan):
    core=Path(plan['evaluation_code']).resolve(strict=True)
    sys.path[:0]=[str(core),str(core/'src')]
    import fastwam,fastwam.research
    if not Path(fastwam.__file__).resolve().is_relative_to(core):raise RuntimeError('Foreign model package imported')
    fastwam.research.__path__.append(str(HELPER/'src/fastwam/research'))
    from scripts.nonreal_job import source_identity
    if source_identity(core)['sha256']!=plan['evaluation_source_sha256']:raise ValueError('Common evaluation source changed')
    if source_identity(HELPER)['sha256']!=plan['controller_source_sha256']:raise ValueError('Immutable collector source changed')
    return core,source_identity

def save_observation(path,obs):
    import numpy as np
    from fastwam.memory.manifest import sha256_array
    values={k:np.asarray(v['rgb']) for k,v in obs['observation'].items() if 'rgb' in v}
    values['native_qpos']=np.asarray(obs['joint_action']['vector'])
    with Path(path).open('xb') as stream:np.savez_compressed(stream,**values)
    return {k:sha256_array(v) for k,v in values.items()}

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

def save_context(capture,path):
    import torch
    from dataclasses import fields
    ctx=capture['context']
    data={f.name:getattr(ctx,f.name).detach().cpu().clone() for f in fields(ctx) if torch.is_tensor(getattr(ctx,f.name))}
    for key in ('base_gaussian','source','conditioning_tokens'):
        if capture.get(key) is not None:data['actual_'+key]=capture[key].detach().cpu().clone()
    with Path(path).open('xb') as stream:torch.save(data,stream)

def execute_recorded_dev(plan,policy,out,result,budget):
    import numpy as np
    source=Path(plan['recorded_inputs']);instruction=plan['instruction']
    class Task:
        def get_instruction(self):return instruction
    result['initial_identity']=dict(recorded_episode=435,input_source_manifest_sha256=plan['recorded_input_manifest_sha256'],instruction=instruction)
    for frame in plan['query_frames']:
        budget();result['active_frame']=frame
        with np.load(source/f'input-frame{frame:04d}.npz',allow_pickle=False) as arrays:
            cameras={k:dict(rgb=arrays[k].copy()) for k in ('head_camera','left_camera','right_camera')}
            raw=arrays['raw_state'].copy();commands=arrays['recorded_commands'].copy()
        if raw.shape!=(14,) or commands.shape!=(4,14):raise ValueError('Frozen recorded DEV prefix shape differs')
        obs=dict(observation=cameras,joint_action=dict(vector=raw))
        with M.capture_replan(policy,out/'generations'/f'replan-{frame//4:04d}') as capture:policy._replan(obs,Task())
        row=dict(frame=frame,**M.replan_receipt(capture));result['queries'].append(row)
        if frame in (0,32,36):save_context(capture,out/f'context-frame{frame:04d}.pt')
        policy.queue.clear()
        # Only after generation, install the four ACTUALLY RECORDED actions.
        # Current input contains no later recorded image or teacher effect.
        for command in commands:
            normalized=policy.action_normalizer.normalize(command[None,:],fail_on_clip=True)[0]
            policy.controller.note_executed_action(command,model_space_action=normalized);policy._executed_policy_actions+=1
        M.atomic_json(out/'progress.json',dict(status='recorded_DEV_replan',frame=frame,queries_complete=len(result['queries'])))
    if not result['queries'][-1]['history_observation_tokens'] or not result['queries'][-1]['history_action_slots']:
        raise ValueError('Real causal recorded history was not consumed')
    policy.seal_active(reason='matched_recorded_DEV_engineering_no_outcome',success=False)
    result.update(status='complete_recorded_DEV',success=None,generated_commands_executed=0,simulator_loaded=False,engineering_only=True)

class TOPPObserver:
    """Record swallowed planner failures without changing official execution."""
    def __init__(self,env):self.env=env;self.original={};self.errors=[]
    def __enter__(self):
        for name in ('left_mplib_planner','right_mplib_planner'):
            obj=getattr(self.env.robot,name);old=obj.TOPP;self.original[name]=(old,'TOPP' in vars(obj))
            def call(*args,_old=old,_name=name,**kwargs):
                try:return _old(*args,**kwargs)
                except Exception as exc:self.errors.append(dict(planner=_name,error=repr(exc)));raise
            obj.TOPP=call
        return self
    def __exit__(self,*_):
        for name,(old,had) in self.original.items():
            obj=getattr(self.env.robot,name)
            if had:obj.TOPP=old
            else:del obj.TOPP
        self.original.clear()

def execute_native(plan,policy,out,result,budget,case_id):
    import numpy as np
    import torch
    from scripts.qualify_rebuttal_rmbench_task import resolved_config,source_identity
    from fastwam.research.native_replay import physical_joint_state
    runtime=Path(plan['native_runtime']).resolve(strict=True)
    if source_identity(runtime)['sha256']!=plan['runtime_source_sha256']:raise ValueError('Native environment source differs')
    isolation=M.read_json(runtime/'MATCHED_WORKER_ISOLATION.json')
    if isolation!=plan['runtime_isolation']:raise ValueError('Unqualified writable simulator worker paths')
    sys.path[:0]=[str(runtime),str(runtime/'description/utils'),plan['curobo_source']];os.chdir(runtime)
    from generate_episode_instructions import generate_episode_descriptions
    roster=M.read_json(plan['roster']);entry=roster['entries'][case_id]
    if entry['episode_id']!=case_id:raise ValueError('Fixed reset roster ID differs')
    env=getattr(importlib.import_module('envs.'+plan['task']),plan['task'])()
    config=resolved_config(runtime,plan['task'],entry['seed']);config['now_ep_num']=case_id
    commands=[];before=[];after=[]
    try:
        random.seed(entry['seed']);np.random.seed(entry['seed']);torch.manual_seed(entry['seed']);torch.cuda.manual_seed_all(entry['seed'])
        env.setup_demo(**config)
        if env.step_lim!=plan['official_step_limit']:raise ValueError('Official native horizon differs')
        text=str(np.random.choice(generate_episode_descriptions(plan['task'],[entry['expert_info']['info']],100)[0]['unseen']))
        env.set_instruction(instruction=text);obs=env.get_obs()
        identity=dict(observation_arrays=save_observation(out/'initial_observation.npz',obs),task_state=task_values(env),instruction=text)
        measured=[{k:v.tolist() if hasattr(v,'tolist') else v for k,v in row.items()} for row in physical_joint_state(env)]
        M.require_initial_native_identity(entry,identity,measured)
        result['initial_measured_physical']=measured
        result['seed']=entry['seed'];result['initial_identity']=identity
        M.atomic_json(out/'initial_identity.json',identity)
        with (out/'physical_commands.jsonl').open('x',encoding='utf-8',buffering=1) as stream,TOPPObserver(env) as observer:
            while env.take_action_cnt<env.step_lim and not env.eval_success:
                budget();obs=env.get_obs();frame=policy._executed_policy_actions
                if not policy.queue:
                    result['active_frame']=frame
                    try:
                        with M.capture_replan(policy,out/'generations'/f'replan-{frame//4:04d}') as capture:policy._replan(obs,env)
                    except M.PolicyOutputError:
                        if 'source' in capture:
                            result['queries'].append(dict(frame=frame,returned_invalid_action=True,**M.replan_receipt(capture)))
                            save_context(capture,out/f'invalid-action-context-frame{frame:04d}.pt')
                        raise
                    result['queries'].append(dict(frame=frame,**M.replan_receipt(capture)))
                    if frame//4 in (0,8,48):
                        save_context(capture,out/f'context-frame{frame:04d}.pt');save_observation(out/f'observation-frame{frame:04d}.npz',obs)
                    M.atomic_json(out/'progress.json',dict(status='native_replan',frame=frame,commands=len(commands),queries_complete=len(result['queries'])))
                queued=policy.queue._items[0];command=np.array(queued.environment_space,copy=True)
                before_value=np.asarray(obs['joint_action']['vector']).copy();physical_before=physical_joint_state(env)
                observer.errors.clear();previous=env.take_action_cnt
                def plain(items):return [{k:v.tolist() if hasattr(v,'tolist') else v for k,v in row.items()} for row in items]
                stream.write(json.dumps(dict(kind='command_begin',index=frame,command=command.tolist(),physical_before=plain(physical_before),take_action_cnt_before=int(previous)))+'\n')
                command_error=None
                try:
                    policy.step(env,obs)
                    if env.take_action_cnt!=previous+1:raise RuntimeError('Official executor command counter did not advance exactly once')
                except BaseException as exc:command_error=repr(exc);raise
                finally:
                    commands.append(command);before.append(before_value)
                    try:after_value=np.asarray(env.get_obs()['joint_action']['vector']).copy()
                    except Exception:after_value=np.full(14,np.nan,dtype=np.float32)
                    after.append(after_value)
                    stream.write(json.dumps(dict(kind='command_end',index=frame,take_action_cnt_after=int(env.take_action_cnt),
                        physical_after=plain(physical_joint_state(env)),observed_TOPP_errors=observer.errors.copy(),error=command_error))+'\n')
                if not np.isfinite(after_value).all():raise RuntimeError('Nonfinite native state;retain as unknown')
        complete=bool(env.eval_success or env.take_action_cnt>=env.step_lim)
        if not complete:raise RuntimeError('Official episode has no native terminal cause')
        result.update(status='complete_native',success=int(bool(env.eval_success)),executed_commands=len(commands),official_step_limit=int(env.step_lim),
            max_reward=float(env.max_reward),final_task_state=task_values(env),engineering_only=False)
        save_observation(out/'final_observation.npz',env.get_obs())
        policy.seal_active(reason='success' if env.eval_success else 'step_limit',success=bool(env.eval_success))
    finally:
        with (out/'qpos_trace.npz').open('xb') as stream:
            np.savez_compressed(stream,commands=np.asarray(commands,dtype=np.float32).reshape(-1,14),
                before=np.asarray(before,dtype=np.float32).reshape(-1,14),after=np.asarray(after,dtype=np.float32).reshape(-1,14))
        result['command_attempts']=len(commands);result['executed_commands']=int(getattr(env,'take_action_cnt',0))
        result.pop('generated_commands_executed',None)
        result['generated_commands_submitted']=len(commands)
        result['official_command_counter']=int(getattr(env,'take_action_cnt',0))
        env.close_env(clear_cache=True)
        if source_identity(runtime)['sha256']!=plan['runtime_source_sha256']:raise ValueError('Native source changed during this role')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--expected-plan-sha256',required=True)
    parser.add_argument('--role',choices=M.ROLES,required=True);parser.add_argument('--case-id',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--validate-only',action='store_true')
    cli=parser.parse_args()
    if M.file_sha(cli.plan)!=cli.expected_plan_sha256:raise ValueError('Frozen matched plan differs')
    plan=M.validate_plan(M.read_json(cli.plan));M.require_deadline(plan)
    M.bind_local_model_root(plan,os.environ)
    if (plan['phase']=='recorded_dev' and cli.case_id!=0) or (plan['phase']=='formal' and cli.case_id not in plan['case_ids']):raise ValueError('Unplanned role/case')
    native_lease=M.activate_native_isolation(plan) if plan['phase']=='formal' else None
    core,source_identity=install_core(plan);M.verify_inputs(plan)
    out=cli.output.resolve();controller=HELPER.resolve()
    if out.is_relative_to(core) or out.is_relative_to(controller):raise ValueError('Raw evidence must be outside both source trees')
    runtime=M.load_runtime(plan,cli.role,out)
    if cli.validate_only:print(json.dumps(dict(status='typed_contract_validated_not_executed',role=cli.role,phase=plan['phase'])));return
    out.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='true',TOKENIZERS_PARALLELISM='false')
    import torch
    from experiments.rmbench.warm_policy.deploy_policy import RMBenchWarmPolicy
    result=dict(schema='warm.rebuttal.matched-native-role.v1',role=cli.role,phase=plan['phase'],task=plan['task'],episode_id=cli.case_id,
        plan_sha256=cli.expected_plan_sha256,checkpoint_sha256=plan['roles'][cli.role]['checkpoint_sha256'],
        evaluation_source_sha256=plan['evaluation_source_sha256'],controller_source_sha256=plan['controller_source_sha256'],
        status='loading_typed_policy',success=None,queries=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        generated_commands_executed=0,engineering_only=plan['phase']=='recorded_dev')
    M.atomic_json(out/'runtime_args.json',runtime);M.atomic_json(out/'result.json',result);policy=None
    def budget():M.require_deadline(plan)
    try:
        torch.cuda.set_per_process_memory_fraction(float(plan.get('memory_fraction',.75)))
        policy=RMBenchWarmPolicy(runtime);typed=M.typed_policy_receipt(policy,plan,cli.role)
        result.update(typed);M.atomic_json(out/'typed_policy_receipt.json',typed)
        policy._next_episode_index=cli.case_id;policy.reset()
        if policy.controller.active_episode_index!=cli.case_id:raise RuntimeError('Native query episode cursor differs')
        if plan['phase']=='recorded_dev':execute_recorded_dev(plan,policy,out,result,budget)
        else:execute_native(plan,policy,out,result,budget,cli.case_id)
    except M.PolicyOutputError as exc:
        result.update(status='qualification_failed_policy_output' if plan['phase']=='recorded_dev' else 'identifiable_policy_failure',
            success=None if plan['phase']=='recorded_dev' else 0,error=repr(exc),traceback=traceback.format_exc(),
            outcome_reason='Returned invalid model action is sealed before consumer validation;no infrastructure error was imputed.')
        raise
    except BaseException as exc:
        result.update(status='unknown_runtime_or_interruption',success=None,error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if policy is not None:
            try:policy.close()
            except Exception as exc:
                result['cleanup_error']=repr(exc)
                result['outcome_before_cleanup_error']=dict(status=result['status'],success=result['success'])
                result.update(status='unknown_runtime_or_interruption',success=None)
        result['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        if source_identity(core)['sha256']!=plan['evaluation_source_sha256'] or source_identity(HELPER)['sha256']!=plan['controller_source_sha256']:
            result['outcome_before_source_change']=dict(status=result['status'],success=result['success'])
            result.update(status='unknown_source_change',success=None);M.atomic_json(out/'result.json',result)
            raise ValueError('Immutable scientific/collector source changed')
        M.atomic_json(out/'result.json',result)
        if result['status'].startswith('unknown'):raise RuntimeError('Role result remains unknown;inspect preserved receipts')
    print(json.dumps(dict(status=result['status'],role=cli.role,queries=len(result['queries']),success=result['success'])),flush=True)

if __name__=='__main__':main()
