"""Collect a frozen original-specialist factual trace and pre-execution probes.

This is a supplementary fixed-point study, not recovery of the submitted DEV
selection. Original model mathematics and validators remain unchanged. The
collector never accesses a Git remote or tests worktree cleanliness. All output
and raw checkpoints stay outside Git. A failed fixed seed is never replaced.
"""
from pathlib import Path
import argparse,dataclasses,datetime,hashlib,importlib,importlib.util,json,os,pickle,sys,traceback

spec=importlib.util.spec_from_file_location('PB_namespace_binding',Path(__file__).resolve().parents[1]/'src/fastwam/research/pb_namespace.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)


TRACE_PROTOCOLS={
 'warm.original-specialist-factual-trace.v1':('blocks_ranking_try','blocks_ranking_try-s3407-v1',14000,600,0),
 'warm.original-specialist-factual-trace.native-BT.v2':('battery_try','battery_try-s3407-v1',12000,1000,0),
 'warm.original-specialist-factual-trace.native-PB.v4':('press_button','press_button-s3407-v2',12000,1500,0),
 'warm.original-specialist-factual-trace.cohort-case0.v3':('battery_try','battery_try-s3407-v1',12000,1000,340800000),
}

def validate_trace_plan(plan):
 if plan.get('schema')==N.SCHEMAS['factual']:
  _,entry=N.validate_execution_binding(plan,'factual');task,run,step,cap,physical_seed='press_button','press_button-s3407-v2',12000,1500,entry['seed']
 else:
  if plan.get('schema') not in TRACE_PROTOCOLS:raise ValueError('Unknown original factual trace protocol')
  task,run,step,cap,physical_seed=TRACE_PROTOCOLS[plan['schema']]
 if (plan.get('physical_seed')!=physical_seed or plan.get('max_native_commands')!=cap or plan.get('capture_replans')!=[8,48,96]
     or plan.get('qualified_original_run')!=run or plan.get('checkpoint_step')!=step):
  raise ValueError('Original fixed task/checkpoint/seed/horizon/probes differ')
 if task in ('battery_try','press_button') and (plan.get('task_name')!=task or plan.get('root_seed')!=3407):raise ValueError('Native task/root binding is required')
 if task=='press_button' and type(plan.get('physical_seed')) is not int:raise ValueError('PB physical seed must be an integer')
 if task=='press_button' and (plan.get('original_source_commit')!='47277e58624366506ee032f4ad27560c7e4bdd19' or plan.get('original_source_sha256')!='4a1d05cb548f16bf6a9b239082b02d4697aa8e8c1f43c2f5d8ffbaec196462fd' or plan.get('checkpoint_sha256')!='a765d987b2df0d228782980592c39aaed0e837040c7f6f48c905e4641f79c285'):raise ValueError('Original PB source/checkpoint binding differs')
 if plan['schema'].endswith('cohort-case0.v3'):
  if plan.get('cohort_case_id')!=0 or plan.get('roster_sha256')!='612cb5d1728b2bebef7c09ee61bd50933c64821be31e1df0a7aa475b61d666b8':
   raise ValueError('Exact pre-frozen BT cohort case0 binding required')
 return task

def physical_joint_values(env):
 return [dict(name=art.name,qpos=art.get_qpos().tolist(),qvel=art.get_qvel().tolist()) for art in env.scene.get_all_articulations()]

def press_button_task_state(env):
 # Do not call the stateful official check_success() for an audit observation.
 names=('card_id_1','card_id_2','press_cnt_1','press_cnt_2','press_cnt_check_button',
        'press_flag_1','press_flag_2','press_flag_check_button')
 values={n:getattr(env,n) for n in names}
 return {n:v.item() if hasattr(v,'item') else v for n,v in values.items()}


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--plan',required=True,type=Path)
 parser.add_argument('--expected-plan-sha256',required=True)
 cli=parser.parse_args();assert hashlib.sha256(cli.plan.read_bytes()).hexdigest()==cli.expected_plan_sha256
 plan=json.loads(cli.plan.read_text());expected_task=validate_trace_plan(plan)
 is_namespace=plan['schema']==N.SCHEMAS['factual']
 is_cohort=plan['schema'].endswith('cohort-case0.v3');is_native=expected_task in ('battery_try','press_button')
 if expected_task=='press_button':
  import fcntl
  lease=(Path(plan['native_worker_lease']) if is_namespace else Path(plan['evidence_root'])/'original_specialist_native_execution.worker.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 P=Path(plan['evidence_root']);C=Path(plan['original_source']);H=Path(plan['reference_controller'])
 CORE_SHA=plan['original_source_sha256'];COMMIT=plan['original_source_commit'];SIM_SHA=plan['simulator_source_sha256']
 out=Path(plan['output']);assert (is_namespace or out.parent==P) and not out.exists();out.mkdir()
 qualification=Path(plan['qualification_root']);qualified_operator=json.loads((qualification/'operator.json').read_text())
 assert qualified_operator['status']=='complete_one_original_replan_four_native_commands'
 assert hashlib.sha256((qualification/'operator.json').read_bytes()).hexdigest()==plan['qualification_operator_sha256']
 entry=None
 if is_namespace:entry=N.bind_case(plan)[1]
 if is_cohort:
  roster_file=Path(plan['roster']);assert hashlib.sha256(roster_file.read_bytes()).hexdigest()==plan['roster_sha256']
  roster=json.loads(roster_file.read_text());assert roster['status']=='qualified_first50_expert_admitted_reset_roster' and len(roster['entries'])==50
  entry=roster['entries'][0];assert entry['episode_id']==0 and entry['seed']==plan['physical_seed']
  registration_file=Path(plan['cohort_registration']);assert hashlib.sha256(registration_file.read_bytes()).hexdigest()==plan['cohort_registration_sha256']
  registration=json.loads(registration_file.read_text());assert registration['case_ids']==list(range(50)) and registration['roster']['sha256']==plan['roster_sha256']

 deadline=datetime.datetime.fromisoformat(plan['absolute_deadline_utc'])
 assert datetime.datetime.now(datetime.timezone.utc)<deadline

 stamp=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 state=dict(status='binding',started_utc=stamp(),scope=plan['claim'],plan_sha256=cli.expected_plan_sha256,planned_commands=plan['max_native_commands'],planned_probe_replans=plan['capture_replans'],inference_queries=0,native_commands=0,model_updates=0)
 def save():
  state['updated_utc']=stamp();t=out/'operator.json.partial';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(out/'operator.json')
 save();env=None;policy=None;children_before=set();hook=None;old_git=None;topp_observer=None
 try:
  spec=importlib.util.spec_from_file_location('stdlib_identity',H/'scripts/run_rebuttal_training_with_runtime_trace.py');identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
  assert identity.source_digest(C)==CORE_SHA and identity.source_digest(H)=='7a0ce8943818b02a7ee493fc3b6d666ba6b55b5362df7ab5f3a6be4c760000ae'
  preflight=Path(plan['runtime_preflight']);qualified=json.loads((preflight/'execution.json').read_text());assert qualified['static_contracts_passed'] and qualified['encoder_runtime_matches']
  args=json.loads((preflight/'runtime_args.json').read_text());args['warm_telemetry_path']=str(out/'full-telemetry.jsonl')
  assert args['task_name']==expected_task
  if is_native and args['seed']!=3407:raise ValueError('Original BT fixed evaluation root seed differs')
  (out/'runtime_args.json').write_text(json.dumps(args,indent=2)+'\n')
  asset_plan=json.loads((P/'resumable_rb_cases0003_0009_v1_plan.json').read_text());runtime=Path(plan['rmbench_runtime'])
  os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='true',DIFFSYNTH_MODEL_BASE_PATH=asset_plan['model_root'],WANDB_MODE='disabled',WARM_IMMUTABLE_ASSET_HASH_CACHE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
  sys.path[:0]=[str(C/'src'),str(C),asset_plan['curobo_source']]
  import torch,numpy as np,random,multiprocessing as mp
  import experiments.rmbench.warm_policy.deploy_policy as legacy
  from fastwam.memory.manifest import sha256_array,sha256_canonical_json
  assert Path(legacy.__file__).resolve().is_relative_to(C)
  spec=importlib.util.spec_from_file_location('frozen_native_setup',H/'scripts/qualify_rebuttal_rmbench_task.py');setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)
  sim=setup.source_identity(runtime);assert sim['sha256']==SIM_SHA
  official=Path(args['warm_task_definition_path']).parent.parent
  for rel,digest in sim['files'].items():assert sha(official/rel)==digest,rel
  state.update(helper_sha256=sha(Path(__file__)),source_sha256=CORE_SHA,simulator_source_sha256=SIM_SHA,checkpoint_sha256=qualified['checkpoint_sha256'],runtime_projection_sha256=qualified['actual_projection_sha256'],original_DEV_selected_step='unverified',static_preflight_sha256=sha(preflight/'execution.json'),old_git_worktree_invoked=False)
  old_git=legacy._git_identity
  def frozen_identity():
   assert identity.source_digest(C)==CORE_SHA
   return COMMIT,False
  legacy._git_identity=frozen_identity
  os.chdir(runtime);sys.path[:0]=[str(runtime),str(runtime/'description/utils')]
  from generate_episode_instructions import generate_episode_descriptions
  instruction_files=[Path('description/utils/generate_episode_instructions.py'),Path('description/task_instruction')/(expected_task+'.json')]
  for relative in instruction_files:assert sha(runtime/relative)==sha(official/relative),str(relative)
  state['instruction_source_sha256']={str(f):sha(runtime/f) for f in instruction_files}
  protocol=np.load(args['warm_initial_states_path'],allow_pickle=False);assert protocol.dtype.kind in 'iu' and protocol.size>0
  physical_seed=entry['seed'] if (is_cohort or is_namespace) else int(protocol.reshape(-1)[0]);assert physical_seed==plan['physical_seed']
  state.update(candidate_seed=physical_seed,seed_protocol_sha256=sha256_array(protocol),seed_selection='Previously frozen all50 cohort case0;not replacement of the separate ineligible diagnostic0.' if is_cohort else 'First scalar of original candidate namespace;no outcome-based replacement.')
  cfg=setup.resolved_config(runtime,args['task_name'],physical_seed)
  if is_namespace:cfg['now_ep_num']=entry['case_id'];state['seed_selection']='First-two expert-admitted namespace cases,chosen before any learned policy trial.'
  (out/'native_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
  children_before={x.pid for x in mp.active_children()};env=getattr(importlib.import_module('envs.'+args['task_name']),args['task_name'])()
  state['status']='fixed_seed_expert_setup_only';save();print('stage=fixed_seed_expert_setup_only',flush=True)
  random.seed(physical_seed);np.random.seed(physical_seed);torch.manual_seed(physical_seed);torch.cuda.manual_seed_all(physical_seed)
  env.setup_demo(**cfg);expert_info=entry['expert_info'] if (is_cohort or is_namespace) else env.play_once()
  if is_native and env.step_lim!=plan['max_native_commands']:raise ValueError('Original native horizon differs')
  accepted=True if (is_cohort or is_namespace) else bool(env.plan_success and env.check_success());state['fixed_expert_setup_accepted']=accepted
  if is_cohort:state.update(cohort_case_id=0,pre_frozen_cohort_sha256=plan['roster_sha256'],expert_admission_source='Existing expert-only frozen cohort;no new expert replay or policy-based admission.')
  (out/'expert_setup_metadata.json').write_text(json.dumps(expert_info,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n');save()
  env.close_env()
  if not accepted:raise RuntimeError('First fixed candidate seed rejected by original expert setup;no replacement admitted')
  state['status']='loading_original_full_policy';save();print('stage=loading_original_full_policy',flush=True)
  policy=legacy.RMBenchWarmPolicy(args)
  assert policy.model._warm_loaded_checkpoint_sha256==qualified['checkpoint_sha256']
  assert sha256_canonical_json(legacy.current_encoder_runtime(args['warm_dino_device']))==policy.contract.encoder_runtime_sha256
  state['original_policy_loaded']=True;state['status']='fresh_learned_episode_reset';save();print('stage=fresh_learned_episode_reset',flush=True)
  random.seed(physical_seed);np.random.seed(physical_seed);torch.manual_seed(physical_seed);torch.cuda.manual_seed_all(physical_seed);env.setup_demo(**cfg)
  descriptions=generate_episode_descriptions(args['task_name'],[expert_info['info']],100)[0]['unseen'];instruction=str(np.random.choice(descriptions));env.set_instruction(instruction=instruction)
  state.update(expert_history_provided_to_policy=False,instruction_metadata_source='Original evaluator expert-setup template metadata;learned episode reset afterwards,no expert actions or future effects in model history.')
  obs=env.get_obs();raw=legacy.factual_cameras(obs);qpos=legacy.factual_joint_state(obs)
  if is_namespace:state.update(N.require_initial(plan,obs,instruction,physical_joint_values(env),press_button_task_state(env)))
  if is_cohort:
   observed={k:sha256_array(np.asarray(v['rgb'])) for k,v in obs['observation'].items() if 'rgb' in v};observed['native_qpos']=sha256_array(np.asarray(obs['joint_action']['vector']))
   assert observed==entry['initial_identity']['observation_arrays'] and instruction==entry['initial_identity']['instruction']
   assert physical_joint_values(env)==entry['physical_initial']
   reader_source=Path(plan['native_TOPP_observer_source']);assert sha(reader_source)=='49888eb2854b29d6f5f384e394aaf91847dd9ebfdf9e89f6bc51fb1f8152dd4d'
   reader_spec=importlib.util.spec_from_file_location('bound_BT_cohort_task_reader',reader_source);reader_module=importlib.util.module_from_spec(reader_spec);reader_spec.loader.exec_module(reader_module)
   assert reader_module.task_values(env)==entry['initial_identity']['task_state']
   state['hidden_task_values_audited_only_not_policy_input']=True

  np.savez_compressed(out/'initial_observation.npz',**raw,native_qpos=qpos)
  with np.load(qualification/'initial_observation.npz',allow_pickle=False) as previous:
   assert set(previous.files)==set(raw)|{'native_qpos'}
   assert all(np.array_equal(previous[k],v) for k,v in {**raw,'native_qpos':qpos}.items()),'Initial native observation differs from qualification'
  state['initial_observation_exact_to_qualification']=True
  if expected_task=='press_button':
   initial=press_button_task_state(env)
   assert initial==json.loads((qualification/'initial_identity.json').read_text())['task_state_audit_only']
   state['hidden_PB_counter_state_audited_only_not_policy_input']=True
  (out/'initial_identity.json').write_text(json.dumps(dict(instruction=instruction,cameras={k:sha256_array(v) for k,v in raw.items()},qpos_sha256=sha256_array(qpos),task=args['task_name'],candidate_seed=physical_seed),indent=2)+'\n')
  actual_runtime=legacy.current_encoder_runtime(args['warm_dino_device']);(out/'actual_encoder_runtime_before_query.json').write_text(json.dumps(actual_runtime,indent=2)+'\n');assert sha256_canonical_json(actual_runtime)==policy.contract.encoder_runtime_sha256
  def cpu(value):
   if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
   if dataclasses.is_dataclass(value):return {f.name:cpu(getattr(value,f.name)) for f in dataclasses.fields(value)}
   if isinstance(value,dict):return {str(k):cpu(v) for k,v in value.items()}
   if isinstance(value,(tuple,list)):return [cpu(v) for v in value]
   if isinstance(value,(str,int,float,bool,type(None))):return value
   return dict(unserialized_type=type(value).__module__+'.'+type(value).__name__)
  capture_queries=set(range((plan['max_native_commands']+3)//4)) if is_cohort else set(plan['capture_replans'])
  state.update(primary_probe_replans=plan['capture_replans'],all_replans_sealed_for_future_audit=is_cohort)
  generations=out/'generations';generations.mkdir()
  probes=out/'probes';probes.mkdir();captured=set();frame_now=0
  def event_hook(_module,_inputs,result):
   query=frame_now//4
   if query in capture_queries:
    qdir=probes/f'q{query:04d}';qdir.mkdir(exist_ok=True)
    torch.save(cpu(result),qdir/'event_adapter.pt')
  hook=policy.model.retrospective_event_adapter.register_forward_hook(event_hook)
  old_infer=policy.model.infer_action
  def infer_observer(*a,**kw):
   query=frame_now//4;gdir=generations/f'q{query:04d}';gdir.mkdir(exist_ok=False)
   if query in capture_queries:
    qdir=probes/f'q{query:04d}';qdir.mkdir(exist_ok=True)
    torch.save(cpu(kw),qdir/'infer_kwargs.pt')
    (qdir/'sampling_rng_before.pkl').write_bytes(pickle.dumps(dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all()),protocol=4))
   value=old_infer(*a,**kw);actions=cpu(value['action'])
   assert actions.shape==(32,14) and bool(torch.isfinite(actions).all())
   if query==0:
    previous=torch.load(qualification/'original_model_output.pt',map_location='cpu',weights_only=False)['action']
    assert torch.equal(actions,previous),'First original replan differs from qualification'
    state['first_replan_exact_to_qualification']=True
   np.save(gdir/'model_actions.npy',actions.numpy(),allow_pickle=False)
   (gdir/'generation.json').write_text(json.dumps(dict(query=query,frame_index=frame_now,before_any_execution_of_this_chunk=True,checkpoint_sha256=qualified['checkpoint_sha256'],actions_sha256=sha(gdir/'model_actions.npy'),telemetry=cpu(value['warm_online_telemetry'])),indent=2)+'\n')
   if query in capture_queries:
    torch.save(cpu(value),qdir/'model_output.pt');captured.add(query)
    (qdir/'probe_seal.json').write_text(json.dumps(dict(query=query,frame_index=frame_now,utc=stamp(),before_any_new_branch_truth=True,independent_candidate_labels=0),indent=2)+'\n')
   state['inference_queries']+=1;return value
  policy.model.infer_action=infer_observer
  if is_namespace:policy._next_episode_index=entry['case_id']
  policy.reset()
  state['status']='collecting_fixed_original_factual_episode';save()
  with np.load(qualification/'native_prefix_qpos.npz',allow_pickle=False) as previous:
   prefix_before=previous['before'].copy();prefix_after=previous['after'].copy()
  qualified_physical=None
  if is_native:
   qualified_physical=json.loads((qualification/'native_prefix_physical.json').read_text())
   observer_source=Path(plan['native_TOPP_observer_source'])
   assert sha(observer_source)==plan['native_TOPP_observer_source_sha256']=='49888eb2854b29d6f5f384e394aaf91847dd9ebfdf9e89f6bc51fb1f8152dd4d'
   native_spec=importlib.util.spec_from_file_location('bound_native_TOPP_observer',observer_source);native_module=importlib.util.module_from_spec(native_spec);native_spec.loader.exec_module(native_module)
   topp_observer=native_module.TOPPObserver(env);topp_observer.__enter__()
  command_capture=[];original_take_action=env.take_action
  def command_observer(command,*a,**kw):
   assert kw.get('action_type')=='qpos'
   native=np.array(command,copy=True);assert native.shape==(14,) and np.isfinite(native).all()
   command_capture.append(native);return original_take_action(command,*a,**kw)
  env.take_action=command_observer
  trace=out/'native_trace.jsonl'
  with trace.open('x',buffering=1) as stream:
   for index in range(plan['max_native_commands']):
    if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('Original factual-trace resource window expired;do not reset or replace seed')
    frame_now=index;obs=env.get_obs();before=legacy.factual_joint_state(obs).copy()
    physical_before=physical_joint_values(env) if is_native else None
    task_before=press_button_task_state(env) if expected_task=='press_button' else None
    if not policy.queue:
     np.savez_compressed(out/f'replan-{index//4:04d}.npz',**legacy.factual_cameras(obs),native_qpos=before)
     policy._replan(obs,env)
    prior_native_count=int(env.take_action_cnt)
    if topp_observer is not None:topp_observer.errors.clear()
    policy.step(env,obs);after=legacy.factual_joint_state(env.get_obs()).copy()
    if is_native and env.take_action_cnt!=prior_native_count+1:raise RuntimeError('Official native command counter did not advance once')
    assert len(command_capture)==index+1
    row=dict(index=index,command=command_capture[-1].tolist(),command_dtype=str(command_capture[-1].dtype),before=before.tolist(),after=after.tolist(),eval_success=bool(env.eval_success),take_action_cnt=int(env.take_action_cnt))
    if expected_task=='press_button':
     row.update(task_state_before=task_before,task_state_after=press_button_task_state(env))
    if physical_before is not None:
     row.update(physical_before=physical_before,physical_after=physical_joint_values(env),observed_TOPP_errors=topp_observer.errors.copy())
     if index<4:
      reference=qualified_physical[index]
      if row['physical_before']!=reference['physical_before'] or row['physical_after']!=reference['physical_after'] or row['observed_TOPP_errors']!=reference['observed_TOPP_errors']:
       raise RuntimeError('First four measured native commands differ from BT qualification')
      if expected_task=='press_button' and (row['task_state_before']!=reference['task_state_before'] or row['task_state_after']!=reference['task_state_after']):
       raise RuntimeError('First four PB counters or flags differ from qualification')
    stream.write(json.dumps(row)+'\n');stream.flush()
    state['native_commands']+=1
    if index<4:assert np.array_equal(before,prefix_before[index]) and np.array_equal(after,prefix_after[index]),'Qualified native prefix changed'
    state.update(last_frame=index,captured_probe_replans=sorted(captured),success_flag=bool(env.eval_success));save()
    if env.eval_success:break
  if is_native and not env.eval_success and env.take_action_cnt!=env.step_lim:raise RuntimeError('Incomplete native BT trajectory;outcome remains unknown')
  state.update(status='complete_fixed_original_factual_episode',source_sha256_after=identity.source_digest(C),ended_utc=stamp(),success_flag=bool(env.eval_success),termination='native_success' if env.eval_success else (f"official_native_{plan['max_native_commands']}_limit" if is_native else 'fixed_600_command_cap'),probe_status={str(q):'sealed_before_execution' if q in captured else 'not_reached_no_replacement' for q in sorted(capture_queries)},independent_candidate_labels=0,independent_source_episodes=1,original_DEV_selected_step='unverified',original_accuracy_reproduction=False)
  assert state['source_sha256_after']==CORE_SHA
  save();print(json.dumps(state),flush=True)
 except BaseException as exc:
  state.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc(),ended_utc=stamp());save();print(json.dumps(state),flush=True);raise
 finally:
  if topp_observer is not None:topp_observer.__exit__(None,None,None)
  if hook is not None:hook.remove()
  if policy is not None:
   try:policy.close()
   except Exception:pass
  if env is not None:
   try:env.close_env()
   except Exception:pass
  if old_git is not None:legacy._git_identity=old_git
  if 'mp' in locals():
   for child in mp.active_children():
    if child.pid not in children_before:
     child.terminate();child.join(timeout=5)

if __name__=='__main__':main()
