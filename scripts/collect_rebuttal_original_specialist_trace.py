"""Collect a frozen original-specialist factual trace and pre-execution probes.

This is a supplementary fixed-point study, not recovery of the submitted DEV
selection. Original model mathematics and validators remain unchanged. The
collector never accesses a Git remote or tests worktree cleanliness. All output
and raw checkpoints stay outside Git. A failed fixed seed is never replaced.
"""
from pathlib import Path
import argparse,dataclasses,datetime,hashlib,importlib,importlib.util,json,os,pickle,sys,traceback

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--plan',required=True,type=Path)
 parser.add_argument('--expected-plan-sha256',required=True)
 cli=parser.parse_args();assert hashlib.sha256(cli.plan.read_bytes()).hexdigest()==cli.expected_plan_sha256
 plan=json.loads(cli.plan.read_text());assert plan['schema']=='warm.original-specialist-factual-trace.v1'
 P=Path(plan['evidence_root']);C=Path(plan['original_source']);H=Path(plan['reference_controller'])
 CORE_SHA=plan['original_source_sha256'];COMMIT=plan['original_source_commit'];SIM_SHA=plan['simulator_source_sha256']
 out=Path(plan['output']);assert out.parent==P and not out.exists();out.mkdir()
 assert plan['physical_seed']==0 and plan['max_native_commands']==600 and plan['capture_replans']==[8,48,96]
 assert plan['qualified_original_run']=='blocks_ranking_try-s3407-v1' and plan['checkpoint_step']==14000
 qualification=Path(plan['qualification_root']);qualified_operator=json.loads((qualification/'operator.json').read_text())
 assert qualified_operator['status']=='complete_one_original_replan_four_native_commands'
 assert hashlib.sha256((qualification/'operator.json').read_bytes()).hexdigest()==plan['qualification_operator_sha256']
 deadline=datetime.datetime.fromisoformat(plan['absolute_deadline_utc'])
 assert datetime.datetime.now(datetime.timezone.utc)<deadline

 stamp=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 state=dict(status='binding',started_utc=stamp(),scope=plan['claim'],plan_sha256=cli.expected_plan_sha256,planned_commands=600,planned_probe_replans=plan['capture_replans'],inference_queries=0,native_commands=0,model_updates=0)
 def save():
  state['updated_utc']=stamp();t=out/'operator.json.partial';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(out/'operator.json')
 save();env=None;policy=None;children_before=set();hook=None;old_git=None
 try:
  spec=importlib.util.spec_from_file_location('stdlib_identity',H/'scripts/run_rebuttal_training_with_runtime_trace.py');identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
  assert identity.source_digest(C)==CORE_SHA and identity.source_digest(H)=='7a0ce8943818b02a7ee493fc3b6d666ba6b55b5362df7ab5f3a6be4c760000ae'
  preflight=Path(plan['runtime_preflight']);qualified=json.loads((preflight/'execution.json').read_text());assert qualified['static_contracts_passed'] and qualified['encoder_runtime_matches']
  args=json.loads((preflight/'runtime_args.json').read_text());args['warm_telemetry_path']=str(out/'full-telemetry.jsonl')
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
  instruction_files=[Path('description/utils/generate_episode_instructions.py'),Path('description/task_instruction/blocks_ranking_try.json')]
  for relative in instruction_files:assert sha(runtime/relative)==sha(official/relative),str(relative)
  state['instruction_source_sha256']={str(f):sha(runtime/f) for f in instruction_files}
  protocol=np.load(args['warm_initial_states_path'],allow_pickle=False);assert protocol.dtype.kind in 'iu' and protocol.size>0
  physical_seed=int(protocol.reshape(-1)[0]);assert physical_seed==plan['physical_seed']==0
  state.update(candidate_seed=physical_seed,seed_protocol_sha256=sha256_array(protocol),seed_selection='First scalar of original candidate namespace;no outcome-based replacement.')
  cfg=setup.resolved_config(runtime,args['task_name'],physical_seed);(out/'native_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
  children_before={x.pid for x in mp.active_children()};env=getattr(importlib.import_module('envs.'+args['task_name']),args['task_name'])()
  state['status']='fixed_seed_expert_setup_only';save();print('stage=fixed_seed_expert_setup_only',flush=True)
  random.seed(physical_seed);np.random.seed(physical_seed);torch.manual_seed(physical_seed);torch.cuda.manual_seed_all(physical_seed)
  env.setup_demo(**cfg);expert_info=env.play_once()
  accepted=bool(env.plan_success and env.check_success());state['fixed_expert_setup_accepted']=accepted
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
  np.savez_compressed(out/'initial_observation.npz',**raw,native_qpos=qpos)
  with np.load(qualification/'initial_observation.npz',allow_pickle=False) as previous:
   assert set(previous.files)==set(raw)|{'native_qpos'}
   assert all(np.array_equal(previous[k],v) for k,v in {**raw,'native_qpos':qpos}.items()),'Initial native observation differs from qualification'
  state['initial_observation_exact_to_qualification']=True
  (out/'initial_identity.json').write_text(json.dumps(dict(instruction=instruction,cameras={k:sha256_array(v) for k,v in raw.items()},qpos_sha256=sha256_array(qpos),task=args['task_name'],candidate_seed=physical_seed),indent=2)+'\n')
  actual_runtime=legacy.current_encoder_runtime(args['warm_dino_device']);(out/'actual_encoder_runtime_before_query.json').write_text(json.dumps(actual_runtime,indent=2)+'\n');assert sha256_canonical_json(actual_runtime)==policy.contract.encoder_runtime_sha256
  def cpu(value):
   if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
   if dataclasses.is_dataclass(value):return {f.name:cpu(getattr(value,f.name)) for f in dataclasses.fields(value)}
   if isinstance(value,dict):return {str(k):cpu(v) for k,v in value.items()}
   if isinstance(value,(tuple,list)):return [cpu(v) for v in value]
   if isinstance(value,(str,int,float,bool,type(None))):return value
   return dict(unserialized_type=type(value).__module__+'.'+type(value).__name__)
  capture_queries=set(plan['capture_replans']);generations=out/'generations';generations.mkdir()
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
  policy.model.infer_action=infer_observer;policy.reset()
  state['status']='collecting_fixed_original_factual_episode';save()
  with np.load(qualification/'native_prefix_qpos.npz',allow_pickle=False) as previous:
   prefix_before=previous['before'].copy();prefix_after=previous['after'].copy()
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
    if not policy.queue:
     np.savez_compressed(out/f'replan-{index//4:04d}.npz',**legacy.factual_cameras(obs),native_qpos=before)
     policy._replan(obs,env)
    policy.step(env,obs);after=legacy.factual_joint_state(env.get_obs()).copy()
    assert len(command_capture)==index+1
    row=dict(index=index,command=command_capture[-1].tolist(),command_dtype=str(command_capture[-1].dtype),before=before.tolist(),after=after.tolist(),eval_success=bool(env.eval_success),take_action_cnt=int(env.take_action_cnt))
    stream.write(json.dumps(row)+'\n');stream.flush()
    state['native_commands']+=1
    if index<4:assert np.array_equal(before,prefix_before[index]) and np.array_equal(after,prefix_after[index]),'Qualified native prefix changed'
    state.update(last_frame=index,captured_probe_replans=sorted(captured),success_flag=bool(env.eval_success));save()
    if env.eval_success:break
  state.update(status='complete_fixed_original_factual_episode',source_sha256_after=identity.source_digest(C),ended_utc=stamp(),success_flag=bool(env.eval_success),termination='native_success' if env.eval_success else 'fixed_600_command_cap',probe_status={str(q):'sealed_before_execution' if q in captured else 'not_reached_no_replacement' for q in sorted(capture_queries)},independent_candidate_labels=0,independent_source_episodes=1,original_DEV_selected_step='unverified',original_accuracy_reproduction=False)
  assert state['source_sha256_after']==CORE_SHA
  save();print(json.dumps(state),flush=True)
 except BaseException as exc:
  state.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc(),ended_utc=stamp());save();print(json.dumps(state),flush=True);raise
 finally:
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
