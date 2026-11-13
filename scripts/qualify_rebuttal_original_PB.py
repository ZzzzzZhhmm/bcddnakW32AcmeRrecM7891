"""Qualify the original PB12000 model on its fixed first state;engineering only.

No Git remote access, worktree gate, training, outcome-based seed replacement,
or published-accuracy reproduction. The independent later trace must reproduce
this initial observation, first model output, and four physical commands exactly.
"""
from pathlib import Path
import argparse,dataclasses,datetime,hashlib,importlib,importlib.util,json,os,sys,traceback

spec=importlib.util.spec_from_file_location('PB_namespace_binding',Path(__file__).resolve().parents[1]/'src/fastwam/research/pb_namespace.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)

P=Path('/mnt/afs/task3_2/L202500276_lwz/projects/WARM_evaluations/rebuttal_20261109_acp')
C=P/'historical_specialist_source_47277e5/code';H=P/'controller_eb70b41/code'
CORE_SHA='4a1d05cb548f16bf6a9b239082b02d4697aa8e8c1f43c2f5d8ffbaec196462fd'
COMMIT='47277e58624366506ee032f4ad27560c7e4bdd19'
SIM_SHA='a97e308ca053c3aeb1bfec8e3c4c4f550aac7f6ae753e36bbce82e9522292379'

PB_PROTOCOL=dict(schema='warm.original-PB-first-state-qualification.v1',task_name='press_button',qualified_original_run='press_button-s3407-v2',checkpoint_step=12000,root_seed=3407,physical_seed=0,native_limit=1500,qualification_queries=1,qualification_commands=4,checkpoint_sha256='a765d987b2df0d228782980592c39aaed0e837040c7f6f48c905e4641f79c285')

def validate_qualification_plan(plan):
 expected=dict(PB_PROTOCOL)
 if plan.get('schema')==N.SCHEMAS['qualification']:
  _,entry=N.validate_execution_binding(plan,'qualification');expected.update(schema=N.SCHEMAS['qualification'],physical_seed=entry['seed'])
 if any(type(plan.get(k)) is not type(v) or plan.get(k)!=v for k,v in expected.items()):raise ValueError('Original PB task/run/checkpoint/seed/horizon or engineering scope differs')
 if plan.get('original_source_sha256')!=CORE_SHA or plan.get('original_source_commit')!=COMMIT:raise ValueError('Original PB frozen model source differs')
 return plan

def press_button_task_state(env):
 # The official check_success() changes counters and resets buttons.
 # Audit stored state without invoking it or exposing hidden state to the policy.
 names=('card_id_1','card_id_2','press_cnt_1','press_cnt_2','press_cnt_check_button',
        'press_flag_1','press_flag_2','press_flag_check_button')
 values={n:getattr(env,n) for n in names}
 return {n:v.item() if hasattr(v,'item') else v for n,v in values.items()}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--expected-plan-sha256',required=True);parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--admission',type=Path);parser.add_argument('--expected-admission-sha256');cli=parser.parse_args()
 sha_plan=hashlib.sha256(cli.plan.read_bytes()).hexdigest();assert sha_plan==cli.expected_plan_sha256
 binding=validate_qualification_plan(json.loads(cli.plan.read_text()))
 is_namespace=binding['schema']==N.SCHEMAS['qualification'];entry=N.bind_case(binding)[1] if is_namespace else None
 for item in binding['small_inputs']:
  assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
 pre=P/'ITER29_ORIGINAL_PB12000_RUNTIME_STATIC_PREFLIGHT_V1';v=json.loads((pre/'execution.json').read_text());args_static=json.loads((pre/'runtime_args.json').read_text())
 assert v['checkpoint_sha256']==PB_PROTOCOL['checkpoint_sha256'] and v['static_contracts_passed'] and v['encoder_runtime_matches']
 assert args_static['task_name']=='press_button' and args_static['seed']==3407 and args_static['action_horizon']==32 and args_static['replan_steps']==4
 out=Path(binding['output']) if is_namespace else P/'original_pb12000_first_state_qualification_v1'
 if cli.preflight_only:
  assert not out.exists();print(json.dumps(dict(status='PB_fixed_binding_preflight_only',model_inference_queries=0,model_updates=0,physics_commands=0,plan_sha256=sha_plan)));return
 assert cli.admission and cli.expected_admission_sha256 and hashlib.sha256(cli.admission.read_bytes()).hexdigest()==cli.expected_admission_sha256
 admission=json.loads(cli.admission.read_text());assert admission['qualification_plan_sha256']==sha_plan and admission['original_windows_unchanged'] and admission['schema']=='warm.original-PB-new-stage-admission.v1'
 deadline=datetime.datetime.fromisoformat(admission['absolute_deadline_utc']);assert datetime.datetime.now(datetime.timezone.utc)<deadline
 import fcntl
 lease=(Path(binding['native_worker_lease']) if is_namespace else P/'original_specialist_native_execution.worker.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not out.exists();out.mkdir()
 stamp=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 state=dict(status='binding',started_utc=stamp(),plan_sha256=sha_plan,admission_sha256=cli.expected_admission_sha256,absolute_deadline_utc=admission['absolute_deadline_utc'],scope='Original-run fixed PB12000/source47277 root3407/physical0:one native live Full replan and four commands only. Not original DEV selection,a complete episode,repair,withdrawal,success rate or candidate-effect accuracy.',inference_queries=0,native_commands=0,model_updates=0)
 if is_namespace:
  state.update(namespace_case_id=entry['case_id'],scope=f"Original-run PB12000 namespace case{entry['case_id']}/physical seed{entry['seed']}/root3407:one native Full replan and four commands only;engineering qualification,not a complete episode or efficacy result.")
 def save():
  state['updated_utc']=stamp();t=out/'operator.json.partial';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(out/'operator.json')
 save();env=None;policy=None;children_before=set();hook=None;old_git=None;topp_observer=None
 try:
  spec=importlib.util.spec_from_file_location('stdlib_identity',H/'scripts/run_rebuttal_training_with_runtime_trace.py');identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
  assert identity.source_digest(C)==CORE_SHA and identity.source_digest(H)=='7a0ce8943818b02a7ee493fc3b6d666ba6b55b5362df7ab5f3a6be4c760000ae'
  preflight=P/'ITER29_ORIGINAL_PB12000_RUNTIME_STATIC_PREFLIGHT_V1';qualified=json.loads((preflight/'execution.json').read_text());assert qualified['static_contracts_passed'] and qualified['encoder_runtime_matches']
  args=json.loads((preflight/'runtime_args.json').read_text());args['warm_telemetry_path']=str(out/'full-telemetry.jsonl')
  (out/'runtime_args.json').write_text(json.dumps(args,indent=2)+'\n')
  plan=json.loads((P/'resumable_rb_cases0003_0009_v1_plan.json').read_text());runtime=Path(binding['rmbench_runtime']) if is_namespace else P.parent/'rebuttal_20261108_p0/rmbench_runtime'
  os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='true',DIFFSYNTH_MODEL_BASE_PATH=plan['model_root'],WANDB_MODE='disabled',WARM_IMMUTABLE_ASSET_HASH_CACHE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
  sys.path[:0]=[str(C/'src'),str(C),plan['curobo_source']]
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
  instruction_files=[Path('description/utils/generate_episode_instructions.py'),Path('description/task_instruction/press_button.json')]
  for relative in instruction_files:assert sha(runtime/relative)==sha(official/relative),str(relative)
  state['instruction_source_sha256']={str(f):sha(runtime/f) for f in instruction_files}
  protocol=np.load(args['warm_initial_states_path'],allow_pickle=False);assert protocol.dtype.kind in 'iu' and protocol.size>0
  physical_seed=entry['seed'] if is_namespace else int(protocol.reshape(-1)[0]);assert physical_seed==binding['physical_seed']
  state.update(candidate_seed=physical_seed,seed_protocol_sha256=sha256_array(protocol),seed_selection='First scalar of original candidate namespace;no outcome-based replacement.')
  cfg=setup.resolved_config(runtime,args['task_name'],physical_seed)
  if is_namespace:cfg['now_ep_num']=entry['case_id'];state['seed_selection']='First-two prospectively expert-admitted namespace cases;not flattened ordinal or policy-outcome selection.'
  (out/'native_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
  children_before={x.pid for x in mp.active_children()};env=getattr(importlib.import_module('envs.'+args['task_name']),args['task_name'])()
  state['status']='fixed_seed_expert_setup_only';save();print('stage=fixed_seed_expert_setup_only',flush=True)
  random.seed(physical_seed);np.random.seed(physical_seed);torch.manual_seed(physical_seed);torch.cuda.manual_seed_all(physical_seed)
  env.setup_demo(**cfg);assert env.step_lim==1500;expert_info=entry['expert_info'] if is_namespace else env.play_once()
  accepted=True if is_namespace else bool(env.plan_success and env.check_success());state['fixed_expert_setup_accepted']=accepted
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
  if is_namespace:
   native=[dict(name=art.name,qpos=art.get_qpos().tolist(),qvel=art.get_qvel().tolist()) for art in env.scene.get_all_articulations()]
   state.update(N.require_initial(binding,obs,instruction,native,press_button_task_state(env)))
   state['scope']='One original-PB12000 replan and four native commands on a prospectively fixed expert-admitted namespace case;engineering qualification only.'
  np.savez_compressed(out/'initial_observation.npz',**raw,native_qpos=qpos)
  (out/'initial_identity.json').write_text(json.dumps(dict(instruction=instruction,cameras={k:sha256_array(v) for k,v in raw.items()},qpos_sha256=sha256_array(qpos),task=args['task_name'],candidate_seed=physical_seed,task_state_audit_only=press_button_task_state(env)),indent=2)+'\n')
  actual_runtime=legacy.current_encoder_runtime(args['warm_dino_device']);(out/'actual_encoder_runtime_before_query.json').write_text(json.dumps(actual_runtime,indent=2)+'\n');assert sha256_canonical_json(actual_runtime)==policy.contract.encoder_runtime_sha256
  def cpu(value):
   if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
   if dataclasses.is_dataclass(value):return {f.name:cpu(getattr(value,f.name)) for f in dataclasses.fields(value)}
   if isinstance(value,dict):return {str(k):cpu(v) for k,v in value.items()}
   if isinstance(value,(tuple,list)):return [cpu(v) for v in value]
   if isinstance(value,(str,int,float,bool,type(None))):return value
   return dict(unserialized_type=type(value).__module__+'.'+type(value).__name__)
  captures=[]
  def event_hook(_module,_inputs,result):
   value=cpu(result);captures.append(value);torch.save(value,out/f'event_adapter_{len(captures):04d}.pt')
  hook=policy.model.retrospective_event_adapter.register_forward_hook(event_hook)
  old_infer=policy.model.infer_action
  def infer_observer(*a,**kw):
   if state['inference_queries']!=0:raise RuntimeError('Only one fixed replan admitted')
   value=old_infer(*a,**kw);torch.save(cpu(value),out/'original_model_output.pt');state['inference_queries']+=1;return value
  policy.model.infer_action=infer_observer
  if is_namespace:policy._next_episode_index=entry['case_id']
  policy.reset();state['status']='one_original_replan';save();print('stage=one_original_replan',flush=True);policy._replan(obs,env)
  assert state['inference_queries']==1;torch.save(captures,out/'event_adapter_outputs.pt')
  state.update(status='executing_fixed_prefix',event_hook_calls=len(captures),hook_return_modified=False);save()
  observer_source=P/'matched_reset_controller_9c0ea3c/code/scripts/run_rebuttal_matched_native.py'
  assert sha(observer_source)=='49888eb2854b29d6f5f384e394aaf91847dd9ebfdf9e89f6bc51fb1f8152dd4d'
  observer_spec=importlib.util.spec_from_file_location('bound_PB_TOPP_observer',observer_source);observer=importlib.util.module_from_spec(observer_spec);observer_spec.loader.exec_module(observer)
  topp_observer=observer.TOPPObserver(env);topp_observer.__enter__()
  def measured():return [dict(name=art.name,qpos=art.get_qpos().tolist(),qvel=art.get_qvel().tolist()) for art in env.scene.get_all_articulations()]
  trace_before=[];trace_after=[];physical=[]
  for i in range(4):
   assert datetime.datetime.now(datetime.timezone.utc)<deadline
   before=measured();task_before=press_button_task_state(env);topp_observer.errors.clear();previous=int(env.take_action_cnt)
   trace_before.append(legacy.factual_joint_state(env.get_obs()));policy.step(env,env.get_obs());trace_after.append(legacy.factual_joint_state(env.get_obs()))
   assert int(env.take_action_cnt)==previous+1
   physical.append(dict(index=i,physical_before=before,physical_after=measured(),task_state_before=task_before,task_state_after=press_button_task_state(env),observed_TOPP_errors=topp_observer.errors.copy()))
   state['native_commands']+=1;save()
  (out/'native_prefix_physical.json').write_text(json.dumps(physical,indent=2)+'\n')
  np.savez_compressed(out/'native_prefix_qpos.npz',before=np.stack(trace_before),after=np.stack(trace_after))
  assert state['inference_queries']==1 and state['native_commands']==4
  state.update(status='complete_one_original_replan_four_native_commands',action_output_finite=bool(torch.isfinite(torch.load(out/'original_model_output.pt',map_location='cpu',weights_only=False)['action']).all().item()),peak_pytorch_allocated_bytes=torch.cuda.max_memory_allocated(),source_sha256_after=identity.source_digest(C),task_success_flag_after_prefix=bool(env.eval_success),success_rate_eligible=False,independent_candidate_labels=0,ended_utc=stamp());assert state['source_sha256_after']==CORE_SHA;save();print(json.dumps(state),flush=True)
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
