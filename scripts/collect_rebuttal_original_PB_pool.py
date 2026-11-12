"""Execute the pinned original PB proposal pool from independently reset parents.

The GPU run consumes a CPU-sealed candidate set. It never generates new policy
actions, changes task counters for an observation, unpacks physics, or uses Git.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib,importlib.util,json,multiprocessing as mp,os,pickle,random,sys,traceback

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
B=load('pinned_PB_mechanism',Path(__file__).resolve().parents[1]/'src/fastwam/research/pb_mechanism.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(p):return json.loads(Path(p).read_text())
def write(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False));t.replace(p)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--expected-plan-sha256',required=True);ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--admission',type=Path);ap.add_argument('--expected-admission-sha256');ap.add_argument('--qualify-only',action='store_true');ap.add_argument('--resume-known-terminal',action='store_true');ap.add_argument('--continuation-manifest',type=Path);ap.add_argument('--expected-continuation-sha256');cli=ap.parse_args();assert sha(cli.plan)==cli.expected_plan_sha256;p=B.validate_plan(rd(cli.plan))
 import fcntl
 C,H,runtime,source,out=[Path(p[k]) for k in ['original_source','reference_controller','rmbench_runtime','factual_source','output']];assert not out.is_relative_to(C) and not out.is_relative_to(H);out.mkdir(exist_ok=True)
 lease=(out/'campaign.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 op=out/('preflight.json' if cli.preflight_only else 'operator.json')
 if op.exists():
  assert not cli.preflight_only and cli.resume_known_terminal
  old=rd(op);assert old['status']=='qualified_PB_query_AABA_pool_pending','Unknown,failed or active prior invocation needs separate archived adjudication'
  B.immutable_json(out/'prior_AABA_operator.json',old)
 elif cli.resume_known_terminal:raise ValueError('No prior terminal invocation to resume')
 state=dict(status='binding',started_utc=stamp(),plan_sha256=cli.expected_plan_sha256,query=p['query_replan'],selected_rank=p['selected_rank'],new_policy_generations=0,model_updates=0,complete_candidate_endpoints=0)
 def save():state['updated_utc']=stamp();write(op,state)
 save();env=None;teacher=None;children_before={c.pid for c in mp.active_children()}
 try:
  for v in p['inputs']:assert sha(v['path'])==v['sha256'],v['path']
  assert sha(p['source_archive'])==p['source_archive_sha256']
  os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='true',DIFFSYNTH_MODEL_BASE_PATH=p['model_root'],TOKENIZERS_PARALLELISM='false')
  sys.path[:0]=[str(C/'src'),str(C)];import fastwam;fastwam.__path__.append(str(H/'src/fastwam'));import fastwam.research;fastwam.research.__path__.insert(0,str(H/'src/fastwam/research'))
  import numpy as np,torch
  from fastwam.memory.manifest import sha256_array
  from fastwam.memory.robotwin_artifacts import RobotwinQposZScore
  from fastwam.models.warm.retrospection_model import _mean_effect
  from experiments.rmbench.warm_policy.policy_core import factual_cameras,factual_joint_state
  from fastwam.research.candidate_pool import decode_branch_actions,centered_effect_metrics
  from fastwam.research.factual_effect_reader import FrozenFactualEffectReader
  from fastwam.research.fresh_branches import OfficialQposGuard
  from fastwam.research.fresh_pool import read_verified_branch,seal_branch
  from fastwam.research.native_replay import native_parent,state_hashes,physical_joint_state,require_observation
  from fastwam.research.branches import InvalidBranchExecution
  identity=load('PB_reference_identity',H/'scripts/run_rebuttal_training_with_runtime_trace.py');assert identity.source_digest(C)==B.CORE_SHA256 and identity.source_digest(H)==p['reference_controller_sha256']
  assert sha(C/'src/fastwam/models/warm/retrospection_model.py')==p['original_model_module_sha256']
  for module in ['fastwam.memory.online_retrieval','fastwam.models.warm.retrospection_model','experiments.rmbench.warm_policy.policy_core']:assert Path(sys.modules[module].__file__).resolve().is_relative_to(C)
  setup=load('PB_native_setup',H/'scripts/qualify_rebuttal_rmbench_task.py');assert setup.source_identity(runtime)['sha256']==p['simulator_source_sha256']
  full=rd(source/'operator.json');trace=[json.loads(x) for x in (source/'native_trace.jsonl').read_text().splitlines()];q=p['query_replan'];selected=p['selected_rank'];frame=q*4;B.validate_source(full,trace,q)
  if any(r.get('observed_TOPP_errors') for r in trace[:frame]):raise ValueError('Recorded factual prefix has TOPP rejections;requires a separate exact two-pass rejection audit before branch truth')
  args=rd(source/'runtime_args.json');contract=rd(args['warm_online_contract_path']);assert args['task_name']=='press_button' and args['seed']==3407 and contract['warm_checkpoint_sha256']==B.WEIGHT_SHA256
  probe=source/f'probes/q{q:04d}';event=torch.load(probe/'event_adapter.pt',map_location='cpu',weights_only=False);model=torch.load(probe/'model_output.pt',map_location='cpu',weights_only=False)
  valid=event['candidate_valid_mask'].numpy()[0];assert valid.shape==(32,) and valid.dtype==np.bool_ and valid[selected];assert int(model['warm_retrospection']['selected_candidate_index'])==selected
  adapted=event['adapted_action_mean'].float().numpy()[0];predicted=event['predicted_effect'].float().numpy()[0];assert adapted.shape==(32,32,14) and predicted.shape==(32,768)
  normalizer=RobotwinQposZScore.from_dataset_stats(rd(args['dataset_stats_path']));refined=decode_branch_actions(normalizer,np.load(source/f'generations/q{q:04d}/model_actions.npy',allow_pickle=False));commands=np.asarray([r['command'] for r in trace],dtype=np.float32)
  assert np.array_equal(refined[:4],commands[frame:frame+4]);native={f'rank-{r}':decode_branch_actions(normalizer,adapted[r]) for r in np.flatnonzero(valid)}
  order=['anchor-a','anchor-aa',f'rank-{selected}','anchor-qual-final']+[f'rank-{r}' for r in np.flatnonzero(valid) if r!=selected]+['anchor-pool-final'];actions={n:refined if n.startswith('anchor-') else native[n] for n in order}
  bank=Path(args['warm_bank_directory']);bm=rd(bank/'manifest.json');assert sha(bank/'manifest.json')==contract['bank_manifest_sha256'] and sha(bank/bm['payload_file'])==bm['content_hashes'][bm['payload_file']]
  rows=np.asarray(model['warm_online_telemetry']['retrieval']['bank_rows'],dtype=np.int64);ids=model['warm_online_telemetry']['retrieval']['ranked_event_ids']
  with np.load(bank/bm['payload_file'],allow_pickle=False) as z:
   offsets=z['_event_dataset_id_offsets'];utf8=z['_event_dataset_id_utf8']
   for j,r in enumerate(rows):
    if valid[j]:assert dict(dataset_id=bytes(utf8[offsets[r]:offsets[r+1]]).decode(),dataset_index=int(z['_event_dataset_index'][r]),episode_index=int(z['_event_episode_index'][r]),start_frame=int(z['_event_start_frame'][r]))==ids[j]
   historical=_mean_effect(torch.from_numpy(z['effect_post'][rows]-z['effect_pre'][rows])).float().numpy()
  assert historical.shape==(32,768)
  seal=dict(query=q,frame=frame,selected_rank=selected,descriptor='original_uniform_mean_DINO_delta_v1',source_plan_sha256=B.TRACE_PLAN_SHA256,checkpoint_sha256=B.WEIGHT_SHA256,rubric=B.RUBRIC,before_any_new_branch_truth=True,branch_order=order,original_valid_ranks=np.flatnonzero(valid).tolist(),action_hashes={n:sha256_array(a) for n,a in actions.items()},predicted_sha256=sha256_array(predicted),historical_sha256=sha256_array(historical),source_inputs=p['inputs'],primary_horizon=32,secondary_prefix=4,required_effect_available=False,population_success_rate_available=False)
  sf=out/'sealed_query.json'
  if sf.exists():
   assert rd(sf)==seal
   with np.load(out/'sealed_arrays.npz',allow_pickle=False) as z:
    assert np.array_equal(z['predicted'],predicted) and np.array_equal(z['historical'],historical) and np.array_equal(z['valid'],valid) and all(np.array_equal(z[n],a) for n,a in actions.items())
  else:np.savez_compressed(out/'sealed_arrays.npz',predicted=predicted,historical=historical,valid=valid,**actions);write(sf,seal)
  continuation=None;Q=None
  if cli.continuation_manifest:
   assert cli.expected_continuation_sha256 and not cli.qualify_only and not cli.resume_known_terminal
   Q=load('PB_remaining_branch_adoption',Path(__file__).resolve().parents[1]/'src/fastwam/research/pb_continuation.py');continuation=Q.bind_manifest(cli.continuation_manifest,cli.expected_continuation_sha256,p,order,sha(sf));Q.adopt_completed(continuation,out,query_id=f'original-pb12000-seed0-q{q}',verifier=read_verified_branch);continued_anchor=read_verified_branch(out/'branches/anchor-a/attempt-001',query_id=f'original-pb12000-seed0-q{q}',branch_id='anchor-a');continued_repeat=read_verified_branch(out/'branches/anchor-aa/attempt-001',query_id=f'original-pb12000-seed0-q{q}',branch_id='anchor-aa');Q.require_old_full_repeats(continued_anchor,continued_repeat);B.immutable_json(out/'continuation_receipt.json',dict(manifest_sha256=cli.expected_continuation_sha256,adopted_completed=list(continuation['completed']),retained_unknown=list(continuation['unknown']),remaining_original_order=continuation['unstarted'],new_boundary_full_controls=[Q.PRE_CONTROL,Q.NEW_CONTROL],new_control_action_sha256=seal['action_hashes']['anchor-a'],original_AABA_complete=all(n in continuation['completed'] for n in order[:4]),old_clock_reset=False));state['continuation_manifest_sha256']=cli.expected_continuation_sha256;order=[Q.PRE_CONTROL]+order+[Q.NEW_CONTROL];actions[Q.PRE_CONTROL]=refined;actions[Q.NEW_CONTROL]=refined
  if cli.preflight_only:
   assert not torch.cuda.is_initialized();state.update(status='complete_PB_CPU_input_seal',CUDA_initialized=False,branch_order=order,candidate_branches_executed=0);save();return
  assert cli.admission and cli.expected_admission_sha256 and sha(cli.admission)==cli.expected_admission_sha256
  admission=rd(cli.admission);assert admission['schema']=='warm.original-PB-physical-pool-admission.v1' and admission['plan_sha256s'][str(cli.plan)]==cli.expected_plan_sha256
  deadline=datetime.datetime.fromisoformat(admission['absolute_deadline_utc']);assert datetime.datetime.now(datetime.timezone.utc)<deadline
  def budget():
   if datetime.datetime.now(datetime.timezone.utc)>=deadline:raise TimeoutError('Fixed PB physical-pool window expired;preserve original clock')
  guard_lease=(Path(p['evidence_root'])/'original_specialist_native_execution.worker.lock').open('a');fcntl.flock(guard_lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch.manual_seed(3407);torch.cuda.manual_seed_all(3407);torch.cuda.set_per_process_memory_fraction(.12);teacher=FrozenFactualEffectReader.from_source_runtime(args,contract);B.immutable_json(out/'reader_receipt.json',teacher.receipt)
  assets=rd(Path(p['evidence_root'])/'resumable_rb_cases0003_0009_v1_plan.json');sys.path[:0]=[str(runtime),str(runtime/'description/utils'),assets['curobo_source']];os.chdir(runtime)
  from generate_episode_instructions import generate_episode_descriptions
  cls=getattr(importlib.import_module('envs.press_button'),'press_button');cfg=setup.resolved_config(runtime,'press_button',0);assert cfg==rd(source/'native_config.json');expert=rd(source/'expert_setup_metadata.json');initial=rd(source/'initial_identity.json');rng=pickle.loads((probe/'sampling_rng_before.pkl').read_bytes());query_id=f'original-pb12000-seed0-q{q}'
  def source_observation(path,obs):
   with np.load(path,allow_pickle=False) as z:require_observation({k:z[k] for k in z.files},{**factual_cameras(obs),'native_qpos':factual_joint_state(obs)})
  def physical():return [{k:v.tolist() if hasattr(v,'tolist') else v for k,v in r.items()} for r in physical_joint_state(env)]
  def parent_scene(label):
   nonlocal env
   budget();random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);env=cls();env.setup_demo(**cfg);assert env.step_lim==1500 and type(env.scene.physx_system).__name__=='PhysxCpuSystem'
   text=str(np.random.choice(generate_episode_descriptions('press_button',[expert['info']],100)[0]['unseen']));assert text==initial['instruction'];env.set_instruction(instruction=text);source_observation(source/'initial_observation.npz',env.get_obs());prefix=[]
   with OfficialQposGuard(env) as guard:
    for i,command in enumerate(commands[:frame]):
     budget();before=factual_joint_state(env.get_obs()).copy();p0=physical();s0=B.observe(env)['native_state'];assert guard.step(command);after=factual_joint_state(env.get_obs()).copy();s1=B.observe(env)['native_state'];p1=physical();row=trace[i]
     assert np.array_equal(before,np.asarray(row['before'],dtype=before.dtype)) and np.array_equal(after,np.asarray(row['after'],dtype=after.dtype)) and int(env.take_action_cnt)==i+1
     assert p0==row['physical_before'] and p1==row['physical_after'] and s0==row['task_state_before'] and s1==row['task_state_after']
     prefix.append(dict(index=i,before=before.tolist(),after=after.tolist(),physical_before=p0,physical_after=p1,task_state_before=s0,task_state_after=s1))
     if (i+1)%4==0:source_observation(source/f'replan-{(i+1)//4:04d}.npz',env.get_obs())
   source_observation(source/f'replan-{q:04d}.npz',env.get_obs());random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch_cpu']);torch.cuda.set_rng_state_all(rng['torch_cuda']);parent=native_parent(env,env.get_obs());ph=state_hashes(parent);B.immutable_json(out/(label+'_prefix.json'),prefix);return parent,ph
  # Two complete native-prefix passes qualify source replay before any candidate endpoint.
  prefix_parents=[]
  for i in range(2):
   _,ph=parent_scene(f'qualification-prefix-{i}');prefix_parents.append(ph);env.close_env(clear_cache=True);env=None
  assert prefix_parents[0]==prefix_parents[1];B.immutable_json(out/'prefix_qualification.json',dict(status='two_exact_native_prefixes_before_branch_truth',query=q,commands_each=frame,parent_hashes=prefix_parents[0],candidate_endpoints=0))
  records={}
  for name in order:
   budget()
   if continuation and name in continuation['unknown']:
    records[name]=Q.unknown_record(continuation,name,query_id);B.immutable_json(out/(name+'_retained_unknown.json'),records[name]);state.update(status='retaining_old_unknown_without_rerun',completed_branches=list(records));save();continue
   branch=out/'branches'/name;branch.mkdir(parents=True,exist_ok=True);done=list(branch.glob('attempt-*/complete.json'));assert len(done)<=1
   if done:
    row=read_verified_branch(done[0].parent,query_id=query_id,branch_id=name);assert row['sealed_query_sha256']==sha(sf) and row['parent_hashes']==prefix_parents[0]
   else:
    existing=list(branch.glob('attempt-*'));assert not existing,'An unresolved physical branch must be archived/adjudicated before any retry';attempt=branch/'attempt-001';attempt.mkdir();write(attempt/'attempt.json',dict(branch=name,actual_pid=os.getpid(),utc=stamp()));state.update(status='recreating_parent',active_branch=name);save();parent,ph=parent_scene(name);assert ph==prefix_parents[0]
    (attempt/'parent.pickle').write_bytes(pickle.dumps(parent,protocol=4));parent_progress=B.observe(env);before_feature=teacher.encode(env.get_obs());assert state_hashes(native_parent(env,env.get_obs()))==ph;np.save(attempt/'parent_dino_spatial.npy',before_feature,allow_pickle=False)
    count=0;status='complete_horizon';error=None;prefix4=None;state['status']='executing_sealed_PB_branch';save()
    with (attempt/'commands.jsonl').open('x',buffering=1) as stream,OfficialQposGuard(env) as guard:
     for i,command in enumerate(actions[name]):
      budget();before=factual_joint_state(env.get_obs()).copy();p0=physical();s0=B.observe(env)
      try:accepted=guard.step(command)
      except InvalidBranchExecution as exc:status='invalid_branch';error=str(exc);break
      if not accepted:status='invalid_branch';error='official native executor stopped';break
      count+=1;p1=physical();s1=B.observe(env);stream.write(json.dumps(dict(index=i,command=command.tolist(),physical_before=p0,physical_after=p1,task_before=s0,task_after=s1))+'\n')
      if name.startswith('anchor-') and i<4:
       r=trace[frame+i];assert p0==r['physical_before'] and p1==r['physical_after'] and s0['native_state']==r['task_state_before'] and s1['native_state']==r['task_state_after'];after=factual_joint_state(env.get_obs());assert np.array_equal(before,np.asarray(r['before'],dtype=before.dtype)) and np.array_equal(after,np.asarray(r['after'],dtype=after.dtype))
      if i==3:
       before_read=state_hashes(native_parent(env,env.get_obs()));f4=teacher.encode(env.get_obs());assert state_hashes(native_parent(env,env.get_obs()))==before_read;np.save(attempt/'prefix4_dino_spatial.npy',f4,allow_pickle=False);prefix4=dict(executed_steps=4,observed_effect=_mean_effect(torch.from_numpy(f4-before_feature)[None])[0].numpy().tolist(),independent_label=B.label(parent_progress,s1,valid_branch=True))
      if env.eval_success and count<32:status='early_success';break
    endpoint=native_parent(env,env.get_obs());eh=state_hashes(endpoint);end_progress=B.observe(env);after_feature=teacher.encode(env.get_obs());assert state_hashes(native_parent(env,env.get_obs()))==eh;effect=_mean_effect(torch.from_numpy(after_feature-before_feature)[None])[0].numpy().tolist() if status=='complete_horizon' else None
    np.save(attempt/'endpoint_dino_spatial.npy',after_feature,allow_pickle=False);(attempt/'endpoint.pickle').write_bytes(pickle.dumps(endpoint,protocol=4));row=dict(query_id=query_id,branch_id=name,parent_recreated=True,parent_hashes=ph,endpoint_hashes=eh,executed_steps=count,endpoint_status=status,error=error,observed_effect=effect,independent_label=B.label(parent_progress,end_progress,valid_branch=status!='invalid_branch'),parent_progress=parent_progress,endpoint_progress=end_progress,secondary_prefix4=prefix4,endpoint_native_success=bool(env.eval_success),sealed_query_sha256=sha(sf));seal_branch(attempt,row);row=read_verified_branch(attempt,query_id=query_id,branch_id=name);env.close_env(clear_cache=True);env=None
   records[name]=row
   if name.startswith('anchor-') and name!='anchor-a':
    reference=records.get('anchor-a',continued_anchor if continuation else None)
    for k in ['parent_hashes','endpoint_hashes','endpoint_status','executed_steps','observed_effect','independent_label','error','parent_progress','endpoint_progress','secondary_prefix4']:assert row[k]==reference[k],k
   if name=='anchor-qual-final':
    if continuation and any(n in continuation['unknown'] for n in ['anchor-a','anchor-aa',f'rank-{selected}','anchor-qual-final']):
     B.immutable_json(out/'original_AABA_remains_unresolved.json',dict(unknown_original_branches=[n for n in ['anchor-a','anchor-aa',f'rank-{selected}','anchor-qual-final'] if n in continuation['unknown']],original_unknowns_not_replaced=True,new_pre_control_exact_to_old_Full=True))
    else:B.immutable_json(out/'AABA_qualification.json',B.require_repeats(records,selected))
    B.immutable_json(out/'selected_raw_refined_PB_contrast.json',B.repair(records,selected))
    if cli.qualify_only:
     assert identity.source_digest(C)==B.CORE_SHA256 and identity.source_digest(H)==p['reference_controller_sha256'];state.update(status='qualified_PB_query_AABA_pool_pending',completed_branches=list(records),full_pool_pending=True,ended_utc=stamp());save();return
   state.update(status='collecting_original_PB_pool',completed_branches=list(records));save()
  if continuation:
   B.immutable_json(out/'selected_raw_refined_PB_contrast.json',B.repair(records,selected));B.immutable_json(out/'continuation_measurement_controls.json',dict(original_unknowns_not_replaced=True,original_AABA_complete=all(n in continuation['completed'] for n in ['anchor-a','anchor-aa',f'rank-{selected}','anchor-qual-final']),new_pre_Full_exact_to_original=True,new_post_Full_exact_to_original=True))
  candidates=[dict(candidate_id=f'rank-{rank}',valid=True,horizon=32,predicted_effect=predicted[rank].tolist(),historical_effect=historical[rank].tolist(),observed_effect=records[f'rank-{rank}']['observed_effect'],endpoint_status=records[f'rank-{rank}']['endpoint_status'],executed_steps=records[f'rank-{rank}']['executed_steps'],applicable=records[f'rank-{rank}']['independent_label']['applicable']) for rank in np.flatnonzero(valid)]
  data=dict(task='press_button',cohort=f'originalPB12000_seed0_q{q}',descriptor='uniform_mean_DINO_delta',required_effect_available=False,candidates=candidates);metrics=centered_effect_metrics(data)
  if 'constant_query' in metrics['references']:metrics['references']['constant_query']['absolute_mse']=None;metrics['references']['constant_query']['absolute_mse_unavailable_reason']='Required query effect not recorded;only centered constant comparison is identified'
  B.immutable_json(out/'candidate_effects.json',data);B.immutable_json(out/'centered_effects.json',metrics);assert identity.source_digest(C)==B.CORE_SHA256 and identity.source_digest(H)==p['reference_controller_sha256'];state.update(status='complete_original_PB_fixed_query_pool',complete_candidate_endpoints=metrics['complete_endpoints'],unknown_candidate_endpoints=metrics['valid_candidates']-metrics['complete_endpoints'],new_policy_generations=0,population_effect_estimated=False,post_pool_final_anchor_verified=records[Q.NEW_CONTROL if continuation else 'anchor-pool-final']['endpoint_status']!='unknown_terminal_infrastructure',additional_resource_boundary_control=Q.NEW_CONTROL if continuation else None,retained_infrastructure_unknown_branches=list(continuation['unknown']) if continuation else [],ended_utc=stamp());save()
 except BaseException as exc:
  state.update(status='failed_or_interrupted_preserve_all',error=repr(exc),traceback=traceback.format_exc(),ended_utc=stamp());save();raise
 finally:
  if env is not None:
   try:env.close_env(clear_cache=True)
   except Exception:pass
  for child in mp.active_children():
   if child.pid not in children_before:child.terminate();child.join(timeout=5)
if __name__=='__main__':main()
