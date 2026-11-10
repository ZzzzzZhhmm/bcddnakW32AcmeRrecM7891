"""Audit two exact factual-prefix replays, preserving every native rejection.

This is an observation audit, not candidate execution or an efficacy metric.
It never retries or replaces a command, suppresses a rejection record, or uses
physical unpack. Existing source command/observation identities must agree.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib,importlib.util,json,os,pickle,random,sys,traceback

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--query-plan',type=Path,required=True);ap.add_argument('--expected-plan-sha256',required=True);ap.add_argument('--output',type=Path,required=True);cli=ap.parse_args()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());assert sha(cli.query_plan)==cli.expected_plan_sha256;plan=read(cli.query_plan);C=Path(plan['original_source']);H=Path(plan['reference_controller']);runtime=Path(plan['rmbench_runtime']);source=Path(plan['factual_source']);P=Path(plan['evidence_root']);out=cli.output;out.mkdir(exist_ok=False)
    sys.path[:0]=[str(C/'src'),str(C)];import fastwam
    fastwam.__path__.append(str(H/'src/fastwam'));import fastwam.research
    fastwam.research.__path__.insert(0,str(H/'src/fastwam/research'))
    import numpy as np,torch,multiprocessing as mp
    from experiments.rmbench.warm_policy.policy_core import factual_cameras,factual_joint_state
    from fastwam.research.fresh_branches import OfficialQposGuard
    from fastwam.research.branches import InvalidBranchExecution
    from fastwam.research.native_replay import native_parent,state_hashes,physical_joint_state,require_observation
    from fastwam.research.fresh_pool import atomic_json
    spec=importlib.util.spec_from_file_location('setup_native',H/'scripts/qualify_rebuttal_rmbench_task.py');setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)
    spec=importlib.util.spec_from_file_location('identity_native',H/'scripts/run_rebuttal_training_with_runtime_trace.py');identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
    assert identity.source_digest(C)==plan['original_source_sha256'] and identity.source_digest(H)==plan['reference_controller_sha256']
    for item in plan['inputs']:assert sha(item['path'])==item['sha256']
    query=int(plan['query_replan']);assert query in (48,96);frame=query*4;trace=[json.loads(s) for s in (source/'native_trace.jsonl').read_text().splitlines()];args=read(source/'runtime_args.json');assets=read(P/'resumable_rb_cases0003_0009_v1_plan.json');sys.path[:0]=[str(runtime),str(runtime/'description/utils'),assets['curobo_source']];os.chdir(runtime)
    from generate_episode_instructions import generate_episode_descriptions
    cls=getattr(importlib.import_module('envs.blocks_ranking_try'),'blocks_ranking_try');cfg=setup.resolved_config(runtime,'blocks_ranking_try',0);assert cfg==read(source/'native_config.json');initial=read(source/'initial_identity.json');expert=read(source/'expert_setup_metadata.json');rng=pickle.loads((source/f'probes/q{query:04d}/sampling_rng_before.pkl').read_bytes());deadline=datetime.datetime.fromisoformat(plan['absolute_deadline_utc']);state=dict(status='admitted_factual_prefix_audit',query=query,frame=frame,source_queries=0,new_candidate_endpoints=0,query_plan_sha256=cli.expected_plan_sha256,passes_complete=0,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());env=None;children={p.pid for p in mp.active_children()};passes=[];parents=[]
    def save():state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic_json(out/'operator.json',state)
    def require_source_obs(index,obs):
        f=source/f'replan-{index//4:04d}.npz'
        with np.load(f,allow_pickle=False) as saved:require_observation({k:saved[k] for k in saved.files},{**factual_cameras(obs),'native_qpos':factual_joint_state(obs)})
    def plain(items):return [{k:v.tolist() if hasattr(v,'tolist') else v for k,v in item.items()} for item in items]
    save()
    try:
        for repeat in range(2):
            random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);env=cls();env.setup_demo(**cfg)
            instruction=str(np.random.choice(generate_episode_descriptions('blocks_ranking_try',[expert['info']],100)[0]['unseen']));assert instruction==initial['instruction'];env.set_instruction(instruction=instruction);require_source_obs(0,env.get_obs());rows=[];errors=[]
            with (out/f'pass-{repeat}.jsonl').open('x',buffering=1) as stream,OfficialQposGuard(env) as guard:
                for index,reference in enumerate(trace[:frame]):
                    assert datetime.datetime.now(datetime.timezone.utc)<deadline,'Original query deadline expired'
                    obs=env.get_obs()
                    if index%4==0:require_source_obs(index,obs)
                    before=factual_joint_state(obs).copy();physical_before=plain(physical_joint_state(env));assert np.array_equal(before,np.asarray(reference['before'],dtype=np.float32));error=None;accepted=False
                    try:accepted=guard.step(np.asarray(reference['command'],dtype=np.float32))
                    except InvalidBranchExecution as exc:
                        error=str(exc)
                        # Observe only the specifically diagnosed TOPP exception;
                        # an RPC violation or another failure remains fatal.
                        assert guard.errors and all("mplib_planner: RuntimeError('Fail to parameterize path')" in v for v in guard.errors),guard.errors
                        errors.append(dict(index=index,error=error))
                    after=factual_joint_state(env.get_obs()).copy();assert np.array_equal(after,np.asarray(reference['after'],dtype=np.float32));assert env.take_action_cnt==reference['take_action_cnt']
                    row=dict(index=index,command=reference['command'],error=error,guard_accepted=accepted,before=before.tolist(),after=after.tolist(),physical_before=physical_before,physical_after=plain(physical_joint_state(env)),take_action_cnt=env.take_action_cnt)
                    if repeat:assert row==passes[0][index],f'Two-pass prefix observation/rejection differs at{index}'
                    stream.write(json.dumps(row)+'\n');rows.append(row);state.update(status='auditing_factual_prefix',repeat=repeat,last_index=index,observed_prefix_rejections=len(errors));save()
            obs=env.get_obs();require_source_obs(frame,obs);random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch_cpu']);torch.cuda.set_rng_state_all(rng['torch_cuda']);parent=native_parent(env,obs);hashes=state_hashes(parent)
            if repeat:assert hashes==parents[0]
            (out/f'parent-pass-{repeat}.pickle').write_bytes(pickle.dumps(parent,protocol=4));passes.append(rows);parents.append(hashes);env.close_env(clear_cache=True);env=None;state.update(passes_complete=repeat+1);save()
        manifest=dict(status='two_factual_prefix_replays_exact_to_source_observations_and_each_other',query=query,frame=frame,commands_each_pass=frame,prefix_errors=errors,all_per_command_source_API_states_exact=True,all_replan_source_RGB_observations_exact=True,all_between_pass_measured_joint_states_exact=True,final_parent_rng_hashes_exact=True,parent_hashes=parents[0],query_plan_sha256=cli.expected_plan_sha256,source_trace_sha256=sha(source/'native_trace.jsonl'),pass_files=[dict(path=str(out/f'pass-{i}.jsonl'),sha256=sha(out/f'pass-{i}.jsonl')) for i in range(2)],new_candidate_endpoints=0,source_failure_flags_were_not_recorded=True,scope='New deterministic replay audit of a factual context,not retroactive certification of every source command as valid. Rejections remain observed failures;no candidate guard or outcome criterion changed.')
        atomic_json(out/'prefix_error_manifest.json',manifest);state.update(status='complete_two_pass_factual_prefix_audit',prefix_rejections=len(errors),ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();print(json.dumps(manifest),flush=True)
    except BaseException as exc:
        state.update(status='failed_factual_prefix_audit',error=repr(exc),traceback=traceback.format_exc(),ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();raise
    finally:
        if env is not None:
            try:env.close_env(clear_cache=True)
            except Exception:pass
        for child in mp.active_children():
            if child.pid not in children:child.terminate();child.join(timeout=5)

if __name__=='__main__':main()
