#!/usr/bin/env python3
"""Run a fixed family-holdout development role using frozen native execution.

No training,Git access,formal-test admission or replacement reset sampling.
"""
from pathlib import Path
import argparse,datetime,importlib.util,json,os,random,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pilot_rules',ROOT/'src/fastwam/research/missing_skill_pilot.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def verify(plan):
    N.validate(plan);N.require_window(plan)
    for item in plan['inputs']:
        if N.sha(item['path'])!=item['sha256']:raise ValueError('Pilot bound input changed: '+item['path'])
    for row in plan['universe_rows'].values():
        if N.sha(row['path'])!=row['sha256']:raise ValueError('Search universe changed')
    b=plan['roles']['full']
    for field in ['bundle_args','training_attestation']:
        if N.sha(b[field])!=b[field+'_sha256']:raise ValueError('Full binding changed')
    proof=N.read(b['complete_training_state_receipt']['path'])
    if N.sha(b['complete_training_state_receipt']['path'])!=b['complete_training_state_receipt']['sha256'] or proof['checkpoint_sha256']!=b['checkpoint_sha256'] or proof['status']!='verified_complete_final_state_and_checkpoint':
        raise ValueError('Complete fixed-final training evidence required')
    identity=load('pilot_identity',Path(plan['native_helper_code'])/'scripts/nonreal_job.py')
    if identity.source_identity(ROOT)['sha256']!=plan['pilot_controller_source_sha256']:raise ValueError('Pilot controller source changed')

def prepare_roster(plan,H,out):
    import importlib,numpy as np,torch
    from scripts.qualify_rebuttal_rmbench_task import resolved_config,source_identity
    from fastwam.research.native_replay import physical_joint_state
    runtime=Path(plan['native_runtime']);assert source_identity(runtime)['sha256']==plan['runtime_source_sha256']
    sys.path[:0]=[str(runtime),str(runtime/'description/utils'),plan['curobo_source']];os.chdir(runtime)
    from generate_episode_instructions import generate_episode_descriptions
    env=getattr(importlib.import_module('envs.'+plan['task']),plan['task'])();entries=[]
    def plain(v):
        if hasattr(v,'tolist'):return v.tolist()
        if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [plain(x) for x in v]
        return v
    try:
        for index,seed in enumerate(plan['registered_pilot_seeds']):
            if index not in plan['admitted_case_ids']:
                entries.append(dict(episode_id=index,seed=seed,status='registered_not_admitted'));continue
            N.require_window(plan);config=resolved_config(runtime,plan['task'],seed);config['now_ep_num']=index
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            env.setup_demo(**config)
            try:info=env.play_once();accepted=bool(env.plan_success and env.check_success())
            finally:env.close_env(clear_cache=True)
            if not accepted:
                entries.append(dict(episode_id=index,seed=seed,status='fixed_expert_rejected_no_replacement',expert_info=plain(info)));continue
            ids=[];phys=[]
            for repeat in range(2):
                N.require_window(plan);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                env.setup_demo(**config)
                try:
                    text=str(np.random.choice(generate_episode_descriptions(plan['task'],[info['info']],100)[0]['unseen']));env.set_instruction(instruction=text)
                    ids.append(dict(observation_arrays=H.save_observation(out/f'case{index}-reset{repeat}.npz',env.get_obs()),task_state=H.task_values(env),instruction=text))
                    phys.append(plain(physical_joint_state(env)))
                finally:env.close_env(clear_cache=True)
            if ids[0]!=ids[1] or phys[0]!=phys[1]:raise ValueError('Actual private native reset is not deterministic')
            entries.append(dict(episode_id=index,seed=seed,status='qualified_fixed_development_reset',expert_info=plain(info),initial_identity=ids[0],physical_initial=phys[0]))
    finally:
        H.M.atomic_json(out/'roster.json',dict(schema='warm.rebuttal.fixed-development-roster.v1',task=plan['task'],entries=entries,formal_TEST=False))
        if source_identity(runtime)['sha256']!=plan['runtime_source_sha256']:raise ValueError('Private simulator source changed')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--expected-plan-sha256',required=True)
    ap.add_argument('--prepare-roster',action='store_true');ap.add_argument('--role',choices=N.ROLES);ap.add_argument('--condition',choices=N.CONDITIONS)
    ap.add_argument('--case-id',type=int);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--roster-sha256')
    cli=ap.parse_args()
    if N.sha(cli.plan)!=cli.expected_plan_sha256:raise ValueError('Pilot plan changed')
    plan=N.read(cli.plan);N.validate(plan);N.require_window(plan)
    if not cli.prepare_roster and (cli.case_id not in plan['admitted_case_ids'] or cli.role is None or cli.condition is None):raise ValueError('Unregistered pilot role')
    H=load('frozen_native_role',Path(plan['native_helper_code'])/'scripts/run_rebuttal_matched_native.py')
    lease=H.M.activate_native_isolation(plan);core,identity=H.install_core(plan)
    # Source-identity helpers import fastwam; install the frozen core first.
    verify(plan)
    out=cli.output.resolve();assert not out.is_relative_to(ROOT) and not out.is_relative_to(core);out.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='true',TOKENIZERS_PARALLELISM='false')
    if cli.prepare_roster:
        prepare_roster(plan,H,out);print(json.dumps(dict(status='fixed_dev_reset_stage_complete',formal_TEST=False)));return
    if N.sha(plan['roster'])!=cli.roster_sha256:raise ValueError('Fixed development roster changed')
    entry=N.read(plan['roster'])['entries'][cli.case_id]
    if entry['status']!='qualified_fixed_development_reset':raise ValueError('Fixed development seed not admitted;never replace it')
    import numpy as np,torch
    from experiments.rmbench.warm_policy.deploy_policy import RMBenchWarmPolicy
    from fastwam.research.repertoire import install_search_restriction
    runtime=H.M.load_runtime(plan,'full',out)
    H.M.validate_training_attestation(plan['roles']['full'],N.read(plan['roles']['full']['training_attestation']))
    result=dict(schema='warm.rebuttal.native-family-dev-role.v1',phase=N.PHASE,task=plan['task'],role=cli.role,
        search_condition=cli.condition,episode_id=cli.case_id,seed=entry['seed'],plan_sha256=cli.expected_plan_sha256,
        checkpoint_sha256=plan['roles']['full']['checkpoint_sha256'],status='loading_typed_policy',success=None,queries=[],
        formal_TEST=False,primary_NI_protocol_replaced=False,independently_measured_missing_skill_coverage=None,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    H.M.atomic_json(out/'runtime_args.json',runtime);H.M.atomic_json(out/'result.json',result);policy=None;last_disk_check=0.
    def budget():
        nonlocal last_disk_check
        N.require_window(plan);now=time.monotonic()
        if now-last_disk_check>=30:
            import shutil
            last_disk_check=now
            if shutil.disk_usage(out).free<plan['minimum_free_during_role_bytes']:raise OSError('Pilot storage guard;retain unknown,not failure')
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>plan['worker_raw_soft_limit_bytes']:raise OSError('Pilot raw budget exceeded;preserve unknown evidence')
    try:
        torch.cuda.set_per_process_memory_fraction(float(plan.get('memory_fraction',.75)))
        policy=RMBenchWarmPolicy(runtime);typed=H.M.typed_policy_receipt(policy,plan,'full')
        result.update({k:v for k,v in typed.items() if k!='role'})
        H.M.atomic_json(out/'typed_policy_receipt.json',dict(typed,intervention_role=cli.role,factory_role='full'))
        if policy.contract.bank_manifest_sha256!=plan['bank_manifest_sha256']:raise ValueError('Foreign bank')
        row=np.load(plan['universe_rows'][cli.condition]['path'],allow_pickle=False)
        if row.shape!=(19540,) or len(np.unique(row))!=19540:raise ValueError('Frozen eligible bank rows differ')
        restriction=install_search_restriction(policy.retriever,row)
        H.M.atomic_json(out/'search_universe.json',dict(condition=cli.condition,restriction=restriction,
            actual_missing_skill_certified=False,counts={name:int(np.sum(policy.retriever._task_indices[row]==i)) for i,name in enumerate(policy.retriever.task_vocabulary)}))
        policy.model.configure_research_probe(capture=True,force_null=cli.role=='candidate_null')
        base_receipt=H.M.replan_receipt
        def observed(capture):
            value=base_receipt(capture);probe=policy.model._last_research_probe
            if probe is None or probe['force_null']!=(cli.role=='candidate_null'):raise ValueError('Actual declared research intervention missing')
            arrays=probe['arrays'];g=float(arrays['g'].reshape(-1)[0]);sg=float(arrays['source_gate'].reshape(-1)[0])
            if cli.role=='candidate_null' and (g!=0 or sg!=0 or value['actual_source']!=value['base_gaussian']):raise ValueError('Candidate-null failed actual zero-source boundary')
            ordinal=len(result['queries'])
            value.update(mode_unmodified=cli.role=='full',legacy_Full_probe_enabled=True,observational_probe_enabled=True,
                declared_force_null=cli.role=='candidate_null',effective_gate=g,source_gate=sg,alpha=float(arrays['alpha'].reshape(-1)[0]),
                configured_gate_threshold=float(probe['gate_threshold']),independently_measured_relevant_coverage=None,false_activation=None)
            if ordinal in (0,8) and cli.role=='full':
                f=out/f'pre-execution-probe-q{ordinal:04d}.pt'
                with f.open('xb') as stream:torch.save(dict(probe=probe,bank_rows=capture['online_step'].bank_rows.detach().cpu(),before_any_local_branch_truth=True),stream)
                value['pre_execution_probe_sha256']=N.sha(f)
            return value
        H.M.replan_receipt=observed
        policy._next_episode_index=cli.case_id;policy.reset()
        if policy.controller.active_episode_index!=cli.case_id:raise ValueError('Development episode cursor differs')
        H.execute_native(plan,policy,out,result,budget,cli.case_id)
    except H.M.PolicyOutputError as exc:
        result.update(status='unknown_invalid_model_output_requires_independent_audit',success=None,error=repr(exc),traceback=traceback.format_exc());raise
    except BaseException as exc:
        result.update(status='unknown_runtime_or_interruption',success=None,error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if policy is not None:
            try:policy.close()
            except BaseException as exc:result.update(status='unknown_cleanup_error',success=None,cleanup_error=repr(exc))
        if identity(core)['sha256']!=plan['evaluation_source_sha256'] or identity(Path(plan['native_helper_code']))['sha256']!=plan['controller_source_sha256']:
            result.update(status='unknown_source_change',success=None)
        result.update(ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),formal_TEST=False,
            evidence_scope='Prespecified native development family-holdout pilot;not formal100pairs or certified skill absence/NI.')
        H.M.atomic_json(out/'result.json',result)
        if result['status'].startswith('unknown'):raise RuntimeError('Pilot outcome remains unknown;preserve and inspect')
    print(json.dumps(dict(status=result['status'],role=cli.role,condition=cli.condition,success=result['success'],queries=len(result['queries']),formal_TEST=False)))

if __name__=='__main__':main()
