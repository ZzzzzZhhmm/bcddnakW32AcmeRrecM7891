#!/usr/bin/env python3
"""Execute sealed null and repeated Full branches at a prespecified parent.

Reuses already verified original Raw/Full branches. Every new branch starts in
an independently reset scene, replays the same original factual commands and
must match the complete existing native parent. No policy inference or unpack.
"""
from __future__ import annotations
import argparse,importlib,json,os,pickle,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('core-code','rmbench-runtime','source','plan','pool-query','null-query','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true',help='Validate actual sealed artifacts on CPU, without a simulator or DINO load')
    parser.add_argument('--episode-index',type=int,default=0)
    parser.add_argument('--query-ordinal',type=int,default=8)
    parser.add_argument('--allow-pending-pool-records',action='store_true',
        help='Pipeline new null branches after two exact initial Full anchors; final pool anchor and selected raw remain explicit report dependencies')
    args=parser.parse_args();core=args.core_code.resolve(strict=True);runtime=args.rmbench_runtime.resolve(strict=True);out=args.output.resolve()
    if out.is_relative_to(ROOT) or out.is_relative_to(core):raise ValueError('evidence must stay outside source')
    out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(core/'src'),str(core)]
    import fastwam.research
    fastwam.research.__path__.insert(0,str(ROOT/'src/fastwam/research'))
    import numpy as np
    import torch
    from scripts.nonreal_job import source_identity
    from scripts.qualify_rebuttal_rmbench_task import resolved_config,source_identity as runtime_identity
    from fastwam.models.warm.retrospection_model import _spatial_effect_summary
    from fastwam.research.resumable_pairs import file_sha,read_json,read_roles,read_committed_trials
    from fastwam.research.sealed_null import require_null_binding,require_native_null_actions
    from fastwam.research.factual_effect_reader import FrozenFactualEffectReader
    from fastwam.research.fresh_pool import atomic_json,read_verified_branch,seal_branch
    from fastwam.research.null_pool_dependencies import load_pool_dependencies
    from fastwam.research.fresh_branches import OfficialQposGuard
    from fastwam.research.branches import InvalidBranchExecution
    from fastwam.research.native_replay import native_parent,observation_arrays,require_observation,require_trace_step,state_hashes,physical_joint_state
    from fastwam.research.rmbench_outcomes import observe_task_progress,label_next_interaction
    from fastwam.research.evidence import append_record
    episode,query=args.episode_index,args.query_ordinal;frame=query*4
    plan=read_json(args.plan);identity=read_json(args.source/'identity.json');task=identity['task']
    if (query not in plan['local_recovery']['query_replan_ordinals'] or task!='rearrange_blocks'
            or not 0<=episode<plan['local_recovery']['source_episodes_per_task']
            or plan['inference']['execution_prefix']!=4):raise ValueError('prespecified fixed RB query and original prefix required')
    bound=source_identity(core);sim=runtime_identity(runtime)
    rows=read_committed_trials(args.source)
    binding=dict(task=task,checkpoint_sha256=plan['checkpoint_sha256'],evidence_plan_sha256=file_sha(args.plan),
        core_source_sha256=bound['sha256'],runtime_source_sha256=sim['sha256'],official_step_limit=700)
    roles=read_roles(args.source,binding,{r['episode_id']:r['seed'] for r in rows})
    full,=[r for r in roles if r['role']=='full' and r['row']['episode_id']==episode]
    seed=full['row']['seed'];case=Path(full['worker_evidence']);expert=read_json(case.parent/'expert_info.json')
    with np.load(case/'qpos_trace.npz',allow_pickle=False) as data:trace={k:data[k].copy() for k in data.files}
    gd=case/f'generations/replan-{query:04d}';generation=read_json(gd/'generation.json')
    pool=args.pool_query.resolve(strict=True);null=args.null_query.resolve(strict=True)
    pool_seal=read_json(pool/'sealed_query.json');null_seal=read_json(null/'sealed_query.json')
    pool_identity=read_json(pool.parent/'identity.json')
    for key,value in dict(task=task,episode_id=episode,seed=seed,checkpoint_sha256=plan['checkpoint_sha256'],
        core_source_sha256=bound['sha256'],runtime_source_sha256=sim['sha256']).items():
        if pool_identity.get(key)!=value:raise ValueError('existing pool identity differs: '+key)
    if pool_seal['query_id']!=f'fixed-fresh-{task}-e{episode:04d}-q{query:04d}' or pool_seal['source_generation_sha256']!=file_sha(gd/'model_actions.npy'):
        raise ValueError('existing pool belongs to a different fixed generation')
    require_null_binding(null_seal,generation,checkpoint_sha256=plan['checkpoint_sha256'],plan_sha256=file_sha(args.plan),
        core_source_sha256=bound['sha256'],runtime_source_sha256=sim['sha256'],generation_file_sha256=file_sha(gd/'generation.json'),
        model_actions_file_sha256=file_sha(gd/'model_actions.npy'),native_actions_file_sha256=file_sha(null/'sealed_native_actions.npz'),
        context_file_sha256=file_sha(null/'sealed_factual_context.pt'),telemetry_file_sha256=file_sha(args.source/'full-telemetry.jsonl'),
        task=task,seed=seed,episode=episode,query=query)
    with np.load(pool/'sealed_native_actions.npz',allow_pickle=False) as data:refined=data['anchor-a'].copy()
    with np.load(null/'sealed_native_actions.npz',allow_pickle=False) as data:actions=require_native_null_actions({k:data[k] for k in data.files},refined)
    if not np.array_equal(refined[:4],trace['commands'][frame:frame+4]):raise ValueError('Full native prefix differs')
    original_probe=case/f'probes/e{episode:04d}-f{frame:04d}/proposal.npz'
    if file_sha(original_probe)!=pool_seal['source_probe_sha256']:raise ValueError('original candidate probe differs')
    with np.load(original_probe,allow_pickle=False) as data:selected=int(data['selected_index'].reshape(-1)[0])
    probes={}
    for role in ('full','candidate-null'):
        path=null/'probes'/role/'proposal.npz';meta=read_json(path.with_suffix('.json'))
        if file_sha(path)!=meta['array_sha256'] or meta['capture_phase']!='before_any_branch_truth':raise ValueError('sealed sampler probe differs')
        with np.load(path,allow_pickle=False) as data:probes[role]={key:data[key].copy() for key in data.files}
    if not np.array_equal(probes['full']['base_gaussian'],probes['candidate-null']['base_gaussian']) or np.any(probes['candidate-null']['g']):
        raise ValueError('null did not retain the exact Gaussian or withdraw candidate influence')
    forbidden={'target_action','target_effect','current_semantic_teacher','target_action_valid_mask','future_valid_mask'}
    context=torch.load(null/'sealed_factual_context.pt',map_location='cpu',weights_only=True)
    non_tensor=read_json(null/'context_non_tensor_fields.json')
    if forbidden.intersection(context) or any(non_tensor.get(key) is not None for key in forbidden):raise ValueError('future/teacher data in sealed inference context')
    dependencies=load_pool_dependencies(pool,query_id=pool_seal['query_id'],selected_rank=selected,
        allow_pending=args.allow_pending_pool_records)
    anchor,raw=dependencies['anchor'],dependencies['raw']
    runtime_args=read_json(args.source/'full-runtime_args.json');contract=read_json(Path(identity['contract_path']))
    fixed=dict(task=task,episode_id=episode,seed=seed,query_ordinal=query,frame_index=frame,horizon=32,
        checkpoint_sha256=plan['checkpoint_sha256'],core_source_sha256=bound['sha256'],runtime_source_sha256=sim['sha256'],
        collector_source_sha256=source_identity(ROOT)['sha256'],pool_query=str(pool),null_query=str(null),
        inputs=[dict(path=str(p),sha256=file_sha(p)) for p in (args.plan,args.source/'identity.json',pool/'sealed_query.json',null/'sealed_query.json',*dependencies['markers'].values())],
        raw_selected_rank=selected,original_raw_label=raw['independent_label']['applicable'] if raw else None,original_refined_label=anchor['independent_label']['applicable'],
        pending_pool_dependencies_at_admission=dependencies['pending'],
        allow_pending_pool_records=args.allow_pending_pool_records,
        physical_unpack=False,new_policy_generation=False,new_independent_policy_success_episode=False,
        scope='One prespecified fixed query: immutable Full anchors, sealed same-context null and exact post-null Full repeat. Pending selected raw/final pool anchor remain unknown; no aggregate efficacy claim.')
    atomic_json(out/'identity.json',fixed);np.savez_compressed(out/'sealed_native_actions.npz',**actions)
    if args.preflight_only:
        atomic_json(out/'summary.json',dict(status='actual_input_contracts_admitted_only',physical_branches_executed=0,
            full_policy_loaded=False,reader_loaded=False,scientific_efficacy_claim=False,
            pending_pool_dependencies=dependencies['pending']))
        print(json.dumps(dict(status='actual_input_contracts_admitted_only',output=str(out))),flush=True)
        return
    report=dict(status='running',new_branches=[],actual_null_executed=False,scientific_efficacy_claim=False)
    atomic_json(out/'summary.json',report)
    # Keep the identical original pool reader and random-state lifecycle.
    torch.manual_seed(3407);torch.cuda.manual_seed_all(3407);torch.cuda.set_per_process_memory_fraction(.12)
    reader=FrozenFactualEffectReader.from_source_runtime(runtime_args,contract)
    if reader.receipt!=read_json(pool.parent/'reader_receipt.json'):raise ValueError('original frozen reader recipe differs')
    atomic_json(out/'reader_receipt.json',reader.receipt)
    os.chdir(runtime);sys.path[:0]=[str(runtime),str(runtime/'description/utils')]
    from generate_episode_instructions import generate_episode_descriptions
    cls=getattr(importlib.import_module('envs.'+task),task);config=resolved_config(runtime,task,seed);config['now_ep_num']=episode
    env=None;started=time.monotonic()
    try:
        for branch_id,commands in (('candidate-null',actions['candidate_null']),('refined-post-null',actions['refined'])):
            attempt=out/branch_id;attempt.mkdir(exist_ok=False);env=cls();random.seed(seed);env.setup_demo(**config)
            if type(env.scene.physx_system).__name__!='PhysxCpuSystem' or env.eval_video_path is not None or env.save_data or env.render_freq:
                raise ValueError('pinned CPU physics and original recording/render settings required')
            instruction=str(np.random.choice(generate_episode_descriptions(task,[expert['info']],100)[0]['unseen']));env.set_instruction(instruction=instruction)
            if instruction!=full['initial_identity']['instruction']:raise ValueError('original instruction differs')
            initial_observation=observation_arrays(env.get_obs());np.savez_compressed(attempt/'initial_observation.npz',**initial_observation)
            with np.load(case/'initial_observation.npz',allow_pickle=False) as data:require_observation({k:data[k] for k in data.files},initial_observation)
            with OfficialQposGuard(env) as guard:
                for index,command in enumerate(trace['commands'][:frame]):
                    before=np.array(env.get_obs()['joint_action']['vector'],copy=True)
                    if not guard.step(command):raise RuntimeError('original factual prefix stopped')
                    require_trace_step(trace,index,before,np.array(env.get_obs()['joint_action']['vector'],copy=True))
            obs=env.get_obs()
            np.savez_compressed(attempt/'query_observation.npz',**observation_arrays(obs))
            with np.load(case/f'replan-{query:04d}.npz',allow_pickle=False) as data:require_observation({k:data[k] for k in data.files},observation_arrays(obs))
            parent=native_parent(env,obs);parent_hash=state_hashes(parent)
            (attempt/'parent.pickle').write_bytes(pickle.dumps(parent,protocol=4))
            atomic_json(attempt/'observed_parent_hashes.json',parent_hash);atomic_json(attempt/'expected_parent_hashes.json',anchor['parent_hashes'])
            if parent_hash!=anchor['parent_hashes']:
                differing=[key for key in parent_hash if parent_hash[key]!=anchor['parent_hashes'].get(key)]
                raise RuntimeError('existing exact native/RNG parent differs: '+str(differing)+'; quarantine, never loosen it')
            before=reader.encode(obs);parent_outcome=observe_task_progress(env)
            np.save(attempt/'parent_dino_spatial.npy',before,allow_pickle=False)
            if not np.array_equal(before,np.load(pool/'parent_dino_spatial.npy',allow_pickle=False)):raise RuntimeError('original parent DINO tokens differ')
            if state_hashes(native_parent(env,env.get_obs()))!=parent_hash:raise RuntimeError('truth read mutated parent')
            count=0;status='complete_horizon';error=None
            with OfficialQposGuard(env) as guard:
                for index,command in enumerate(commands):
                    physical_before=physical_joint_state(env)
                    try:executed=guard.step(command)
                    except InvalidBranchExecution as exc:status='invalid_branch';error=str(exc);break
                    if not executed:status='invalid_branch';error='official executor stopped';break
                    physical_after=physical_joint_state(env)
                    append_record(attempt/'commands.jsonl',dict(step=index,command=command.tolist(),
                        physical_before=[{k:v.tolist() if hasattr(v,'tolist') else v for k,v in a.items()} for a in physical_before],
                        physical_after=[{k:v.tolist() if hasattr(v,'tolist') else v for k,v in a.items()} for a in physical_after]))
                    count+=1
                    if env.eval_success and count<32:status='early_success';break
            endpoint=native_parent(env,env.get_obs());endpoint_hash=state_hashes(endpoint);after=reader.encode(env.get_obs());outcome=observe_task_progress(env)
            if state_hashes(native_parent(env,env.get_obs()))!=endpoint_hash:raise RuntimeError('truth read mutated endpoint')
            effect=_spatial_effect_summary(torch.from_numpy(after-before)[None],torch.from_numpy(before)[None])[0].numpy() if status=='complete_horizon' else None
            row=dict(query_id=pool_seal['query_id'],branch_id=branch_id,parent_recreated=True,parent_hashes=parent_hash,endpoint_hashes=endpoint_hash,
                endpoint_status=status,error=error,executed_steps=count,observed_effect=effect.tolist() if effect is not None else None,
                parent_outcome=parent_outcome,endpoint_outcome=outcome,independent_label=label_next_interaction(parent_outcome,outcome,valid_branch=status!='invalid_branch'),
                original_null_seal_sha256=file_sha(null/'sealed_query.json'))
            (attempt/'endpoint.pickle').write_bytes(pickle.dumps(endpoint,protocol=4));np.save(attempt/'endpoint_dino_spatial.npy',after,allow_pickle=False)
            seal_branch(attempt,row)
            if branch_id=='refined-post-null' and (endpoint_hash!=anchor['endpoint_hashes'] or row['observed_effect']!=anchor['observed_effect']):
                raise RuntimeError('post-null Full endpoint/effect differs from existing anchors; quarantine')
            report['new_branches'].append(dict(role=branch_id,status=status,steps=count,label=row['independent_label']['applicable']))
            report['actual_null_executed']=True;atomic_json(out/'summary.json',report)
            env.close_env(clear_cache=True);env=None
            print(json.dumps(dict(stage='fresh_null_branch_complete',role=branch_id,status=status,steps=count)),flush=True)
        final_dependencies=load_pool_dependencies(pool,query_id=pool_seal['query_id'],selected_rank=selected,
            allow_pending=args.allow_pending_pool_records)
        raw=final_dependencies['raw']
        report.update(status=('qualified_one_fixed_query_shared_parent_null_contrast' if final_dependencies['complete']
            else 'complete_fixed_null_branches_pending_pool_dependencies'),post_null_Full_endpoint_exact=True,
            pending_pool_dependencies=final_dependencies['pending'],
            reused_original_raw_label=raw['independent_label']['applicable'] if raw else None,
            reused_original_refined_label=anchor['independent_label']['applicable'])
    except BaseException as exc:
        report.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if env is not None and hasattr(env,'scene'):env.close_env(clear_cache=True)
        report.update(elapsed_seconds=time.monotonic()-started,core_source_sha256_after=source_identity(core)['sha256'],runtime_source_sha256_after=runtime_identity(runtime)['sha256'])
        atomic_json(out/'summary.json',report)
    if report['core_source_sha256_after']!=bound['sha256'] or report['runtime_source_sha256_after']!=sim['sha256']:raise RuntimeError('bound source changed')


if __name__=='__main__':main()
