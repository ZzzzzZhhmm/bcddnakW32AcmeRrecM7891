#!/usr/bin/env python3
"""Resume fixed Full-source K32 branches using exact fresh-scene parents.

Original predictions/actions are already sealed before truth. Each query runs
refined/refined, all original valid ranks in rank order, and a final refined
anchor. Completed branches are immutable; interrupted attempts remain unknown.
This does not generate null actions or estimate full-policy success rates.
"""
from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('core-code','rmbench-runtime','source','plan','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--episode-index',type=int,required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    args=parser.parse_args()
    core=args.core_code.resolve(strict=True);runtime=args.rmbench_runtime.resolve(strict=True);out=args.output.resolve()
    if out.is_relative_to(ROOT) or out.is_relative_to(core):raise ValueError('evidence must stay outside source')
    sys.path[:0]=[str(core/'src'),str(core)]
    import fastwam.research
    fastwam.research.__path__.insert(0,str(ROOT/'src/fastwam/research'))
    import fcntl
    import numpy as np
    import torch
    from scripts.nonreal_job import source_identity
    from scripts.qualify_rebuttal_rmbench_task import resolved_config,source_identity as runtime_identity
    from fastwam.memory.robotwin_artifacts import RobotwinQposZScore
    from fastwam.models.warm.retrospection_model import _spatial_effect_summary
    from fastwam.research.branches import InvalidBranchExecution
    from fastwam.research.candidate_pool import decode_branch_actions,centered_effect_metrics
    from fastwam.research.evidence import append_record
    from fastwam.research.factual_effect_reader import FrozenFactualEffectReader
    from fastwam.research.fresh_branches import OfficialQposGuard
    from fastwam.research.fresh_pool import atomic_json,read_verified_branch,seal_branch
    from fastwam.research.native_replay import native_parent,observation_arrays,require_observation,require_trace_step,state_hashes,physical_joint_state
    from fastwam.research.resumable_pairs import file_sha,read_json,read_roles
    from fastwam.research.rmbench_outcomes import observe_task_progress,label_next_interaction

    if Path(sys.modules['fastwam.models.warm.retrospection_model'].__file__).resolve()!=core/'src/fastwam/models/warm/retrospection_model.py':
        raise ValueError('effect projection must come from the original fixed core')
    plan=read_json(args.plan);episode=args.episode_index;identity=read_json(args.source/'identity.json');task=identity['task']
    if task not in plan['tasks'] or not 0<=episode<plan['candidate_specific_effects']['source_episodes_per_task']:
        raise ValueError('source is outside the prespecified candidate-effect cohort')
    queries=plan['candidate_specific_effects']['query_replan_ordinals']
    if queries!=[8,48] or plan['candidate_specific_effects']['candidate_slots']!=32:
        raise ValueError('original K32/q8/q48 protocol required')
    if plan['inference']['execution_prefix']!=4 or plan['inference']['horizon']!=32:raise ValueError('original H32/prefix4 required')
    bound=source_identity(core);sim=runtime_identity(runtime)
    trials=args.source/'role_trials.jsonl'
    if not trials.exists():trials=args.source/'pairs.jsonl'
    rows=[json.loads(s) for s in trials.read_text().splitlines() if s.strip()]
    binding=dict(task=task,checkpoint_sha256=plan['checkpoint_sha256'],evidence_plan_sha256=file_sha(args.plan),
        core_source_sha256=bound['sha256'],runtime_source_sha256=sim['sha256'],official_step_limit={'rearrange_blocks':700,'battery_try':1000}[task])
    admitted=read_roles(args.source,binding,{r['episode_id']:r['seed'] for r in rows})
    full,=[r for r in admitted if r['role']=='full' and r['row']['episode_id']==episode]
    seed=full['row']['seed'];case=Path(full['worker_evidence']);expert=read_json(case.parent/'expert_info.json')
    runtime_args=read_json(args.source/'full-runtime_args.json');contract=read_json(Path(identity['contract_path']))
    with np.load(case/'qpos_trace.npz',allow_pickle=False) as loaded:trace={k:loaded[k].copy() for k in loaded.files}
    inputs=[args.plan,args.source/'identity.json',trials,case/'qpos_trace.npz',Path(identity['contract_path'])]
    collector=source_identity(ROOT)
    fixed=dict(schema='warm.fresh-scene-candidate-pool.v1',task=task,episode_id=episode,seed=seed,query_ordinals=queries,
        checkpoint_sha256=plan['checkpoint_sha256'],core_source_sha256=bound['sha256'],runtime_source_sha256=sim['sha256'],
        collector_source_sha256=collector['sha256'],inputs=[dict(path=str(p),sha256=file_sha(p)) for p in inputs],
        total_window_seconds=12*3600,collector_torch_seed=3407,main_process_pytorch_memory_fraction=.12,
        source_policy_rng_regenerated=False,physical_unpack=False,factual_policy_updated=False,
        scope='New fixed-source candidate effects. No candidate-null, aggregate policy SR or original-paper number replacement.')
    out.mkdir(parents=True,exist_ok=True)
    lease=(out/'campaign.lock').open('a+')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);os.set_inheritable(lease.fileno(),True)
    if (out/'identity.json').exists():
        old=read_json(out/'identity.json');started_utc=old.pop('started_utc')
        if old!=fixed:raise ValueError('resume must preserve exact science/source/budget bindings')
    else:
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(out/'identity.json',dict(fixed,started_utc=started_utc))
    remaining=datetime.datetime.fromisoformat(started_utc).timestamp()+fixed['total_window_seconds']-time.time()
    if remaining<=0:raise TimeoutError('persistent campaign window expired; do not reset it')
    deadline=time.monotonic()+remaining
    invocation=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'-{os.getpid()}'
    if (out/'summary.json').exists():
        previous=out/'previous_invocations';previous.mkdir(exist_ok=True)
        atomic_json(previous/(invocation+'-previous-summary.json'),read_json(out/'summary.json'))
    append_record(out/'invocations.jsonl',dict(invocation=invocation,event='start',persistent_started_utc=started_utc,remaining_seconds=remaining))
    torch.manual_seed(3407);torch.cuda.manual_seed_all(3407);torch.cuda.set_per_process_memory_fraction(.12)
    reader=FrozenFactualEffectReader.from_source_runtime(runtime_args,contract)
    if (out/'reader_receipt.json').exists():
        if read_json(out/'reader_receipt.json')!=reader.receipt:raise ValueError('reader receipt differs on resume')
    else:atomic_json(out/'reader_receipt.json',reader.receipt)
    normalizer=RobotwinQposZScore.from_dataset_stats(read_json(Path(runtime_args['dataset_stats_path'])))
    os.chdir(runtime);sys.path[:0]=[str(runtime),str(runtime/'description/utils')]
    from generate_episode_instructions import generate_episode_descriptions
    cls=getattr(importlib.import_module('envs.'+task),task)
    config=resolved_config(runtime,task,seed);config['now_ep_num']=episode
    env=None
    report=dict(status='running',completed_queries=[],source_episode=episode,task=task)
    atomic_json(out/'summary.json',report)
    try:
        for query in queries:
            frame=query*4;qdir=out/f'q{query:04d}';qdir.mkdir(exist_ok=True)
            if frame>=len(trace['commands']):
                atomic_json(qdir/'summary.json',dict(status='not_reached',frame_index=frame,original_commands=len(trace['commands'])))
                report['completed_queries'].append(dict(query=query,status='not_reached'));continue
            gd=case/f'generations/replan-{query:04d}';generation=read_json(gd/'generation.json')
            if not generation['before_any_execution_of_this_chunk'] or generation['frame_index']!=frame or generation['checkpoint_sha256']!=plan['checkpoint_sha256']:
                raise ValueError('source generation is not sealed to this factual query')
            if file_sha(gd/'model_actions.npy')!=generation['array_sha256']:raise ValueError('source generation changed')
            candidates=[]
            for metadata_path in (case/'probes').glob('*/proposal.json'):
                meta=read_json(metadata_path)
                if meta['identity']['frame_index']==frame and meta['identity']['seed']==seed and meta['identity']['role']=='full':candidates.append(metadata_path)
            metadata_path,=candidates;metadata=read_json(metadata_path);probe_path=metadata_path.with_suffix('.npz')
            if file_sha(probe_path)!=metadata['array_sha256'] or metadata['capture_phase']!='before_any_branch_truth':raise ValueError('source probe changed')
            if metadata['identity']['checkpoint_sha256']!=plan['checkpoint_sha256']:raise ValueError('source probe model changed')
            with np.load(probe_path,allow_pickle=False) as data:probe={k:data[k].copy() for k in data.files}
            if probe['valid'].dtype!=np.bool_ or probe['valid'].shape!=(1,32) or probe['adapted_actions'].shape!=(1,32,32,14):raise ValueError('exact original K32 pool required')
            refined=decode_branch_actions(normalizer,np.load(gd/'model_actions.npy',allow_pickle=False))
            prefix=min(4,len(trace['commands'])-frame)
            if not np.array_equal(refined[:prefix],trace['commands'][frame:frame+prefix]):raise ValueError('refined prefix differs from factual source')
            native={f'rank-{rank}':decode_branch_actions(normalizer,probe['adapted_actions'][0,rank]) for rank in np.flatnonzero(probe['valid'][0])}
            actions={'anchor-a':refined,'anchor-aa':refined,**native,'anchor-final':refined}
            query_id=f'fixed-fresh-{task}-e{episode:04d}-q{query:04d}'
            seal=dict(query_id=query_id,source_query_id=metadata['query_id'],source_metadata_sha256=file_sha(metadata_path),
                source_probe_sha256=file_sha(probe_path),source_generation_sha256=file_sha(gd/'model_actions.npy'),
                expected_branch_order=list(actions),valid_candidates=len(native),horizon=32,before_any_new_truth=True)
            if (qdir/'sealed_query.json').exists():
                if read_json(qdir/'sealed_query.json')!=seal:raise ValueError('sealed query differs on resume')
                with np.load(qdir/'sealed_native_actions.npz',allow_pickle=False) as saved:
                    if set(saved.files)!=set(actions) or any(not np.array_equal(saved[k],v) for k,v in actions.items()):
                        raise ValueError('sealed native actions differ on resume')
            else:
                temporary=qdir/'sealed_native_actions.npz.partial'
                with temporary.open('wb') as stream:np.savez_compressed(stream,**actions)
                temporary.replace(qdir/'sealed_native_actions.npz');atomic_json(qdir/'sealed_query.json',seal)
            records={}
            for branch_id,commands in actions.items():
                branch_root=qdir/branch_id;branch_root.mkdir(exist_ok=True)
                completed=[]
                for done in branch_root.glob('attempt-*/complete.json'):
                    completed.append(read_verified_branch(done.parent,query_id=query_id,branch_id=branch_id))
                if len(completed)>1:raise ValueError('duplicate completed branch; inspect rather than choose a favorable attempt')
                if completed:
                    row=completed[0]
                    if row['sealed_query_sha256']!=file_sha(qdir/'sealed_query.json'):raise ValueError('completed branch belongs to a different seal')
                    if records and row['parent_hashes']!=records['anchor-a']['parent_hashes']:raise RuntimeError('completed branch parent differs')
                    records[branch_id]=row
                    if branch_id in ('anchor-aa','anchor-final') and (row['endpoint_hashes']!=records['anchor-a']['endpoint_hashes'] or row['observed_effect']!=records['anchor-a']['observed_effect']):
                        raise RuntimeError('previous refined anchor mismatch remains quarantined')
                    continue
                if time.monotonic()>=deadline:raise TimeoutError('persistent campaign window expired')
                number=1+len(list(branch_root.glob('attempt-*')));attempt=branch_root/f'attempt-{number:03d}';attempt.mkdir(exist_ok=False)
                atomic_json(attempt/'attempt.json',dict(query_id=query_id,branch_id=branch_id,pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
                env=cls();random.seed(seed);env.setup_demo(**config)
                if type(env.scene.physx_system).__name__!='PhysxCpuSystem' or env.eval_video_path is not None or env.save_data or env.render_freq:
                    raise ValueError('pinned native CPU physics and no video/data/render required')
                instruction=str(np.random.choice(generate_episode_descriptions(task,[expert['info']],100)[0]['unseen']));env.set_instruction(instruction=instruction)
                if instruction!=full['initial_identity']['instruction']:raise ValueError('original instruction differs')
                with np.load(case/'initial_observation.npz',allow_pickle=False) as data:require_observation({k:data[k] for k in data.files},observation_arrays(env.get_obs()))
                with OfficialQposGuard(env) as guard:
                    for index,command in enumerate(trace['commands'][:frame]):
                        if time.monotonic()>=deadline:raise TimeoutError('persistent window expired during original replay prefix')
                        before=np.array(env.get_obs()['joint_action']['vector'],copy=True)
                        if not guard.step(command):raise RuntimeError('original replay prefix terminated')
                        after=np.array(env.get_obs()['joint_action']['vector'],copy=True);require_trace_step(trace,index,before,after)
                obs=env.get_obs()
                with np.load(case/f'replan-{query:04d}.npz',allow_pickle=False) as data:require_observation({k:data[k] for k in data.files},observation_arrays(obs))
                parent=native_parent(env,obs);parent_hash=state_hashes(parent)
                if records and parent_hash!=records['anchor-a']['parent_hashes']:raise RuntimeError('fresh branch parent differs')
                before=reader.encode(obs);parent_outcome=observe_task_progress(env)
                if state_hashes(native_parent(env,env.get_obs()))!=parent_hash:raise RuntimeError('truth/outcome read mutated parent')
                if records:
                    reference=np.load(qdir/'parent_dino_spatial.npy',allow_pickle=False)
                    if not np.array_equal(before,reference):raise RuntimeError('frozen parent features differ')
                else:np.save(qdir/'parent_dino_spatial.npy',before,allow_pickle=False)
                (attempt/'parent.pickle').write_bytes(pickle.dumps(parent,protocol=4))
                np.save(attempt/'parent_dino_spatial.npy',before,allow_pickle=False)
                count=0;status='complete_horizon';error=None
                with OfficialQposGuard(env) as guard:
                    for index,command in enumerate(commands):
                        if time.monotonic()>=deadline:raise TimeoutError('persistent window expired during candidate branch')
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
                endpoint=native_parent(env,env.get_obs());endpoint_hash=state_hashes(endpoint)
                after=reader.encode(env.get_obs());outcome=observe_task_progress(env)
                if state_hashes(native_parent(env,env.get_obs()))!=endpoint_hash:raise RuntimeError('endpoint truth read mutated native state')
                effect=None
                if status=='complete_horizon':
                    effect=_spatial_effect_summary(torch.from_numpy(after-before)[None],torch.from_numpy(before)[None])[0].numpy()
                    if not np.isfinite(effect).all():raise ValueError('nonfinite effect truth')
                label=label_next_interaction(parent_outcome,outcome,valid_branch=status!='invalid_branch')
                (attempt/'endpoint.pickle').write_bytes(pickle.dumps(endpoint,protocol=4));np.save(attempt/'endpoint_dino_spatial.npy',after,allow_pickle=False)
                row=dict(query_id=query_id,branch_id=branch_id,parent_recreated=True,parent_hashes=parent_hash,endpoint_hashes=endpoint_hash,
                    executed_steps=count,endpoint_status=status,error=error,observed_effect=effect.tolist() if effect is not None else None,
                    parent_outcome=parent_outcome,endpoint_outcome=outcome,independent_label=label,
                    sealed_query_sha256=file_sha(qdir/'sealed_query.json'))
                seal_branch(attempt,row);records[branch_id]=read_verified_branch(attempt,query_id=query_id,branch_id=branch_id)
                if branch_id in ('anchor-aa','anchor-final'):
                    if row['endpoint_hashes']!=records['anchor-a']['endpoint_hashes'] or row['observed_effect']!=records['anchor-a']['observed_effect']:
                        raise RuntimeError('repeated refined endpoint/effect differs; quarantine this query')
                env.close_env(clear_cache=True);env=None
                atomic_json(qdir/'progress.json',dict(status='running',completed_branches=list(records),expected_branches=list(actions)))
                print(json.dumps(dict(stage='fresh_pool_branch_complete',query=query,branch=branch_id,status=status,commands=count)),flush=True)
            assembled=dict(task=task,episode_id=episode,query_id=query_id,cohort='fixed_full_factual_fresh_scene_candidate_pool',
                checkpoint_sha256=plan['checkpoint_sha256'],required_effect=probe['required_effect'][0].tolist(),candidates=[])
            for rank in np.flatnonzero(probe['valid'][0]):
                row=records[f'rank-{rank}'];assembled['candidates'].append(dict(candidate_id=f'rank-{rank}',valid=True,horizon=32,
                    predicted_effect=probe['predicted_effect'][0,rank].tolist(),historical_effect=probe['historical_effect'][0,rank].tolist(),
                    observed_effect=row['observed_effect'],endpoint_status=row['endpoint_status'],executed_steps=row['executed_steps'],
                    applicable=row['independent_label']['applicable'],label_provenance=row['independent_label']['label_provenance']))
            atomic_json(qdir/'candidate_effects.json',assembled);atomic_json(qdir/'centered_effects.json',centered_effect_metrics(assembled))
            atomic_json(qdir/'summary.json',dict(status='complete_fixed_query',candidate_slots=32,valid_candidates=len(native),
                complete_endpoints=sum(r['endpoint_status']=='complete_horizon' for name,r in records.items() if name.startswith('rank-')),
                repeated_refined_endpoints=True,full_anchor_support=all(records[name]['endpoint_status']=='complete_horizon' for name in ('anchor-a','anchor-aa','anchor-final')),
                original_saved_retrieval_tensor_available=False))
            # Append a query-level receipt, without Git operations or worktree
            # cleanliness checks. An interruption cannot duplicate the marker.
            marker='fresh-pool:'+query_id+':'+file_sha(qdir/'sealed_query.json')
            with args.ledger.open('a+',encoding='utf-8') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX);stream.seek(0)
                existing=stream.read()
                if marker not in existing:
                    summary=read_json(qdir/'summary.json')
                    stream.write('\n\n### 自动采集：fresh-scene K32 '+query_id+'\n\n'+
                        marker+'；原K32有效候选'+str(len(native))+'，完整H32端点'+str(summary['complete_endpoints'])+\
                        '，完整refined重复支持'+str(summary['full_anchor_support'])+'；原件`'+str(qdir)+'`，'+\
                        '候选结果SHA `'+file_sha(qdir/'candidate_effects.json')+'`，centered结果SHA `'+file_sha(qdir/'centered_effects.json')+'`。'+\
                        '**仅固定source/query的直接候选效应数据；不替换原论文表格，不视作整个cohort或策略收益。**\n')
                    stream.flush();os.fsync(stream.fileno())
                fcntl.flock(stream,fcntl.LOCK_UN)
            report['completed_queries'].append(dict(query=query,status='complete_fixed_query'));atomic_json(out/'summary.json',report)
        report['status']='complete_fixed_source_queries'
    except BaseException as exc:
        report.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if env is not None and hasattr(env,'scene'):
            try:env.close_env(clear_cache=True)
            except Exception as exc:report['close_error']=repr(exc)
        report.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),core_source_sha256_after=source_identity(core)['sha256'],runtime_source_sha256_after=runtime_identity(runtime)['sha256'])
        atomic_json(out/'summary.json',report)
        append_record(out/'invocations.jsonl',dict(invocation=invocation,event='end',**report))
        fcntl.flock(lease,fcntl.LOCK_UN);lease.close()
    if report['core_source_sha256_after']!=bound['sha256'] or report['runtime_source_sha256_after']!=sim['sha256']:
        raise RuntimeError('bound source changed during execution')


if __name__=='__main__':main()
