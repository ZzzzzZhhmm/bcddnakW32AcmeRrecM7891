#!/usr/bin/env python3
"""Collect fresh-reset Full/candidate-null RMBench pairs with factual histories.

The two policies own independent model/retriever capabilities. No simulator
restoration is used for episode success: each role resets from the same admitted
seed and must reproduce the initial state, cameras and unseen instruction.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-args',type=Path,required=True)
    parser.add_argument('--rmbench-runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('qualification','formal'),required=True)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/nonreal/rebuttal_p0_evidence_plan.json')
    parser.add_argument('--seed-start',type=int,required=True)
    parser.add_argument('--candidate-limit',type=int,default=2000)
    parser.add_argument('--cases',type=int,required=True)
    parser.add_argument('--case-index-offset',type=int,default=0)
    parser.add_argument('--command-cap',type=int,help='Qualification only; never an official SR trial')
    args=parser.parse_args()
    if min(args.cases,args.candidate_limit)<1 or min(args.seed_start,args.case_index_offset)<0:
        raise ValueError('positive counts and nonnegative identities required')
    if args.phase=='formal' and args.command_cap is not None:
        raise ValueError('formal episodes must retain official task step limits')
    if args.command_cap is not None and args.command_cap<1:
        raise ValueError('positive qualification command cap required')
    plan=json.loads(args.plan.read_text())
    if args.phase=='formal' and (args.seed_start<plan['closed_loop']['reset_seed_start'] or
            args.seed_start>=plan['closed_loop']['reset_seed_start']+plan['closed_loop']['candidate_seed_limit'] or
            args.case_index_offset+args.cases>plan['closed_loop']['accepted_episodes_per_task_per_condition']):
        raise ValueError('formal cohort identity/count disagrees with the frozen plan')
    out=args.output.resolve()
    if out.is_relative_to(ROOT):raise ValueError('runtime evidence must be outside source checkout')
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from scripts import build_warm_rmbench_contract_bundle as builder
    from scripts.run_warm_rmbench_matrix import load_matrix,select_experiments
    from scripts.qualify_rebuttal_rmbench_task import resolved_config,source_identity
    from experiments.rmbench.warm_policy.deploy_policy import RMBenchWarmPolicy
    from fastwam.benchmarks.rmbench import task_by_name
    from fastwam.memory.manifest import sha256_array,sha256_file
    from fastwam.research.evidence import append_record,write_probe
    from fastwam.research.policy_state import deployed_policy_fingerprint

    bundle_args=builder.build_parser().parse_args(json.loads(args.bundle_args.read_text()))
    if len(bundle_args.task)!=1 or bundle_args.experiment!=['full_warm']:
        raise ValueError('one task and the full_warm contract required; null is a declared research intervention')
    task=bundle_args.task[0]
    if task not in ('rearrange_blocks','battery_try'):raise ValueError('unqualified task')
    matrix=load_matrix(bundle_args.matrix);experiment,=select_experiments(matrix,bundle_args.experiment)
    builder.validate_contract_bundle(bundle_args.output_root,experiment_id=experiment.id,task_names=[task])
    artifacts=builder._artifact_paths(bundle_args)
    runtime=builder._runtime_args(bundle_args,artifacts,experiment=experiment,task_name=task,
        contract_path=bundle_args.output_root/experiment.id/f'{task}.json',
        seed_protocol_path=bundle_args.output_root/'seeds'/f'{task}.seed_protocol.npy',
        task_definition_path=bundle_args.rmbench_root/'envs'/f'{task}.py',root_seed=matrix.root_seed)
    expected={'action_horizon':'horizon','replan_steps':'execution_prefix','num_inference_steps':'nfe','warm_top_k':'top_k',
              'warm_recent_event_capacity':'recent_capacity','warm_action_summary_capacity':'summary_capacity'}
    if matrix.root_seed!=3407 or any(runtime[key]!=plan['inference'][name] for key,name in expected.items()):
        raise ValueError('runtime budget/seed differs from the frozen evidence plan')
    checkpoint_sha256=sha256_file(artifacts['warm_checkpoint'])
    if checkpoint_sha256!=plan['checkpoint_sha256']:raise ValueError('author-selected checkpoint identity differs')
    (out/'frozen_plan.json').write_bytes(args.plan.read_bytes())
    runtime_root=args.rmbench_runtime.resolve(strict=True)
    before_source=source_identity(runtime_root)
    for relative,digest in before_source['files'].items():
        if sha256_file(bundle_args.rmbench_root/relative)!=digest:
            raise ValueError('runtime official source differs from the contract checkout: '+relative)
    (out/'runtime_source_manifest.json').write_text(json.dumps(before_source,indent=2))
    os.chdir(runtime_root);sys.path[:0]=[str(runtime_root),str(runtime_root/'description/utils')]
    from generate_episode_instructions import generate_episode_descriptions
    from envs.utils.create_actor import UnStableError
    env=getattr(importlib.import_module('envs.'+task),task)()
    config=resolved_config(runtime_root,task,args.seed_start)
    config.update(eval_mode=True,is_test=True)
    (out/'resolved_simulator_config.json').write_text(json.dumps(config,indent=2))
    policies={};completed=0;next_seed=args.seed_start;started=time.monotonic()
    summary=dict(schema='warm.rebuttal.fresh-reset-pairs.v1',phase=args.phase,status='running',task=task,
        candidate_seed_start=args.seed_start,requested_pairs=args.cases,case_index_offset=args.case_index_offset,
        official_step_limit=task_by_name(task).step_limit,qualified_complete_pairs=0,
        candidate_null='Retrieved source and conditioning withdrawn; native episode recollection retained',
        contract_path=runtime['warm_online_contract_path'],checkpoint_sha256=checkpoint_sha256,
        plan_sha256=sha256_file(args.plan))
    (out/'identity.json').write_text(json.dumps(summary,indent=2))

    def save_observation(path,obs):
        values={name:np.asarray(value['rgb']) for name,value in obs['observation'].items() if 'rgb' in value}
        values['native_qpos']=np.asarray(obs['joint_action']['vector'])
        with path.open('xb') as f:np.savez_compressed(f,**values)
        return {name:sha256_array(value) for name,value in values.items()}

    def initial_task_values():
        if task=='rearrange_blocks':
            return dict(block1=env.block1.get_pose().p.tolist(),block2=env.block2.get_pose().p.tolist(),
                target1=env.target_pose1.p.tolist(),target2=env.target_pose2.p.tolist(),
                stage_id=int(env.stage_id),press_cnt=int(env.press_cnt))
        return dict(battery1_p=env.battery1.get_pose().p.tolist(),battery1_q=env.battery1.get_pose().q.tolist(),
            battery2_p=env.battery2.get_pose().p.tolist(),battery2_q=env.battery2.get_pose().q.tolist(),
            target1=np.asarray(env.target_pose1_p).tolist(),target2=np.asarray(env.target_pose2_p).tolist(),
            correct_combination=list(env.correct_combination))

    try:
        for candidate_number in range(args.candidate_limit):
            seed=args.seed_start+candidate_number;next_seed=seed+1
            config.update(seed=seed,now_ep_num=completed+args.case_index_offset)
            admission={'kind':'seed_admission','seed':seed,'candidate_number':candidate_number}
            print(f'stage=expert_seed_check task={task} seed={seed}',flush=True)
            try:
                env.setup_demo(**config)
                info=env.play_once()
                env.close_env()
                admitted=bool(env.plan_success and env.check_success())
                admission.update(status='accepted' if admitted else 'expert_rejected',plan_success=bool(env.plan_success))
            except UnStableError as exc:
                admission.update(status='unstable_seed',error=str(exc));admitted=False
                env.close_env()
            except Exception as exc:
                append_record(out/'seed_admission.jsonl',{**admission,'status':'invalid_runtime','error':repr(exc)})
                raise
            append_record(out/'seed_admission.jsonl',admission)
            if not admitted:continue
            index=completed+args.case_index_offset
            case=out/f'case-{index:04d}-seed-{seed}';case.mkdir()
            (case/'expert_info.json').write_text(json.dumps(info,indent=2))
            # Fixed before seeing either role's result. Separate owned policy
            # capabilities avoid replaying a consumed BoundOnlineStep.
            if not policies:
                for role in ('full','candidate_null'):
                    policy_args=dict(runtime,warm_telemetry_path=str(out/f'{role}-telemetry.jsonl'))
                    (out/f'{role}-runtime_args.json').write_text(json.dumps(policy_args,indent=2))
                    print('stage=load_policy role='+role,flush=True)
                    policy=RMBenchWarmPolicy(policy_args)
                    policy.model.configure_research_probe(capture=True,force_null=role=='candidate_null')
                    policy._next_episode_index=args.case_index_offset
                    policies[role]=policy
            pair={};reference=None
            # Counterbalance which implementation runs first by frozen case ID.
            role_order=('full','candidate_null') if index%2==0 else ('candidate_null','full')
            for role in role_order:
                policy=policies[role];config.update(seed=seed,now_ep_num=index)
                role_dir=case/role;role_dir.mkdir()
                record=dict(kind='episode_attempt',task=task,episode_id=index,seed=seed,role=role,
                    phase=args.phase,status='attempted',success=None,executed_commands=0)
                append_record(out/'events.jsonl',record)
                commands=[];qpos_before=[];qpos_after=[]
                try:
                    env.setup_demo(**config)
                    descriptions=generate_episode_descriptions(task,[info['info']],100)
                    instruction=str(np.random.choice(descriptions[0]['unseen']))
                    env.set_instruction(instruction=instruction)
                    obs=env.get_obs();initial_images=save_observation(role_dir/'initial_observation.npz',obs)
                    identity={'observation_arrays':initial_images,'task_state':initial_task_values(),'instruction':instruction}
                    (role_dir/'initial_identity.json').write_text(json.dumps(identity,indent=2))
                    if reference is None:reference=identity
                    elif identity!=reference:raise RuntimeError('fresh-reset role pair has different initial state, cameras or instruction')
                    policy.reset()
                    if policy.controller.active_episode_index!=index:raise RuntimeError('policy episode cursor does not match the pair')
                    limit=env.step_lim if args.command_cap is None else min(env.step_lim,args.command_cap)
                    replan_ordinal=0
                    while env.take_action_cnt<limit and not env.eval_success:
                        obs=env.get_obs()
                        if not policy.queue:
                            frame=policy._executed_policy_actions
                            policy._replan(obs,env)
                            probe=policy.model._last_research_probe
                            if probe is None:raise RuntimeError('missing pre-execution research capture')
                            if role=='candidate_null' and bool(probe['arrays']['g'].any()):
                                raise RuntimeError('candidate-null generated a nonzero retrieved influence')
                            capture=None
                            if replan_ordinal in plan['local_recovery']['query_replan_ordinals']:
                                capture=write_probe(role_dir/'probes',f'e{index:04d}-f{frame:04d}',probe,
                                    dict(task=task,role=role,seed=seed,frame_index=frame,checkpoint_sha256=checkpoint_sha256,
                                         policy_state_sha256=deployed_policy_fingerprint(policy)))
                                save_observation(role_dir/f'replan-{replan_ordinal:04d}.npz',obs)
                            append_record(role_dir/'replan_progress.jsonl',dict(frame_index=frame,replan_ordinal=replan_ordinal,
                                gate=float(probe['arrays']['g'].reshape(-1)[0]),task_state=initial_task_values(),capture=capture))
                            replan_ordinal+=1
                        queued=policy.queue._items[0]
                        commands.append(np.array(queued.environment_space,copy=True))
                        qpos_before.append(np.array(obs['joint_action']['vector'],copy=True))
                        previous=env.take_action_cnt
                        policy.step(env,obs)
                        if env.take_action_cnt!=previous+1:raise RuntimeError('official executor did not advance exactly one qpos command')
                        qpos_after.append(np.array(env.get_obs()['joint_action']['vector'],copy=True))
                        record['executed_commands']+=1
                    complete=bool(env.eval_success or env.take_action_cnt>=env.step_lim)
                    record.update(status='complete' if complete else 'qualification_censored',
                        success=int(bool(env.eval_success)) if complete else None,max_reward=float(env.max_reward),
                        final_policy_state_sha256=deployed_policy_fingerprint(policy))
                    save_observation(role_dir/'final_observation.npz',env.get_obs())
                    policy.seal_active(reason='success' if env.eval_success else 'step_limit' if complete else 'qualification_command_cap',success=bool(env.eval_success))
                except Exception as exc:
                    record.update(status='invalid_runtime',error=repr(exc),traceback=traceback.format_exc())
                    raise
                finally:
                    with (role_dir/'qpos_trace.npz').open('xb') as f:
                        np.savez_compressed(f,commands=np.asarray(commands),before=np.asarray(qpos_before),after=np.asarray(qpos_after))
                    (role_dir/'result.json').write_text(json.dumps(record,indent=2))
                    append_record(out/'events.jsonl',record)
                    env.close_env(clear_cache=True)
                pair[role]=record
            valid=all(r['status']=='complete' for r in pair.values())
            append_record(out/'pairs.jsonl',dict(task=task,episode_id=index,seed=seed,phase=args.phase,
                status='complete' if valid else 'qualification_censored',roles=pair,
                full_minus_null=pair['full']['success']-pair['candidate_null']['success'] if valid else None))
            completed+=1;summary['qualified_complete_pairs']+=int(valid)
            summary.update(completed_pairs=completed,next_candidate_seed=next_seed)
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            print(json.dumps(dict(stage='pair_complete',index=index,seed=seed,roles={k:v['success'] for k,v in pair.items()})),flush=True)
            if completed==args.cases:break
        if completed!=args.cases:raise RuntimeError('candidate-seed budget exhausted; do not silently change the cohort')
        summary['status']='complete'
    except BaseException as exc:
        summary.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        for policy in policies.values():policy.close()
        summary.update(completed_pairs=completed,next_candidate_seed=next_seed,elapsed_seconds=time.monotonic()-started,
            runtime_source_sha256=before_source['sha256'],runtime_source_sha256_after=source_identity(runtime_root)['sha256'])
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    if summary['runtime_source_sha256']!=summary['runtime_source_sha256_after']:
        raise RuntimeError('official simulator source changed')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
