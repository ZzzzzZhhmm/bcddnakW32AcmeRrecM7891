#!/usr/bin/env python3
"""Seal actual-policy raw/refined/null actions and execute same-state branches.

This first entry qualifies actual history, capability ownership and deterministic
restoration on an excluded pilot. Formal cohorts require separate stage admission.
The original bound core/config root and the collector extension are recorded.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,fields
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core-code',type=Path,required=True)
    p.add_argument('--bundle-args',type=Path,required=True)
    p.add_argument('--rmbench-runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--trajectory-commands',type=int,default=8)
    p.add_argument('--query-ordinals',type=int,nargs='+',default=[1])
    args=p.parse_args()
    if args.trajectory_commands<8 or any(q<1 for q in args.query_ordinals):
        raise ValueError('actual factual history and a later query are required')
    core=args.core_code.resolve(strict=True);out=args.output.resolve()
    if out.is_relative_to(ROOT) or out.is_relative_to(core):raise ValueError('evidence must be outside source trees')
    out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(core/'src'),str(core)]
    import fastwam.research
    fastwam.research.__path__.append(str(ROOT/'src/fastwam/research'))
    import numpy as np
    import torch
    from scripts import build_warm_rmbench_contract_bundle as builder
    from scripts.run_warm_rmbench_matrix import load_matrix,select_experiments
    from scripts.qualify_rebuttal_rmbench_task import resolved_config,source_identity as simulator_identity
    from scripts.nonreal_job import source_identity
    from experiments.rmbench.warm_policy.deploy_policy import RMBenchWarmPolicy
    from experiments.rmbench.warm_policy.policy_core import factual_cameras
    from fastwam.memory.manifest import sha256_array,sha256_file
    from fastwam.research.evidence import append_record,write_probe,canonical
    from fastwam.research.branches import execute_branches
    from fastwam.research.rmbench_branches import RMBenchQposBackend,prepare_native_query
    from fastwam.research.policy_state import deployed_policy_fingerprint,digest_state
    from fastwam.research.query_capture import capture_online_replan,generate_captured_null
    from fastwam.research.rmbench_outcomes import observe_task_progress,label_next_interaction
    from fastwam.models.warm.retrospection_model import _spatial_effect_summary

    if Path(builder.__file__).resolve()!=core/'scripts/build_warm_rmbench_contract_bundle.py':
        raise RuntimeError('wrong bound core/config root imported')
    a=builder.build_parser().parse_args(json.loads(args.bundle_args.read_text()))
    if len(a.task)!=1 or a.experiment!=['full_warm']:raise ValueError('one Full task contract required')
    task=a.task[0];matrix=load_matrix(a.matrix);experiment,=select_experiments(matrix,a.experiment)
    builder.validate_contract_bundle(a.output_root,experiment_id=experiment.id,task_names=[task])
    artifacts=builder._artifact_paths(a)
    runtime=builder._runtime_args(a,artifacts,experiment=experiment,task_name=task,
        contract_path=a.output_root/experiment.id/f'{task}.json',seed_protocol_path=a.output_root/'seeds'/f'{task}.seed_protocol.npy',
        task_definition_path=a.rmbench_root/'envs'/f'{task}.py',root_seed=matrix.root_seed)
    runtime_root=args.rmbench_runtime.resolve(strict=True)
    source=source_identity(core)
    identity=dict(core_source_sha256=source['sha256'],collector_source_sha256=source_identity(ROOT)['sha256'],
        simulator_source=simulator_identity(runtime_root),checkpoint_sha256=sha256_file(artifacts['warm_checkpoint']),
        task=task,seed=args.seed,phase='excluded_actual-history_restoration_pilot',horizon=32,execution_prefix=4,
        planned_query_ordinals=args.query_ordinals,trajectory_commands=args.trajectory_commands)
    (out/'identity.json').write_bytes(canonical(identity))
    os.chdir(runtime_root);sys.path[:0]=[str(runtime_root),str(runtime_root/'description/utils')]
    from generate_episode_instructions import generate_episode_descriptions
    env=getattr(importlib.import_module('envs.'+task),task)()
    config=resolved_config(runtime_root,task,args.seed);policies={};collected=0;started=time.monotonic()
    report=dict(status='running',scientific_collection_qualified=False,queries=0,independent_branch_labels=0)
    try:
        random.seed(args.seed);env.setup_demo(**config);info=env.play_once();env.close_env()
        if not env.plan_success or not env.check_success():raise RuntimeError('prespecified pilot expert seed rejected')
        random.seed(args.seed);env.setup_demo(**config)
        instruction=str(np.random.choice(generate_episode_descriptions(task,[info['info']],100)[0]['unseen']))
        env.set_instruction(instruction=instruction)
        for role in ('full','candidate_null'):
            print('stage=load_policy role='+role,flush=True)
            policy=RMBenchWarmPolicy(dict(runtime,warm_telemetry_path=str(out/f'{role}-telemetry.jsonl')))
            policy.model.configure_research_probe(capture=True,force_null=role=='candidate_null')
            policies[role]=policy
        full,null=policies['full'],policies['candidate_null'];full.reset()
        ordinal=0
        while env.take_action_cnt<args.trajectory_commands and not env.eval_success:
            obs=env.get_obs();capture=None
            if not full.queue:
                preparation=[]
                obs,proof=prepare_native_query(env,emit=lambda r:preparation.append(r))
                with capture_online_replan(full) as capture:full._replan(obs,env)
                if ordinal in args.query_ordinals:
                    query_id=f'pilot-{task}-seed{args.seed}-q{ordinal:04d}'
                    qdir=out/query_id;qdir.mkdir()
                    (qdir/'preparation.pickle').write_bytes(pickle.dumps(preparation,protocol=4))
                    (qdir/'preparation.json').write_bytes(canonical(proof))
                    history=full.controller.retrospective_episode_memory._memory.snapshot()
                    if history.counters.observation_updates<1:raise RuntimeError('live factual history was not populated')
                    (qdir/'actual_policy_state.json').write_text(history.to_json())
                    alternative=generate_captured_null(full,null,capture)
                    probe=capture['probe'];idx=int(probe['arrays']['selected_index'].reshape(-1)[0])
                    if idx<0:raise RuntimeError('pilot has no raw selected proposal; do not substitute another query')
                    raw=probe['arrays']['adapted_actions'][0,idx].float().cpu().numpy()
                    model_actions={'raw':raw,'refined':capture['action'],'candidate_null':alternative['action'],
                                   'refined_repeat':capture['action'].copy()}
                    commands={}
                    for role,action in model_actions.items():
                        native=full.action_normalizer.denormalize(action)
                        returned=full.action_normalizer.normalize(native,fail_on_clip=True)
                        if not np.allclose(returned,action,rtol=1e-5,atol=2e-6):
                            raise RuntimeError('shared action normalization round trip failed: '+role)
                        commands[role]=native
                    np.savez_compressed(qdir/'sealed_actions.npz',**commands)
                    step=capture['online_step']
                    query_identity=dict(identity,bound_step_sha256=step.step_sha256,
                        candidate_payload_sha256=step.candidate_payload_sha256,
                        model_input_sha256=step.model_input_sha256,derived_seed=int(step.derived_seed),
                        candidate_event_ids=[None if event is None else asdict(event) for event in step.event_ids],
                        full_policy_sha256_after_null=alternative['full_policy_state_sha256'])
                    context_values={field.name:getattr(capture['context'],field.name) for field in fields(capture['context'])}
                    context_arrays={name:value.detach().cpu() for name,value in context_values.items() if torch.is_tensor(value)}
                    torch.save(context_arrays,qdir/'sealed_factual_context.pt')
                    (qdir/'context_non_tensor_fields.json').write_bytes(canonical({name:value for name,value in context_values.items() if not torch.is_tensor(value)}))
                    proposal=write_probe(qdir/'probes','full',probe,query_identity)
                    write_probe(qdir/'probes','candidate-null',alternative['probe'],query_identity)
                    sealed=dict(identity,query_id=query_id,gate=float(probe['arrays']['g'].reshape(-1)[0]),
                        native_actions_sha256=sha256_file(qdir/'sealed_actions.npz'),
                        factual_context_sha256=sha256_file(qdir/'sealed_factual_context.pt'),
                        source_probe=proposal,history_sha256=history.sha256,history_observation_updates=history.counters.observation_updates,
                        before_any_branch_truth=True,normalization_stats_sha256=full.contract.normalization_stats_sha256)
                    (qdir/'sealed_query.json').write_bytes(canonical(sealed));seal_sha=sha256_file(qdir/'sealed_query.json')
                    def encode(observation):
                        retriever=full.retriever
                        prepared=retriever._image_adapter.prepare(factual_cameras(observation))
                        features=retriever._dino.encode(prepared.camera_frames[retriever._semantic_processor_camera],batch_size=retriever._dino_batch_size)
                        return np.asarray(features.spatial[0],dtype=np.float32)
                    fingerprint=lambda:digest_state([deployed_policy_fingerprint(full),deployed_policy_fingerprint(null)])
                    backend=RMBenchQposBackend(env,obs,encode=encode,policy_fingerprint=fingerprint,canonicalization=proof)
                    parent_outcome=observe_task_progress(env);endpoints=[]
                    def observe_outcome():
                        outcome=observe_task_progress(env)
                        observed={'task_progress':outcome,'endpoint_fingerprint':dict(backend.fingerprint()),
                            'rgb_sha256':{k:sha256_array(v['rgb']) for k,v in backend.obs['observation'].items() if 'rgb' in v}}
                        (qdir/f'endpoint-{len(endpoints):02d}.pickle').write_bytes(pickle.dumps(backend.snapshot(),protocol=4))
                        endpoints.append(observed)
                        return observed
                    try:
                        print('stage=sealed_branch_execution query_id='+query_id,flush=True)
                        results=execute_branches(backend,query_id=query_id,candidates=commands,proposal_sha256=seal_sha,
                            horizon=32,emit=lambda row:append_record(qdir/'branches.jsonl',row),
                            project_effect=lambda before,delta:_spatial_effect_summary(torch.from_numpy(delta)[None],torch.from_numpy(before)[None])[0].numpy(),
                            observe_outcome=observe_outcome)
                    finally:backend.close()
                    # Repeated refined actions must reproduce all saved endpoint
                    # components and RGB; do not derive tolerances from outcomes.
                    if endpoints[1]['endpoint_fingerprint']!=endpoints[3]['endpoint_fingerprint'] or endpoints[1]['rgb_sha256']!=endpoints[3]['rgb_sha256']:
                        raise RuntimeError('actual-model refined branch repeat is not exact')
                    labels={r['candidate_id']:label_next_interaction(parent_outcome,r['independent_outcome']['task_progress'],
                        valid_branch=r['endpoint_status']!='invalid_branch') for r in results[:3]}
                    row=dict(schema='warm.recovery.query.v1',task=task,episode_id=f'pilot-{args.seed}',query_id=query_id,
                        protocol_sha256=sha256_file(ROOT/'configs/nonreal/rebuttal_p0_evidence_plan.json'),
                        checkpoint_sha256=full.checkpoint_sha256,cohort='excluded_live_history_pilot',horizon=32,
                        gate=sealed['gate'],gate_intervention='natural',branches=labels,
                        exact_refined_repeat=True,raw_evidence=str(qdir))
                    append_record(out/'recovery_queries.jsonl',row)
                    collected+=1;print(json.dumps(dict(stage='branch_query_complete',query_id=query_id,labels=labels)),flush=True)
                ordinal+=1
            full.step(env,obs)
        if collected!=len(args.query_ordinals):raise RuntimeError('planned pilot queries not all reached; no replacement allowed')
        report.update(status='qualified_actual_history_queries_only',queries=collected,
            independent_branch_labels=3*collected,all_planned_scientific_stages_qualified=False)
    except BaseException as exc:
        report.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        for policy in policies.values():policy.close()
        if hasattr(env,'scene'):env.close_env(clear_cache=True)
        report.update(elapsed_seconds=time.monotonic()-started,core_source_sha256_after=source_identity(core)['sha256'])
        (out/'summary.json').write_bytes(canonical(report))
    if report['core_source_sha256_after']!=source['sha256']:raise RuntimeError('bound core changed')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
