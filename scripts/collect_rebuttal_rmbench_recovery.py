#!/usr/bin/env python3
"""Seal actual-policy raw/refined/null actions and execute same-state branches.

This first entry qualifies actual history, capability ownership and deterministic
restoration on an excluded pilot. Formal cohorts require separate stage admission.
The original bound core/config root and the collector extension are recorded.
"""
from __future__ import annotations
import argparse
from contextlib import closing
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
from types import SimpleNamespace

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
    p.add_argument('--null-execution',choices=('independent','shared_weights'),default='independent')
    p.add_argument('--candidate-pool',action='store_true',help='Also execute all valid original K32 adapted proposals from each sealed parent')
    args=p.parse_args()
    if args.trajectory_commands<8 or any(q<1 for q in args.query_ordinals):
        raise ValueError('actual factual history and a later query are required')
    core=args.core_code.resolve(strict=True);out=args.output.resolve()
    if out.is_relative_to(ROOT) or out.is_relative_to(core):raise ValueError('evidence must be outside source trees')
    out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(core/'src'),str(core)]
    import fastwam.research
    fastwam.research.__path__.insert(0,str(ROOT/'src/fastwam/research'))
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
    from fastwam.research.inference_replica import shared_weight_inference_replica,assert_shared_weights_unchanged
    from fastwam.research.policy_state import THREAD_FIELDS
    from fastwam.research.candidate_pool import decode_branch_actions,prepare_candidate_pool,centered_effect_metrics
    from scripts.assemble_nonreal_candidates import assemble
    from fastwam.research.query_capture import capture_online_replan,generate_captured_null
    from fastwam.research.rmbench_outcomes import observe_task_progress,label_next_interaction
    from fastwam.models.warm.retrospection_model import _spatial_effect_summary

    if Path(builder.__file__).resolve()!=core/'scripts/build_warm_rmbench_contract_bundle.py':
        raise RuntimeError('wrong bound core/config root imported')
    if Path(sys.modules['fastwam.research.query_capture'].__file__).resolve()!=ROOT/'src/fastwam/research/query_capture.py':
        raise RuntimeError('wrong explicitly recorded research extension imported')
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
        planned_query_ordinals=args.query_ordinals,trajectory_commands=args.trajectory_commands,null_execution=args.null_execution,
        execute_candidate_pool=args.candidate_pool)
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
        roles=('full','candidate_null') if args.null_execution=='independent' else ('full',)
        for role in roles:
            print('stage=load_policy role='+role,flush=True)
            policy=RMBenchWarmPolicy(dict(runtime,warm_telemetry_path=str(out/f'{role}-telemetry.jsonl')))
            policy.model.configure_research_probe(capture=True,force_null=role=='candidate_null')
            policies[role]=policy
        full=policies['full'];shared_receipt=None
        if args.null_execution=='shared_weights':
            allocated_before=torch.cuda.memory_allocated()
            null_model,shared_receipt=shared_weight_inference_replica(full.model)
            null_model.configure_research_probe(capture=True,force_null=True)
            null=SimpleNamespace(model=null_model,checkpoint_sha256=full.checkpoint_sha256)
            (out/'shared_weights_receipt.json').write_bytes(canonical(dict(
                mode='serial_readonly_shared_weights_independent_module_state',
                shared_tensor_references=len(shared_receipt),
                allocated_before=allocated_before,allocated_after=torch.cuda.memory_allocated(),
                native_independent_model_equivalence_requires_separate_qualification=True)))
        else:null=policies['candidate_null']
        full.reset()

        def policy_fingerprint():
            if shared_receipt is None:return digest_state([deployed_policy_fingerprint(full),deployed_policy_fingerprint(null)])
            assert_shared_weights_unchanged(shared_receipt)
            return digest_state([deployed_policy_fingerprint(full),
                {name:getattr(null.model,name) for name in THREAD_FIELDS},
                null.model._warm_online_experiment_locked,null.model._warm_research_probe])
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
                    if shared_receipt is not None:assert_shared_weights_unchanged(shared_receipt)
                    probe=capture['probe'];idx=int(probe['arrays']['selected_index'].reshape(-1)[0])
                    if idx<0:raise RuntimeError('pilot has no raw selected proposal; do not substitute another query')
                    raw=probe['arrays']['adapted_actions'][0,idx].float().cpu().numpy()
                    model_actions={'raw':raw,'refined':capture['action'],'candidate_null':alternative['action'],
                                   'refined_repeat':capture['action'].copy()}
                    commands={}
                    for role,action in model_actions.items():
                        commands[role]=decode_branch_actions(full.action_normalizer,action)
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
                    pool_commands=None
                    if args.candidate_pool:
                        pool_commands,pool_native,pool_evidence=prepare_candidate_pool(probe,full.action_normalizer)
                        # This second index uses the actual query ID expected
                        # by the candidate assembler, also before any truth.
                        pool_probe=write_probe(qdir/'candidate-pool-probe',query_id,probe,
                            dict(query_identity,event_ids=query_identity['candidate_event_ids']))
                        np.savez_compressed(qdir/'sealed_pool_native_actions.npz',actions=pool_native)
                        pool_evidence.update(native_actions_sha256=sha256_file(qdir/'sealed_pool_native_actions.npz'),
                            probe_metadata_sha256=pool_probe['metadata_sha256'],before_any_branch_truth=True)
                        (qdir/'sealed_candidate_pool.json').write_bytes(canonical(pool_evidence))
                        pool_index=dict(task=task,episode_id=f'pilot-{args.seed}',query_id=query_id,
                            cohort='excluded_live_history_candidate_pool_qualification',probe_path=pool_probe['path'],
                            probe_metadata_sha256=pool_probe['metadata_sha256'])
                        append_record(out/'candidate_index.jsonl',pool_index)
                    sealed=dict(identity,query_id=query_id,gate=float(probe['arrays']['g'].reshape(-1)[0]),
                        raw_definition='Selected current-state adapted proposal immediately before source mixing/flow; not unwarped stored qpos.',
                        factual_rng_isolated=alternative['factual_rng_isolated'],
                        native_actions_sha256=sha256_file(qdir/'sealed_actions.npz'),
                        factual_context_sha256=sha256_file(qdir/'sealed_factual_context.pt'),
                        source_probe=proposal,history_sha256=history.sha256,history_observation_updates=history.counters.observation_updates,
                        before_any_branch_truth=True,normalization_stats_sha256=full.contract.normalization_stats_sha256)
                    (qdir/'sealed_query.json').write_bytes(canonical(sealed));seal_sha=sha256_file(qdir/'sealed_query.json')
                    def encode(observation):
                        retriever=full.retriever
                        raw_batch,_=retriever._prepare_raw_cameras(factual_cameras(observation))
                        prepared=retriever._image_adapter.prepare(raw_batch)
                        features=retriever._dino.encode(prepared.camera_frames[retriever._semantic_processor_camera],batch_size=retriever._dino_batch_size)
                        return np.asarray(features.spatial[0],dtype=np.float32)
                    with closing(RMBenchQposBackend(env,obs,encode=encode,policy_fingerprint=policy_fingerprint,canonicalization=proof)) as backend:
                        (qdir/'parent_state.pickle').write_bytes(pickle.dumps(backend.snapshot(),protocol=4))
                        before_read=dict(backend.fingerprint());parent_features=encode(obs)
                        if dict(backend.fingerprint())!=before_read or not np.array_equal(parent_features,step.factual_world_tokens):
                            raise RuntimeError('independent effect reader differs from actual retrieval tokens or mutated parent')
                        np.save(qdir/'parent_dino_spatial.npy',parent_features,allow_pickle=False)
                        outcome_parent_fingerprint=dict(backend.fingerprint())
                        parent_outcome=observe_task_progress(env)
                        if dict(backend.fingerprint())!=outcome_parent_fingerprint:
                            raise RuntimeError('independent parent outcome reader mutated live state')
                        (qdir/'parent_outcome.json').write_bytes(canonical(parent_outcome))
                        (qdir/'parent_receipt.json').write_bytes(canonical(dict(sealed_query_sha256=seal_sha,
                            parent_state_sha256=sha256_file(qdir/'parent_state.pickle'),
                            parent_dino_spatial_sha256=sha256_file(qdir/'parent_dino_spatial.npy'),
                            parent_outcome_sha256=sha256_file(qdir/'parent_outcome.json'),
                            exact_retrieval_spatial_tokens=True,before_any_branch_truth=True)))
                        endpoints=[]
                        def observe_outcome():
                            outcome=observe_task_progress(env)
                            observed={'task_progress':outcome,'endpoint_fingerprint':dict(backend.fingerprint()),
                                'rgb_sha256':{k:sha256_array(v['rgb']) for k,v in backend.obs['observation'].items() if 'rgb' in v}}
                            (qdir/f'endpoint-{len(endpoints):02d}.pickle').write_bytes(pickle.dumps(backend.snapshot(),protocol=4))
                            endpoints.append(observed)
                            return observed
                        print('stage=sealed_branch_execution query_id='+query_id,flush=True)
                        results=execute_branches(backend,query_id=query_id,candidates=commands,proposal_sha256=seal_sha,
                            horizon=32,emit=lambda row:append_record(qdir/'branches.jsonl',row),
                            project_effect=lambda before,delta:_spatial_effect_summary(torch.from_numpy(delta)[None],torch.from_numpy(before)[None])[0].numpy(),
                            observe_outcome=observe_outcome)
                        if endpoints[1]['endpoint_fingerprint']!=endpoints[3]['endpoint_fingerprint'] or endpoints[1]['rgb_sha256']!=endpoints[3]['rgb_sha256']:
                            raise RuntimeError('actual-model refined branch repeat is not exact')
                        if pool_commands is not None:
                            print('stage=sealed_K32_execution query_id='+query_id,flush=True)
                            pool_trace=[]
                            def emit_pool(row):
                                pool_trace.append(row);append_record(out/'candidate_branches.jsonl',row)
                            pool_results=execute_branches(backend,query_id=query_id,candidates=pool_commands,
                                proposal_sha256=pool_probe['metadata_sha256'],horizon=32,emit=emit_pool,
                                project_effect=lambda before,delta:_spatial_effect_summary(torch.from_numpy(delta)[None],torch.from_numpy(before)[None])[0].numpy(),
                                observe_outcome=observe_outcome,continue_invalid_execution=True)
                            selected_result=next((r for r in pool_results if r['candidate_id']==f'rank-{idx}'),None)
                            if selected_result is None or selected_result['independent_outcome']!=results[0]['independent_outcome']:
                                raise RuntimeError('selected pool proposal did not reproduce the earlier raw endpoint')
                            pool_labels=[]
                            for result in pool_results:
                                if result['endpoint_status']=='invalid_branch':
                                    label=dict(applicable=None,label_provenance=parent_outcome['rubric'],
                                        reason='Known executor rejection; no complete endpoint or imputed label.',
                                        terminal_success=None)
                                else:
                                    label=label_next_interaction(parent_outcome,result['independent_outcome']['task_progress'],valid_branch=True)
                                row=dict(query_id=query_id,candidate_id=result['candidate_id'],**label)
                                pool_labels.append(row);append_record(out/'candidate_labels.jsonl',row)
                            assembled,=assemble([pool_index],pool_trace,pool_labels)
                            append_record(out/'candidate_effects.jsonl',assembled)
                            centered=dict(task=task,episode_id=pool_index['episode_id'],query_id=query_id,
                                **centered_effect_metrics(assembled))
                            append_record(out/'centered_effects.jsonl',centered)
                            print(json.dumps(dict(stage='candidate_pool_complete',query_id=query_id,
                                complete_horizons=sum(r['endpoint_status']=='complete_horizon' for r in pool_results),
                                independently_known_labels=sum(r['applicable'] is not None for r in pool_labels),
                                observed_centered_variance=centered['observed_centered_variance'])),flush=True)
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
