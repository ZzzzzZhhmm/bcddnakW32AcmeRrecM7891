#!/usr/bin/env python3
"""Seal actual-policy raw/refined/null actions and execute same-state branches.

Formal collection admits fixed source episode IDs from complete Full trials and
checks the instrumented factual trajectory against their original native traces.
The original bound core/config root and collector extension are recorded.
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
    p.add_argument('--phase',choices=('qualification','formal'),default='qualification')
    p.add_argument('--plan',type=Path)
    p.add_argument('--formal-source',type=Path,help='Completed official Full worker directory; admission never selects by its success value')
    p.add_argument('--episode-index',type=int,default=0)
    p.add_argument('--trajectory-commands',type=int)
    p.add_argument('--query-ordinals',type=int,nargs='+')
    p.add_argument('--null-execution',choices=('independent','shared_weights'),default='independent')
    p.add_argument('--candidate-pool',action='store_true',help='Also execute all valid original K32 adapted proposals from each sealed parent')
    p.add_argument('--factual-replay-only',action='store_true',help='Engineering witness only: reproduce the complete fixed factual source without state round-trips or counterfactual branches')
    p.add_argument('--seal-null-on-factual-replay',action='store_true',help='Also seal same-context/null H32 actions without executing any counterfactual branch')
    args=p.parse_args()
    if args.seal_null_on_factual_replay and (not args.factual_replay_only or args.phase!='formal' or args.null_execution!='shared_weights'):
        raise ValueError('null sealing requires the fixed formal factual witness and separately qualified read-only shared weights')
    if args.phase=='formal':
        if args.plan is None or args.formal_source is None or args.trajectory_commands is not None or args.query_ordinals is not None or args.candidate_pool:
            raise ValueError('formal sources require a frozen plan; command caps and query/pool overrides are forbidden')
    else:
        if args.factual_replay_only:raise ValueError('factual replay witnesses require a fixed complete formal source')
        if args.formal_source is not None or args.episode_index!=0:raise ValueError('formal source identities are not excluded pilots')
        args.trajectory_commands=8 if args.trajectory_commands is None else args.trajectory_commands
        args.query_ordinals=[1] if args.query_ordinals is None else args.query_ordinals
        if args.trajectory_commands<8 or any(q<1 for q in args.query_ordinals):
            raise ValueError('actual factual history and a later qualification query are required')
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
    from fastwam.research.recovery_protocol import admit_formal_recovery
    from fastwam.research.resumable_pairs import read_committed_trials,verify_role
    from scripts.assemble_nonreal_candidates import assemble
    from fastwam.research.query_capture import capture_online_replan,generate_captured_null,require_original_generation
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
    runtime_identity=simulator_identity(runtime_root)
    for relative,digest in runtime_identity['files'].items():
        if sha256_file(a.rmbench_root/relative)!=digest:
            raise ValueError('runtime official source differs from bound checkout: '+relative)
    source=source_identity(core)
    formal=args.phase=='formal';admission=None;reference_trace=None;query_outcomes={};source_replans={}
    if formal:
        plan=json.loads(args.plan.read_text());source_dir=args.formal_source.resolve(strict=True)
        source_identity_row=json.loads((source_dir/'identity.json').read_text())
        trials=read_committed_trials(source_dir)
        admission=admit_formal_recovery(plan,source_identity_row,trials,task=task,seed=args.seed,
            episode_index=args.episode_index,plan_sha256=sha256_file(args.plan),core_source_sha256=source['sha256'],
            search_condition='full_bank')
        source_row,=[row for row in trials if row['episode_id']==args.episode_index and row['seed']==args.seed]
        verified_source=verify_role(source_dir,source_row,'full',dict(task=task,
            checkpoint_sha256=plan['checkpoint_sha256'],evidence_plan_sha256=sha256_file(args.plan),
            core_source_sha256=source['sha256'],runtime_source_sha256=runtime_identity['sha256'],
            official_step_limit={'rearrange_blocks':700,'battery_try':1000}[task]),{args.episode_index:args.seed})
        args.query_ordinals=admission['query_ordinals'];pool_query_ordinals=admission['candidate_pool_query_ordinals']
        source_case=source_dir/f'case-{args.episode_index:04d}-seed-{args.seed}'/'full'
        if args.seal_null_on_factual_replay:
            source_telemetry=source_dir/'full-telemetry.jsonl'
            for line in source_telemetry.read_text().splitlines():
                row=json.loads(line)
                if row.get('kind')=='replan' and row.get('episode_index')==args.episode_index:
                    frame=row['frame_index']
                    if frame in source_replans:raise ValueError('duplicate original factual replan evidence')
                    source_replans[frame]=row
        with np.load(source_case/'qpos_trace.npz',allow_pickle=False) as data:reference_trace={key:data[key].copy() for key in data.files}
        reference_initial=json.loads((source_case/'initial_identity.json').read_text())
        expected={'action_horizon':'horizon','replan_steps':'execution_prefix','num_inference_steps':'nfe','warm_top_k':'top_k',
            'warm_recent_event_capacity':'recent_capacity','warm_action_summary_capacity':'summary_capacity'}
        if matrix.root_seed!=3407 or any(runtime[key]!=plan['inference'][name] for key,name in expected.items()):
            raise ValueError('formal runtime budget differs from the frozen plan')
        if sha256_file(artifacts['warm_checkpoint'])!=plan['checkpoint_sha256']:raise ValueError('formal checkpoint differs')
        (out/'frozen_plan.json').write_bytes(args.plan.read_bytes())
        (out/'formal_admission.json').write_bytes(canonical(dict(admission,source=str(source_dir),
            source_identity_sha256=sha256_file(source_dir/'identity.json'),source_trace_sha256=sha256_file(source_case/'qpos_trace.npz'),
            verified_source_files=verified_source['files'],
            use='Reference trace is checked after generation only; no teacher forcing or model inputs from future actions.')))
    else:pool_query_ordinals=args.query_ordinals if args.candidate_pool else []
    episode_id=args.episode_index if formal else f'pilot-{args.seed}'
    cohort='fixed_full_factual_recovery' if formal else 'excluded_live_history_pilot'
    identity=dict(core_source_sha256=source['sha256'],collector_source_sha256=source_identity(ROOT)['sha256'],
        simulator_source=runtime_identity,checkpoint_sha256=sha256_file(artifacts['warm_checkpoint']),
        task=task,seed=args.seed,phase='formal' if formal else 'excluded_actual-history_restoration_pilot',horizon=32,execution_prefix=4,
        planned_query_ordinals=args.query_ordinals,trajectory_commands=args.trajectory_commands,null_execution=args.null_execution,
        candidate_pool_query_ordinals=pool_query_ordinals,episode_id=episode_id,
        factual_replay_only=args.factual_replay_only,
        seal_null_on_factual_replay=args.seal_null_on_factual_replay,
        plan_sha256=sha256_file(args.plan) if formal else sha256_file(ROOT/'configs/nonreal/rebuttal_p0_evidence_plan.json'))
    (out/'identity.json').write_bytes(canonical(identity))
    os.chdir(runtime_root);sys.path[:0]=[str(runtime_root),str(runtime_root/'description/utils')]
    from generate_episode_instructions import generate_episode_descriptions
    env=getattr(importlib.import_module('envs.'+task),task)()
    config=resolved_config(runtime_root,task,args.seed);policies={};collected=0;started=time.monotonic()
    config['now_ep_num']=args.episode_index
    report=dict(status='running',scientific_collection_qualified=False,queries=0,independent_branch_labels=0)
    factual_commands=[];factual_before=[];factual_after=[]
    try:
        random.seed(args.seed);env.setup_demo(**config);info=env.play_once();env.close_env()
        if not env.plan_success or not env.check_success():raise RuntimeError('prespecified pilot expert seed rejected')
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
        # Match the original source lifecycle: construct every model before the
        # fresh role reset. Model initialization must not consume episode RNG.
        random.seed(args.seed);env.setup_demo(**config)
        instruction=str(np.random.choice(generate_episode_descriptions(task,[info['info']],100)[0]['unseen']))
        env.set_instruction(instruction=instruction)
        if formal:
            initial=env.get_obs()
            images={name:sha256_array(np.asarray(value['rgb'])) for name,value in initial['observation'].items() if 'rgb' in value}
            images['native_qpos']=sha256_array(np.asarray(initial['joint_action']['vector']))
            if task=='rearrange_blocks':
                logical=dict(block1=env.block1.get_pose().p.tolist(),block2=env.block2.get_pose().p.tolist(),
                    target1=env.target_pose1.p.tolist(),target2=env.target_pose2.p.tolist(),
                    stage_id=int(env.stage_id),press_cnt=int(env.press_cnt))
            else:
                logical=dict(battery1_p=env.battery1.get_pose().p.tolist(),battery1_q=env.battery1.get_pose().q.tolist(),
                    battery2_p=env.battery2.get_pose().p.tolist(),battery2_q=env.battery2.get_pose().q.tolist(),
                    target1=np.asarray(env.target_pose1_p).tolist(),target2=np.asarray(env.target_pose2_p).tolist(),
                    correct_combination=list(env.correct_combination))
            if images!=reference_initial['observation_arrays'] or instruction!=reference_initial['instruction'] or logical!=reference_initial['task_state']:
                raise RuntimeError('formal recovery reset differs from sealed source observations, logical task state or instruction')
            args.trajectory_commands=env.step_lim
        full._next_episode_index=args.episode_index
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
                if not args.factual_replay_only:
                    obs,proof=prepare_native_query(env,emit=lambda r:preparation.append(r))
                with capture_online_replan(full) as capture:full._replan(obs,env)
                if formal:
                    position=len(factual_commands)
                    if (position>=len(reference_trace['commands']) or
                        not np.array_equal(full.queue._items[0].environment_space,reference_trace['commands'][position]) or
                        not np.array_equal(np.asarray(obs['joint_action']['vector']),reference_trace['before'][position])):
                        np.savez_compressed(out/'rejected_factual_replan.npz',
                            generated_command=full.queue._items[0].environment_space,
                            reference_command=reference_trace['commands'][position],
                            generated_qpos=np.asarray(obs['joint_action']['vector']),reference_qpos=reference_trace['before'][position])
                        raise RuntimeError('instrumented factual replan differs from fixed source before branching; never teacher-force it')
                if args.factual_replay_only and ordinal in args.query_ordinals:
                    write_probe(out/'factual-witness-probes',f'q{ordinal:04d}',capture['probe'],
                        dict(identity,query_ordinal=ordinal,frame_index=len(factual_commands),counterfactual_branches_executed=False))
                    query_outcomes[ordinal]='factual_witness_only'
                    if args.seal_null_on_factual_replay:
                        from fastwam.research.native_replay import require_observation,observation_arrays
                        reference_generation=source_case/f'generations/replan-{ordinal:04d}'
                        generation=json.loads((reference_generation/'generation.json').read_text())
                        original_actions=np.load(reference_generation/'model_actions.npy',allow_pickle=False)
                        reference_replan=source_replans[len(factual_commands)]
                        if (reference_replan['bound_step_sha256']!=generation['bound_step_sha256'] or
                            reference_replan['model']['retrieval']['step_sha256']!=generation['bound_step_sha256'] or
                            reference_replan['model_action_chunk_sha256']!=sha256_array(original_actions)):
                            raise ValueError('original replan and generation seals disagree')
                        binding_dir=out/f'query-binding-q{ordinal:04d}';binding_dir.mkdir(exist_ok=False)
                        np.save(binding_dir/'reconstructed_full_H32.npy',capture['action'],allow_pickle=False)
                        step=capture['online_step']
                        (binding_dir/'observed_binding.json').write_bytes(canonical(dict(
                            actual_step_sha256=step.step_sha256,model_input_sha256=step.model_input_sha256,
                            candidate_payload_sha256=step.candidate_payload_sha256,derived_seed=int(step.derived_seed),
                            actual_latency=dict(step.telemetry),original_latency=reference_replan['model']['retrieval']['latency_s'],
                            original_telemetry_sha256=sha256_file(source_telemetry))))
                        binding_proof=require_original_generation(capture,generation,original_actions,
                            original_actions_sha256=sha256_file(reference_generation/'model_actions.npy'),
                            original_retrieval_latency=reference_replan['model']['retrieval']['latency_s'])
                        (binding_dir/'comparison_proof.json').write_bytes(canonical(binding_proof))
                        with np.load(source_case/f'replan-{ordinal:04d}.npz',allow_pickle=False) as data:
                            require_observation({k:data[k] for k in data.files},observation_arrays(obs))
                        alternative=generate_captured_null(full,null,capture)
                        assert_shared_weights_unchanged(shared_receipt)
                        qdir=out/f'null-seal-q{ordinal:04d}';qdir.mkdir(exist_ok=False)
                        native_actions={role:decode_branch_actions(full.action_normalizer,value) for role,value in
                            (('refined',capture['action']),('candidate_null',alternative['action']))}
                        np.savez_compressed(qdir/'sealed_native_actions.npz',**native_actions)
                        context_values={field.name:getattr(capture['context'],field.name) for field in fields(capture['context'])}
                        torch.save({name:value.detach().cpu() for name,value in context_values.items() if torch.is_tensor(value)},qdir/'sealed_factual_context.pt')
                        (qdir/'context_non_tensor_fields.json').write_bytes(canonical({name:value for name,value in context_values.items() if not torch.is_tensor(value)}))
                        write_probe(qdir/'probes','full',capture['probe'],dict(identity,query_ordinal=ordinal))
                        write_probe(qdir/'probes','candidate-null',alternative['probe'],dict(identity,query_ordinal=ordinal))
                        step=capture['online_step']
                        seal=dict(identity,query_ordinal=ordinal,frame_index=len(factual_commands),
                            original_generation_sha256=sha256_file(reference_generation/'generation.json'),
                            original_model_actions_sha256=sha256_file(reference_generation/'model_actions.npy'),
                            original_telemetry_sha256=sha256_file(source_telemetry),binding_comparison=binding_proof,
                            native_actions_sha256=sha256_file(qdir/'sealed_native_actions.npz'),
                            factual_context_sha256=sha256_file(qdir/'sealed_factual_context.pt'),
                            bound_step_sha256=step.step_sha256,model_input_sha256=step.model_input_sha256,
                            candidate_payload_sha256=step.candidate_payload_sha256,derived_seed=int(step.derived_seed),
                            exact_original_full_H32=True,exact_original_query_RGB_and_drive_target=True,
                            same_context_and_Gaussian=True,factual_rng_isolated=True,
                            full_policy_state_sha256=alternative['full_policy_state_sha256'],
                            before_any_generated_null_execution=True,counterfactual_branches_executed=0)
                        (qdir/'sealed_query.json').write_bytes(canonical(seal))
                        query_outcomes[ordinal]='factual_witness_null_actions_sealed'
                    print(json.dumps(dict(stage='exact_factual_replan_witness',query_ordinal=ordinal,
                        executed_commands=len(factual_commands))),flush=True)
                if ordinal in args.query_ordinals and not args.factual_replay_only:
                    query_outcomes[ordinal]='attempted'
                    query_id=f'{"formal" if formal else "pilot"}-{task}-seed{args.seed}-q{ordinal:04d}'
                    qdir=out/query_id;qdir.mkdir()
                    (qdir/'preparation.pickle').write_bytes(pickle.dumps(preparation,protocol=4))
                    (qdir/'preparation.json').write_bytes(canonical(proof))
                    history=full.controller.retrospective_episode_memory._memory.snapshot()
                    if ordinal>0 and history.counters.observation_updates<1:raise RuntimeError('live factual history was not populated')
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
                    if ordinal in pool_query_ordinals:
                        pool_commands,pool_native,pool_evidence=prepare_candidate_pool(probe,full.action_normalizer)
                        # This second index uses the actual query ID expected
                        # by the candidate assembler, also before any truth.
                        pool_probe=write_probe(qdir/'candidate-pool-probe',query_id,probe,
                            dict(query_identity,event_ids=query_identity['candidate_event_ids']))
                        np.savez_compressed(qdir/'sealed_pool_native_actions.npz',actions=pool_native)
                        pool_evidence.update(native_actions_sha256=sha256_file(qdir/'sealed_pool_native_actions.npz'),
                            probe_metadata_sha256=pool_probe['metadata_sha256'],before_any_branch_truth=True)
                        (qdir/'sealed_candidate_pool.json').write_bytes(canonical(pool_evidence))
                        pool_index=dict(task=task,episode_id=episode_id,query_id=query_id,
                            cohort='fixed_full_factual_candidate_pool' if formal else 'excluded_live_history_candidate_pool_qualification',probe_path=pool_probe['path'],
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
                    row=dict(schema='warm.recovery.query.v1',task=task,episode_id=episode_id,query_id=query_id,
                        protocol_sha256=identity['plan_sha256'],
                        checkpoint_sha256=full.checkpoint_sha256,cohort=cohort,horizon=32,
                        gate=sealed['gate'],gate_intervention='natural',branches=labels,
                        exact_refined_repeat=True,raw_evidence=str(qdir))
                    append_record(out/'recovery_queries.jsonl',row)
                    query_outcomes[ordinal]='complete'
                    collected+=1;print(json.dumps(dict(stage='branch_query_complete',query_id=query_id,labels=labels)),flush=True)
                ordinal+=1
            command=np.array(full.queue._items[0].environment_space,copy=True)
            before=np.asarray(obs['joint_action']['vector']).copy();position=len(factual_commands)
            if formal and (position>=len(reference_trace['commands']) or not np.array_equal(command,reference_trace['commands'][position])
                or not np.array_equal(before,reference_trace['before'][position])):
                raise RuntimeError('instrumented factual command or native state differs from fixed source; never teacher-force it')
            previous=env.take_action_cnt;full.step(env,obs)
            after=np.asarray(env.get_obs()['joint_action']['vector']).copy()
            factual_commands.append(command);factual_before.append(before);factual_after.append(after)
            if env.take_action_cnt!=previous+1:raise RuntimeError('official factual executor did not advance exactly one command')
            if formal and not np.array_equal(after,reference_trace['after'][position]):
                raise RuntimeError('instrumented factual native endpoint differs from fixed source')
        if not formal and collected!=len(args.query_ordinals):raise RuntimeError('planned pilot queries not all reached; no replacement allowed')
        if formal and (len(factual_commands)!=len(reference_trace['commands']) or int(bool(env.eval_success))!=admission['source_success']):
            raise RuntimeError('formal factual trajectory did not reproduce the fixed complete source')
        report.update(status='qualified_complete_factual_replay_witness_only' if args.factual_replay_only else 'qualified_actual_history_queries_only',queries=collected,
            independent_branch_labels=3*collected,all_planned_scientific_stages_qualified=False,
            exact_complete_source_trace=formal,scientific_collection_qualified=formal and not args.factual_replay_only)
    except BaseException as exc:
        report.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        with (out/'factual_trace.npz').open('xb') as stream:
            np.savez_compressed(stream,commands=np.asarray(factual_commands),before=np.asarray(factual_before),after=np.asarray(factual_after))
        for query in args.query_ordinals:
            append_record(out/'planned_query_status.jsonl',dict(episode_id=episode_id,query_ordinal=query,
                status=query_outcomes.get(query,'not_reached'),source_run_status=report['status'],replacement_allowed=False))
        report['factual_executed_commands']=len(factual_commands)
        report.update(queries=collected,independent_branch_labels=3*collected,
            factual_witness_queries=sum(value in ('factual_witness_only','factual_witness_null_actions_sealed') for value in query_outcomes.values()),
            factual_null_actions_sealed=sum(value=='factual_witness_null_actions_sealed' for value in query_outcomes.values()))
        for policy in policies.values():policy.close()
        if hasattr(env,'scene'):env.close_env(clear_cache=True)
        report.update(elapsed_seconds=time.monotonic()-started,core_source_sha256_after=source_identity(core)['sha256'])
        (out/'summary.json').write_bytes(canonical(report))
    if report['core_source_sha256_after']!=source['sha256']:raise RuntimeError('bound core changed')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
