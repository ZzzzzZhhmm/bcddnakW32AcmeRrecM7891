#!/usr/bin/env python3
"""Measure natural gates on fixed saved query0 inputs with nonempty banks.

Each condition runs in a fresh process. No simulator is loaded and no generated
command is executed. The original200-pair missing-skill protocol is unchanged.
"""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,traceback

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--expected-plan-sha256',required=True)
    parser.add_argument('--condition',choices=('clean_size_matched','target_skill_family_holdout'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-only',action='store_true')
    args=parser.parse_args()
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
        return h.hexdigest()
    if sha(args.plan)!=args.expected_plan_sha256:raise ValueError('diagnostic plan changed')
    plan=json.loads(args.plan.read_text())
    if (plan['schema']!='warm.rebuttal.initial-gate-availability.v1' or plan['query_ordinal']!=0
        or [r['episode_id'] for r in plan['sources']]!=[0,1,2]
        or plan['task']!='rearrange_blocks'
        or plan['conditions']!=['clean_size_matched','target_skill_family_holdout']):
        raise ValueError('original3 fixed initial states and both frozen conditions required')
    core=Path(plan['core_code']).resolve(strict=True);out=args.output.resolve()
    if not args.output.is_absolute() or out.is_relative_to(ROOT) or out.is_relative_to(core):
        raise ValueError('absolute evidence output outside both source roots required')
    out.mkdir(parents=True,exist_ok=False)
    (out/'frozen_plan.json').write_bytes(args.plan.read_bytes())
    sys.path[:0]=[str(core/'src'),str(core)]
    import fastwam,fastwam.research
    if Path(fastwam.__file__).resolve()!=core/'src/fastwam/__init__.py':raise ValueError('wrong core package')
    fastwam.research.__path__.insert(0,str(ROOT/'src/fastwam/research'))
    import numpy as np
    from fastwam.memory.manifest import sha256_array
    from fastwam.research.initial_gate import initial_observation,require_empty_initial_history,gate_measurement
    from fastwam.research.fresh_pool import atomic_json
    from scripts.nonreal_job import source_identity
    collector=source_identity(ROOT)['sha256']
    if collector!=plan['collector_source_sha256'] or source_identity(core)['sha256']!=plan['core_source_sha256']:
        raise ValueError('collector/scientific core source differs')
    for item in plan['inputs']:
        if sha(item['path'])!=item['sha256']:raise ValueError('diagnostic input changed: '+item['path'])
    bank_report=json.loads(Path(plan['universe_summary']).read_text())
    universe,=[r for r in bank_report['reports'] if r['target_task']==plan['task']]
    eligible=np.load(universe['row_files'][args.condition]['path'],allow_pickle=False)
    task_indices=np.load(plan['task_indices'],allow_pickle=False)
    if len(eligible)!=19540 or len(np.unique(eligible))!=len(eligible):raise ValueError('frozen RB search universe differs')
    observations=[]
    for source in plan['sources']:
        identity=json.loads(Path(source['initial_identity']).read_text())
        obs=initial_observation(source['observation'],identity['observation_arrays'])
        rows=[json.loads(line) for line in Path(source['telemetry']).read_text().splitlines() if line.strip()]
        original,=[r for r in rows if r.get('kind')=='replan' and r['episode_index']==source['episode_id'] and r['frame_index']==0]
        observations.append((source,identity,obs,original))
    runtime=json.loads(Path(plan['runtime_args']).read_text())
    runtime.update(warm_telemetry_path=str(out/'policy_telemetry.jsonl'),
        warm_research_probe_config=None,warm_research_probe_output=None)
    (out/'runtime_args.json').write_text(json.dumps(runtime,indent=2)+'\n')
    atomic_json(out/'data_qualification.json',dict(status='qualified_recorded_initial_inputs_only',
        source_episodes=3,query_ordinal=0,eligible_bank_rows=len(eligible),
        row_array_sha256=sha256_array(eligible),policy_loaded=False,executed_commands=0))
    if args.data_only:
        print(json.dumps(dict(status='qualified_recorded_initial_inputs_only',policy_loaded=False,executed_commands=0)),flush=True)
        return
    from fastwam.research.encoder_provenance import install
    install(plan['encoder_qualification_plan'],plan['encoder_qualification_plan_sha256'],
        out/'encoder_qualification.json',memory_fraction=.45)
    import torch
    from experiments.rmbench.warm_policy import deploy_policy
    if Path(deploy_policy.__file__).resolve()!=core/'experiments/rmbench/warm_policy/deploy_policy.py':raise ValueError('wrong original policy')
    from fastwam.research.repertoire import install_search_restriction
    from fastwam.research.query_capture import capture_online_replan
    from fastwam.research.evidence import append_record,write_probe
    os.chdir(plan['runtime'])
    policy=None;rows=[];started=time.monotonic()
    summary=dict(status='running',condition=args.condition,expected_queries=3,completed_queries=0,
        executed_commands=0,task_success_measured=False,independent_relevance_labels_measured=False,
        collector_source_sha256=collector,core_source_sha256=plan['core_source_sha256'])
    atomic_json(out/'summary.json',summary)
    try:
        print('stage=load_policy',flush=True)
        policy=deploy_policy.RMBenchWarmPolicy(runtime)
        if policy.checkpoint_sha256!=plan['checkpoint_sha256']:raise ValueError('author-selected weights differ')
        policy.model.configure_research_probe(capture=True,force_null=False)
        if sha256_array(policy.retriever._task_indices)!=bank_report['bank_task_indices_array_sha256']:
            raise ValueError('actual bank task provenance differs')
        restriction=install_search_restriction(policy.retriever,eligible)
        atomic_json(out/'restriction.json',restriction)
        for source,identity,obs,original in observations:
            policy.reset()
            if policy.controller.active_episode_index!=source['episode_id']:raise ValueError('native episode cursor differs')
            require_empty_initial_history(*policy.controller.retrospective_history_kwargs())
            class InitialInstruction:
                def get_instruction(self):return identity['instruction']
            with capture_online_replan(policy) as captured:
                policy._replan(obs,InitialInstruction())
            step=captured['online_step'];probe=captured['probe']
            for key in ('model_input_sha256','proprio_sha256','context_key_sha256','factual_world_tokens_sha256'):
                if getattr(step,key)!=original[key]:raise ValueError('original current input changed: '+key)
            if dict(step.raw_camera_sha256)!=original['raw_camera_sha256']:raise ValueError('original camera bytes changed')
            measured=gate_measurement(probe,step.bank_rows,eligible,task_indices,bank_report['vocabulary'])
            if not np.array_equal(step.candidate_valid_mask,np.asarray(probe['arrays']['valid'])[0]):
                raise ValueError('bound and probed candidate validity differ')
            episode=source['episode_id'];qdir=f'e{episode:04d}-q0000'
            reference=write_probe(out/'probes',qdir,probe,dict(condition=args.condition,
                source_episode=episode,frame_index=0,derived_seed=int(step.derived_seed),
                checkpoint_sha256=policy.checkpoint_sha256,model_input_sha256=step.model_input_sha256,
                candidate_payload_sha256=step.candidate_payload_sha256,
                event_ids=[deploy_policy._event_dict(e) for e in step.event_ids],history='native_empty_at_query0'))
            np.save(out/f'{qdir}-model_actions.npy',captured['action'],allow_pickle=False)
            np.save(out/f'{qdir}-bank_rows.npy',step.bank_rows,allow_pickle=False)
            row=dict(measured,source_episode=episode,source_seed=source['seed'],condition=args.condition,
                frame_index=0,derived_seed=int(step.derived_seed),probe=reference,
                model_input_sha256=step.model_input_sha256,factual_world_tokens_sha256=step.factual_world_tokens_sha256,
                proprio_sha256=step.proprio_sha256,context_key_sha256=step.context_key_sha256,
                base_gaussian_sha256=sha256_array(probe['arrays']['base_gaussian'].float().numpy()),
                bound_step_sha256=step.step_sha256,native_empty_history=True,
                executed_commands=0,cleanup_success_field_is_not_task_outcome=True)
            append_record(out/'queries.jsonl',row);rows.append(row)
            policy.queue.clear();policy.seal_active(reason='initial_gate_diagnostic_no_outcome',success=False)
            summary.update(completed_queries=len(rows));atomic_json(out/'summary.json',summary)
            print(json.dumps(dict(source=episode,condition=args.condition,g=row['g'],alpha=row['alpha'],valid=row['valid_candidates'])),flush=True)
        summary.update(status='complete_recorded_initial_gate_diagnostic',queries=rows,
            checkpoint_sha256=policy.checkpoint_sha256,scope='3 fixed recorded initial states; availability response only. No physical execution, relevant-coverage labels, false-activation rate or task SR.')
    except BaseException as exc:
        summary.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if policy is not None:policy.close()
        summary.update(elapsed_seconds=time.monotonic()-started,collector_source_sha256_after=source_identity(ROOT)['sha256'])
        atomic_json(out/'summary.json',summary)
    if summary['collector_source_sha256_after']!=collector:raise ValueError('collector source changed')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
