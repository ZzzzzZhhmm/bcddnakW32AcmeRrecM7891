"""Check native replay of a fixed complete Full source without physical unpack.

Runs two fresh scenes through every recorded command. Exact native before/after
qpos and saved query RGB must match the source; all new query-state components
must repeat across the two scenes. This is engineering evidence only: no new
policy SR, model-effect validity or counterfactual benefit is measured.
"""
from __future__ import annotations
import argparse,importlib,json,os,pickle,random,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('core-code','rmbench-runtime','source','plan','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--episode-index',type=int,required=True)
    a=p.parse_args();out=a.output.resolve();core=a.core_code.resolve(strict=True);runtime=a.rmbench_runtime.resolve(strict=True)
    if out.is_relative_to(ROOT) or out.is_relative_to(core):raise ValueError('evidence must stay outside source')
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    from scripts.nonreal_job import source_identity
    from scripts.qualify_rebuttal_rmbench_task import resolved_config,source_identity as runtime_identity
    from fastwam.research.native_replay import observation_arrays,require_observation,native_parent,state_hashes,require_trace_step
    from fastwam.research.resumable_pairs import file_sha,read_json,read_roles
    from fastwam.research.evidence import append_record
    identity=read_json(a.source/'identity.json');plan=read_json(a.plan)
    trial_path=a.source/'role_trials.jsonl'
    if not trial_path.exists():trial_path=a.source/'pairs.jsonl'
    rows=[json.loads(s) for s in trial_path.read_text().splitlines() if s.strip()]
    row,=[r for r in rows if r['episode_id']==a.episode_index]
    task=identity['task'];seed=row['seed'];native_identity=runtime_identity(runtime);bound_source=source_identity(core)
    original_queries=plan['local_recovery']['query_replan_ordinals']
    binding=dict(task=task,checkpoint_sha256=plan['checkpoint_sha256'],evidence_plan_sha256=file_sha(a.plan),
        core_source_sha256=bound_source['sha256'],runtime_source_sha256=native_identity['sha256'],official_step_limit={'rearrange_blocks':700,'battery_try':1000}[task])
    source_records=read_roles(a.source,binding,{r['episode_id']:r['seed'] for r in rows})
    full,=[r for r in source_records if r['role']=='full' and r['row']['episode_id']==a.episode_index]
    case=Path(full['worker_evidence']);expert=read_json(case.parent/'expert_info.json')
    with np.load(case/'qpos_trace.npz',allow_pickle=False) as loaded:trace={k:loaded[k].copy() for k in loaded.files}
    query_frames={q:q*plan['inference']['execution_prefix'] for q in original_queries if q*plan['inference']['execution_prefix']<len(trace['commands'])}
    report=dict(status='running',scope='Engineering native command replay only; no policy inference, new independent SR, or mechanism benefit.',
        source=str(a.source),source_trace_sha256=file_sha(case/'qpos_trace.npz'),source_initial_sha256=file_sha(case/'initial_identity.json'),
        task=task,seed=seed,episode_id=a.episode_index,planned_query_ordinals=original_queries,
        full_source_success=full['row']['roles']['full']['success'],core_source_sha256=bound_source['sha256'],
        runtime_source_sha256=native_identity['sha256'],collector_source_sha256=source_identity(ROOT)['sha256'],
        checkpoint_sha256=plan['checkpoint_sha256'],plan_sha256=file_sha(a.plan),replays_completed=0,
        counterfactual_branches_executed=0,full_scientific_qualification=False)
    (out/'identity.json').write_text(json.dumps(report,indent=2));started=time.monotonic();env=None;reference={}
    os.chdir(runtime);sys.path[:0]=[str(runtime),str(runtime/'description/utils')]
    from generate_episode_instructions import generate_episode_descriptions
    cls=getattr(importlib.import_module('envs.'+task),task)
    config=resolved_config(runtime,task,seed);config['now_ep_num']=a.episode_index
    try:
        for repeat in range(2):
            env=cls();random.seed(seed);env.setup_demo(**config)
            instruction=str(np.random.choice(generate_episode_descriptions(task,[expert['info']],100)[0]['unseen']))
            env.set_instruction(instruction=instruction)
            if instruction!=full['initial_identity']['instruction']:raise ValueError('source instruction differs')
            with np.load(case/'initial_observation.npz',allow_pickle=False) as data:
                require_observation({k:data[k] for k in data.files},observation_arrays(env.get_obs()))
            for index,command in enumerate(trace['commands']):
                obs=env.get_obs();before=np.array(obs['joint_action']['vector'],copy=True)
                for query,frame in query_frames.items():
                    if index!=frame:continue
                    with np.load(case/f'replan-{query:04d}.npz',allow_pickle=False) as data:
                        require_observation({k:data[k] for k in data.files},observation_arrays(obs))
                    state=native_parent(env,obs);hashes=state_hashes(state)
                    target=out/f'replay-{repeat}-query-{query:04d}.pickle';target.write_bytes(pickle.dumps(state,protocol=4))
                    append_record(out/'query_checks.jsonl',dict(repeat=repeat,query_ordinal=query,frame_index=index,
                        state_hashes=hashes,state_path=str(target),state_sha256=file_sha(target),exact_source_rgb_and_qpos=True))
                    if repeat==0:reference[query]=hashes
                    elif reference[query]!=hashes:
                        differences=[k for k in hashes if hashes[k]!=reference[query][k]]
                        raise ValueError('fresh-scene native parent components differ: '+','.join(differences))
                    print(json.dumps(dict(stage='native_query_replay',repeat=repeat,query_ordinal=query,frame_index=index)),flush=True)
                previous=env.take_action_cnt;env.take_action(command.copy(),action_type='qpos')
                after=np.array(env.get_obs()['joint_action']['vector'],copy=True)
                if env.take_action_cnt!=previous+1:raise ValueError('official command count did not advance')
                require_trace_step(trace,index,before,after)
            if int(bool(env.eval_success))!=report['full_source_success']:raise ValueError('source terminal outcome differs')
            report['replays_completed']+=1
            env.close_env(clear_cache=True);env=None
        report.update(status='qualified_native_command_replay_only',exact_source_commands=len(trace['commands']),
            exact_query_parents=len(reference),not_reached_queries=[q for q in original_queries if q not in query_frames])
    except BaseException as exc:
        report.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if env is not None and hasattr(env,'scene'):env.close_env(clear_cache=True)
        report.update(elapsed_seconds=time.monotonic()-started,runtime_source_sha256_after=runtime_identity(runtime)['sha256'],core_source_sha256_after=source_identity(core)['sha256'])
        (out/'summary.json').write_text(json.dumps(report,indent=2))
    if report['runtime_source_sha256_after']!=native_identity['sha256'] or report['core_source_sha256_after']!=bound_source['sha256']:raise ValueError('bound source changed')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
