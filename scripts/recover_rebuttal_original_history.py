"""Reconstruct one original BT query history from recorded past-only facts.

The original controller and memory implementation are used without a policy,
retriever, simulator, GPU, checkpoint load, or historical online capability.
Every action and memory update must match the recorded evidence exactly.
"""
from pathlib import Path
import argparse,dataclasses,datetime,hashlib,importlib.util,json,os,sys,traceback


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence-root',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    cli=ap.parse_args();P=cli.evidence_root.resolve(strict=True);out=cli.output.resolve()
    if out.exists() or out.parent!=P:raise ValueError('fresh Git-external output under the evidence root required')
    C=P/'historical_specialist_source_f77c633/code'
    S=P/'original_BT12000_cohort_case0_native1000_v1'
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('original_identity',P/'controller_eb70b41/code/scripts/run_rebuttal_training_with_runtime_trace.py')
    identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
    expected='ff32342d286929ddc5904d55814c66e965269f6b749c1e1b6df1060aaef0b7fe'
    assert identity.source_digest(C)==expected
    assert read(S/'operator.json')['status']=='complete_fixed_original_factual_episode'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path[:0]=[str(C/'src'),str(C)]
    import numpy as np
    import torch
    from fastwam.memory.online_episode_controller import OnlineEpisodeController
    from fastwam.memory.online_episode_memory import OnlineRetrospectiveEpisodeMemory
    from fastwam.memory.robotwin_artifacts import RobotwinQposZScore,ROBOTWIN_GRIPPER_DIMS
    from fastwam.memory.manifest import sha256_array
    from fastwam.models.warm.online_contract import WarmOnlineRunContract
    args=read(S/'runtime_args.json')
    assert args['task_name']=='battery_try' and args['seed']==3407
    assert args['warm_recent_event_capacity']==20 and args['warm_action_summary_capacity']==8
    contract=WarmOnlineRunContract.from_dict(read(Path(args['warm_online_contract_path'])))
    assert contract.warm_checkpoint_sha256=='27c137d689007b38f6204d3dd12bb038550b1e58403506d7a9894bd6fcd43a9f'
    assert sha(Path(args['dataset_stats_path']))==contract.normalization_stats_sha256
    normalize=RobotwinQposZScore.from_dataset_stats(read(Path(args['dataset_stats_path'])))
    lines=[json.loads(s) for s in (S/'full-telemetry.jsonl').read_text().splitlines()]
    replans={int(x['frame_index']):x for x in lines if x['kind']=='replan'}
    executed={int(x['frame_index']):x for x in lines if x['kind']=='executed_action'}
    physical={int(x['index']):x for x in (json.loads(s) for s in (S/'native_trace.jsonl').read_text().splitlines())}
    memory=OnlineRetrospectiveEpisodeMemory(action_dim=14,action_horizon=32,semantic_dim=768,
        gripper_indices=ROBOTWIN_GRIPPER_DIMS,recent_event_capacity=20,action_summary_capacity=8,episode_namespace='rmbench-eval')
    controller=OnlineEpisodeController(contract=contract,retriever=None,retrospective_episode_memory=memory)
    controller.begin_episode(0)
    out.mkdir();state=dict(status='reconstructing_original_history',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        query=8,frame=32,source_sha256=expected,model_inferences=0,physical_commands_executed=0,checkpoint_loaded=False,
        retriever_constructed=False,historical_BoundOnlineStep_constructed=False,reconstructed_past_observations=0,verified_past_actions=0)
    def save():
        state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (out/'operator.json').write_text(json.dumps(state,indent=2))
    save();inputs={}
    def authenticate(p):
        digest=sha(p);inputs[str(p)]=dict(bytes=p.stat().st_size,sha256=digest);return p
    for p in [S/'runtime_args.json',S/'full-telemetry.jsonl',S/'native_trace.jsonl',Path(args['dataset_stats_path']),Path(args['warm_online_contract_path'])]:authenticate(p)
    audits=[]
    try:
        for query in range(9):
            frame=query*4;record=replans[frame]
            qid=controller.issue_query_id(frame)
            assert dataclasses.asdict(qid)==record['query_id']
            history,evidence=controller.retrospective_history_kwargs()
            assert evidence==record['history_before_replan'],(query,evidence,record['history_before_replan'])
            audits.append(dict(query=query,query_id_exact=True,history_evidence_exact=True,history=evidence))
            if query==8:
                assert set(history)=={'episode_tokens','episode_mask','episode_action_summaries','episode_action_mask'}
                np.savez_compressed(out/'recovered_history.npz',**history)
                state.update(status='complete_original_q8_history_reconstruction_CPU_only',query_id=dataclasses.asdict(qid),
                    original_history_evidence=evidence,history_arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=sha256_array(v)) for k,v in history.items()},
                    exact_history_evidence_matches=9,all32_past_action_hashes_exact=True,
                    full_factual_context_recovered=False,real_checkpoint_GPU_replay_qualified=False)
                break
            path=authenticate(S/'probes'/f'q{query:04d}'/'model_output.pt')
            output=torch.load(path,map_location='cpu',weights_only=True)
            committed=controller.commit_factual_replan_observation(frame_index=frame,model_output=output)
            assert committed==record['factual_update_after_replan'],(query,committed,record['factual_update_after_replan'])
            state['reconstructed_past_observations']+=1
            for index in range(frame,frame+4):
                command=np.asarray(physical[index]['command'],dtype=np.float32)
                model=normalize.normalize(command,fail_on_clip=True)
                assert sha256_array(command)==executed[index]['environment_action_sha256'],index
                assert sha256_array(model)==executed[index]['model_action_sha256'],index
                controller.note_executed_action(command,model_space_action=model)
                state['verified_past_actions']+=1
        assert state['verified_past_actions']==32 and not torch.cuda.is_initialized()
        assert identity.source_digest(C)==expected
        assert all(sha(Path(n))==v['sha256'] for n,v in inputs.items())
        (out/'inputs_manifest.json').write_text(json.dumps(inputs,indent=2))
        (out/'history_audit.json').write_text(json.dumps(audits,indent=2))
        state.update(recovered_history_sha256=sha(out/'recovered_history.npz'),original_source_unchanged=True,
            original_inputs_unchanged=True,CUDA_initialized=False,ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save();print(json.dumps(state))
    except BaseException as error:
        state.update(status='failed_history_reconstruction',error=repr(error),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
