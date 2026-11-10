#!/usr/bin/env python3
"""Qualify the typed all-K control on fixed recorded DEV prefixes.

Real policy inference, causal recorded history and independently trained weights;
no simulator, generated-command execution, TEST selection or efficacy estimate.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,sys,traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--expected-plan-sha256',required=True)
    parser.add_argument('--data-only',action='store_true')
    args=parser.parse_args()
    def sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as stream:
            for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
        return h.hexdigest()
    read=lambda p:json.loads(Path(p).read_text())
    if sha(args.plan)!=args.expected_plan_sha256:raise ValueError('Frozen DEV qualification plan differs')
    plan=read(args.plan)
    if (plan['schema']!='warm.rebuttal.pooled-native-dev-qualification.v1' or plan['task']!='press_button'
        or plan['query_frames']!=list(range(0,40,4)) or plan['checkpoint_update']!=100
        or plan['checkpoint_sha256']!='cbdfcef2432f2e2b99ddfc22a95dc16450df43979c094c85e3f7787faecfe5ec'):
        raise ValueError('Fixed pool100/earliest PB DEV prefix cohort required')
    out=Path(plan['data_output'] if args.data_only else plan['output'])
    if not out.is_absolute() or out.resolve().is_relative_to(ROOT):raise ValueError('Evidence must stay outside code')
    out.mkdir(parents=True,exist_ok=False)
    from fastwam.research.fresh_pool import atomic_json
    from scripts.nonreal_job import source_identity
    state=dict(status='binding',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        plan_sha256=args.expected_plan_sha256,checkpoint_update=100,checkpoint_sha256=plan['checkpoint_sha256'],
        generated_commands_executed=0,task_success_measured=False,queries_complete=0,engineering_only=True)
    def save():atomic_json(out/'operator.json',state)
    def budget():
        if datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat(plan['absolute_deadline_utc']):
            raise TimeoutError('Original qualification resource window expired;do not reset it')
    save();policy=None;hook=None;output_hook=None;original_conditioning=None;had_conditioning=False
    try:
        budget()
        if source_identity(ROOT)['sha256']!=plan['evaluation_source_sha256']:raise ValueError('Evaluation source differs')
        for item in plan['inputs']:
            if sha(item['path'])!=item['sha256']:raise ValueError('Frozen input differs: '+item['path'])
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        from scripts import build_warm_rmbench_contract_bundle as builder
        from scripts.run_warm_rmbench_matrix import load_matrix,select_experiments
        from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
        from fastwam.memory.manifest import sha256_array
        from fastwam.research.evidence import append_record,write_probe
        from fastwam.research.query_capture import capture_online_replan
        from fastwam.research.pooled_memory_prior import PooledMemoryPriorFastWAM,POOLED_SCHEMA
        from experiments.rmbench.warm_policy.deploy_policy import RMBenchWarmPolicy,_normalize_state
        ba=builder.build_parser().parse_args(read(plan['bundle_args']))
        if ba.task!=['press_button'] or ba.experiment!=['full_warm'] or ba.replan_steps!=4:
            raise ValueError('Fixed native input task/prefix differs')
        builder.validate_contract_bundle(ba.output_root,experiment_id='full_warm',task_names=['press_button'])
        experiment,=select_experiments(load_matrix(ba.matrix),ba.experiment)
        artifacts=builder._artifact_paths(ba)
        runtime=builder._runtime_args(ba,artifacts,experiment=experiment,task_name='press_button',
            contract_path=ba.output_root/'full_warm/press_button.json',
            seed_protocol_path=ba.output_root/'seeds/press_button.seed_protocol.npy',
            task_definition_path=ba.rmbench_root/'envs/press_button.py',root_seed=3407)
        runtime['warm_telemetry_path']=str(out/'policy_telemetry.jsonl')
        (out/'runtime_args.json').write_text(json.dumps(runtime,indent=2)+'\n')
        cfg=OmegaConf.load(plan['train_config'])
        if cfg.data.val.episode_split!='dev' or cfg.data.val.is_training_set:raise ValueError('Explicit DEV data required')
        state['status']='loading_fixed_DEV_inputs';save()
        dataset=BaseLerobotDataset(dataset_dirs=list(cfg.data.val.dataset_dirs),
            shape_meta=OmegaConf.to_container(cfg.data.val.shape_meta,resolve=True),obs_size=5,action_size=4,
            val_set_proportion=0,is_training_set=False,episode_catalog_path=str(cfg.data.val.episode_catalog_path),
            episode_split='dev',episode_task_allowlist=['press_button'],strict_sample_loading=True)
        dataset._set_return_images(True)
        episodes=[];offset=0
        for di,child in enumerate(dataset.multi_dataset._datasets):
            for ep,start,stop in zip(child.episodes,child.episode_data_index['from'].tolist(),child.episode_data_index['to'].tolist(),strict=True):
                episodes.append(dict(dataset_index=di,recorded_episode=int(ep),start=offset+int(start),length=int(stop-start)))
            offset+=len(child)
        episode=min(episodes,key=lambda e:(e['dataset_index'],e['recorded_episode']))
        if any(episode[k]!=v for k,v in plan['episode'].items()) or episode['length']<40:
            raise ValueError('Original earliest DEV episode differs;no replacement allowed')
        cached=[];instruction=None
        for frame in plan['query_frames']:
            sample=dataset[episode['start']+frame]
            if (int(torch.as_tensor(sample['episode_index']).reshape(-1)[0])!=episode['recorded_episode']
                or int(torch.as_tensor(sample['frame_index']).reshape(-1)[0])!=frame
                or bool(sample['action_is_pad'].any()) or bool(sample['state_is_pad'][0]) or bool(sample['image_is_pad'][0])):
                raise ValueError('DEV current observation/action prefix is incomplete')
            cameras={}
            for key,target in [('cam_high','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]:
                value=sample['images'][key]
                if value.dtype!=torch.uint8 or tuple(value.shape)!=(5,3,240,320):raise ValueError('DEV raw camera shape differs')
                cameras[target]=dict(rgb=np.ascontiguousarray(value[0].permute(1,2,0).numpy()))
            raw=np.ascontiguousarray(sample['state']['default'][0].numpy(),dtype=np.float32)
            commands=np.ascontiguousarray(sample['action']['default'].numpy(),dtype=np.float32)
            if raw.shape!=(14,) or commands.shape!=(4,14) or not np.isfinite(raw).all() or not np.isfinite(commands).all():
                raise ValueError('Finite native current state and recorded prefix required')
            text=sample['text']
            if not isinstance(text,str) or not text:raise ValueError('Current DEV instruction unavailable')
            if instruction is None:instruction=text
            if instruction!=text:raise ValueError('DEV instruction changed')
            obs=dict(observation=cameras,joint_action=dict(vector=raw))
            np.savez_compressed(out/f'input-frame{frame:04d}.npz',raw_state=raw,recorded_commands=commands,
                **{k:v['rgb'] for k,v in cameras.items()})
            cached.append((frame,obs,commands))
        atomic_json(out/'data_qualification.json',dict(status='all10_fixed_DEV_prefixes_loaded',episode=episode,
            query_frames=plan['query_frames'],future_observations_discarded_before_policy=True,
            generated_actions_not_replayed_as_truth=True,simulator_loaded=False))
        if args.data_only:
            state.update(status='complete_DEV_data_only',policy_loaded=False);save();print(json.dumps(state));return
        budget();state['status']='loading_typed_pool100_policy';save()
        os.chdir(ba.rmbench_root)
        policy=RMBenchWarmPolicy(runtime)
        if not isinstance(policy.model,PooledMemoryPriorFastWAM):raise ValueError('Incorrect model constructor')
        if policy.checkpoint_sha256!=plan['checkpoint_sha256'] or policy.model._retrospection_state()['schema']!=POOLED_SCHEMA:
            raise ValueError('Typed trained pool100 checkpoint differs')
        policy.model.configure_research_probe(capture=True,force_null=False)
        observed=[];conditioning=[]
        def tensor_info(v):
            return dict(shape=list(v.shape),dtype=str(v.dtype),sha256=sha256_array(v.detach().cpu().contiguous().view(torch.uint8).numpy()),
                finite=bool(torch.isfinite(v).all()),nonzero=int(torch.count_nonzero(v)))
        def inputs_observer(module,unused,kwargs):
            forbidden={'target_action','target_effect','current_semantic_teacher','future_valid_mask'}
            if forbidden.intersection(kwargs):raise ValueError('Future teacher entered pooled event adapter')
            fields=('warped_actions','candidate_valid_mask','event_pre_tokens','event_delta_tokens',
                'event_delta_mask','candidate_context','candidate_phase_features','gripper_timing','gist_tokens','query_context','proprio')
            observed.append({k:tensor_info(kwargs[k]) for k in fields})
        def output_observer(module,unused,value):
            observed[-1]['pooled_output']=tensor_info(value.action_context_tokens)
        hook=policy.model.retrospective_event_adapter.register_forward_pre_hook(inputs_observer,with_kwargs=True)
        output_hook=policy.model.retrospective_event_adapter.register_forward_hook(output_observer)
        original_conditioning=policy.model._source_only_conditioning
        had_conditioning='_source_only_conditioning' in vars(policy.model)
        def observe_conditioning(*a,**kw):
            value=original_conditioning(*a,**kw);conditioning.append(tensor_info(value));return value
        policy.model._source_only_conditioning=observe_conditioning
        policy.reset();rows=[]
        class Instruction:
            def get_instruction(self):return instruction
        for frame,obs,commands in cached:
            budget();observed.clear();conditioning.clear();state.update(status='native_policy_replan',active_DEV_frame=frame);save()
            with capture_online_replan(policy) as captured:policy._replan(obs,Instruction())
            actions=np.asarray(captured['action']);step=captured['online_step'];context=captured['context']
            if actions.shape!=(32,14) or not np.isfinite(actions).all():raise ValueError('Native H32 output is invalid')
            if not observed or any(v['gist_tokens']['shape'][:2]!=[1,8] or v['candidate_valid_mask']['shape']!=[1,32] for v in observed):
                raise ValueError('Actual history/all-K adapter route differs')
            if any(not f['finite'] for v in observed for f in v.values()):raise ValueError('Nonfinite actual conditioning input')
            if any(v['pooled_output']['shape'][:3]!=[1,32,4] for v in observed):raise ValueError('Actual all-K output token shape differs')
            if not conditioning or any(v['shape'][:2]!=[1,12] or not v['finite'] for v in conditioning):
                raise ValueError('Actual sampler 8-history plus4-pooled condition route differs')
            expected=_normalize_state(obs['joint_action']['vector'],policy.processor).to(dtype=policy.model.torch_dtype,device=policy.model.device)
            if tensor_info(expected)['sha256']!=observed[0]['proprio']['sha256']:raise ValueError('Actual current proprio binding differs')
            probe=write_probe(out/'probes',f'frame{frame:04d}',captured['probe'],dict(recorded_DEV_frame=frame,checkpoint_update=100,scope='engineering_only'))
            np.save(out/f'actions-frame{frame:04d}.npy',actions,allow_pickle=False)
            row=dict(recorded_frame=frame,derived_seed=int(step.derived_seed),output_shape=list(actions.shape),output_sha256=sha256_array(actions),
                model_input_sha256=step.model_input_sha256,proprio_sha256=step.proprio_sha256,candidate_payload_sha256=step.candidate_payload_sha256,
                context_key_sha256=step.context_key_sha256,bank_rows=step.bank_rows.tolist(),valid_candidates=int(step.candidate_valid_mask.sum()),
                episode_event_history_valid=int(context.episode_mask.sum()),episode_action_history_valid=int(context.episode_action_mask.sum()),
                actual_adapter_calls=observed.copy(),actual_sampler_conditioning=conditioning.copy(),probe=probe,generated_commands_executed=0)
            append_record(out/'queries.jsonl',row);rows.append(row)
            policy.queue.clear()
            for command in commands:
                normalized=policy.action_normalizer.normalize(command[None,:],fail_on_clip=True)[0]
                policy.controller.note_executed_action(command,model_space_action=normalized);policy._executed_policy_actions+=1
            state['queries_complete']=len(rows);save()
        if not any(r['episode_action_history_valid'] for r in rows[1:]):raise ValueError('No real causal action history observed')
        if not any(r['episode_event_history_valid'] for r in rows[-2:]):raise ValueError('No real causal event history observed by frame36')
        policy.seal_active(reason='recorded_DEV_control_qualification_no_outcome',success=False)
        if source_identity(ROOT)['sha256']!=plan['evaluation_source_sha256']:raise ValueError('Immutable evaluation source changed')
        state.update(status='complete_typed_pool100_native_DEV_qualification',queries_complete=10,actual_model_class=type(policy.model).__name__,
            original_training_source_sha256=plan['training_source_sha256'],evaluation_source_sha256=plan['evaluation_source_sha256'],
            factual_history_not_zero_placeholder=True,all_K_records_received=True,H32_finite_outputs=True,
            scope='DEV recorded causal-history replay;no simulator/generated-command execution/final3000 TEST/efficacy or timing benchmark')
        save();print(json.dumps(state),flush=True)
    except BaseException as exc:
        state.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc());save();raise
    finally:
        if hook is not None:hook.remove()
        if output_hook is not None:output_hook.remove()
        if original_conditioning is not None:
            if had_conditioning:policy.model._source_only_conditioning=original_conditioning
            else:del policy.model._source_only_conditioning
        if policy is not None:policy.close()
        state['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()


if __name__=='__main__':main()
