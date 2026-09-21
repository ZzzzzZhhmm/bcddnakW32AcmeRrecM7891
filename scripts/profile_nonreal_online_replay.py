#!/usr/bin/env python3
"""Profile the deployed online policy on recorded DEV observations.

This is teacher-forced observation/action-history replay, not simulator rollout
and not a success-rate evaluation. Disk decoding is outside the timed region.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def summarize(rows, *, expected_count):
    import numpy as np
    measured = [row for row in rows if not row['warmup']]
    if len(measured) != expected_count:
        raise ValueError('incomplete measured query count')
    times = np.asarray([row['end_to_end_s'] for row in measured], dtype=float)
    if not np.isfinite(times).all() or np.any(times <= 0):
        raise ValueError('invalid query timings')
    keys = [(r['dataset_index'], r['recorded_episode'], r['recorded_frame']) for r in measured]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate measured prefixes')
    return dict(status='complete', queries=len(measured),
                episodes=len({key[:2] for key in keys}),
                warmup_queries=sum(row['warmup'] for row in rows),
                median_s=float(np.median(times)), p95_s=float(np.quantile(times, .95)),
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in measured),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in measured),
                scope='recorded DEV prefix online-pipeline timing; smoke checkpoint; no task SR')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-args', type=Path, required=True)
    parser.add_argument('--train-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--queries-per-episode', type=int, default=40)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--data-only', action='store_true', help='Validate all recorded inputs before loading a GPU model')
    args = parser.parse_args()
    if min(args.episodes, args.queries_per_episode, args.warmup) < 1:
        raise ValueError('all sample counts must be positive')
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('outputs must be outside source checkout')
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from scripts import build_warm_rmbench_contract_bundle as builder
    from scripts.run_warm_rmbench_matrix import load_matrix, select_experiments
    from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    from fastwam.memory.manifest import sha256_file
    from fastwam.research.evidence import append_record
    from experiments.rmbench.warm_policy.deploy_policy import RMBenchWarmPolicy

    started = time.perf_counter()
    bundle_args = builder.build_parser().parse_args(json.loads(args.bundle_args.read_text()))
    if len(bundle_args.task) != 1 or len(bundle_args.experiment) != 1:
        raise ValueError('one task and experiment per profiling process')
    task = bundle_args.task[0]
    matrix = load_matrix(bundle_args.matrix)
    experiment, = select_experiments(matrix, bundle_args.experiment)
    builder.validate_contract_bundle(bundle_args.output_root, experiment_id=experiment.id, task_names=[task])
    artifacts = builder._artifact_paths(bundle_args)
    runtime = builder._runtime_args(
        bundle_args, artifacts, experiment=experiment, task_name=task,
        contract_path=bundle_args.output_root/experiment.id/f'{task}.json',
        seed_protocol_path=bundle_args.output_root/'seeds'/f'{task}.seed_protocol.npy',
        task_definition_path=bundle_args.rmbench_root/'envs'/f'{task}.py', root_seed=matrix.root_seed)
    runtime['warm_telemetry_path'] = str(output/'policy_telemetry.jsonl')
    cfg = OmegaConf.load(args.train_config)
    if cfg.data.val.episode_split != 'dev' or cfg.data.val.is_training_set:
        raise ValueError('requires an explicit DEV split')
    print('stage=recorded_dev_dataset', flush=True)
    dataset = BaseLerobotDataset(
        dataset_dirs=list(cfg.data.val.dataset_dirs),
        shape_meta=OmegaConf.to_container(cfg.data.val.shape_meta, resolve=True),
        obs_size=1, action_size=bundle_args.replan_steps, val_set_proportion=0,
        is_training_set=False, episode_catalog_path=str(cfg.data.val.episode_catalog_path),
        episode_split='dev', episode_task_allowlist=[task], strict_sample_loading=True)
    dataset._set_return_images(True)
    episodes = []
    offset = 0
    needed = max(args.queries_per_episode, args.warmup)*bundle_args.replan_steps
    for di, child in enumerate(dataset.multi_dataset._datasets):
        for ep, start, stop in zip(child.episodes, child.episode_data_index['from'].tolist(),
                                   child.episode_data_index['to'].tolist(), strict=True):
            if stop-start < needed:
                raise ValueError(f'predeclared episode {ep} too short; do not silently replace it')
            episodes.append(dict(dataset_index=di, recorded_episode=int(ep), start=offset+int(start)))
        offset += len(child)
    episodes = sorted(episodes, key=lambda e: (e['dataset_index'],e['recorded_episode']))[:args.episodes]
    if len(episodes) != args.episodes:
        raise ValueError('insufficient DEV episodes')
    (output/'frozen_episodes.json').write_text(json.dumps(episodes,indent=2))
    (output/'runtime_args.json').write_text(json.dumps(runtime,indent=2))
    # Decode and validate the complete, predeclared cohort before GPU loading.
    # A replay never substitutes another episode when a requested input fails.
    from fastwam.memory.robotwin_artifacts import RobotwinQposZScore
    normalizer=RobotwinQposZScore.from_dataset_stats(json.loads(artifacts['normalization_stats'].read_text()))
    cached={}
    for number, episode in enumerate(episodes):
        count=max(args.queries_per_episode,args.warmup if number==0 else 0)
        for q in range(count):
            frame=q*bundle_args.replan_steps
            sample=dataset[episode['start']+frame]
            observed=(int(torch.as_tensor(sample['episode_index']).reshape(-1)[0]),
                      int(torch.as_tensor(sample['frame_index']).reshape(-1)[0]))
            if observed!=(episode['recorded_episode'],frame) or bool(sample['action_is_pad'].any()):
                raise ValueError('recorded sample identity or complete action-prefix check failed')
            for key in ('cam_high','cam_left_wrist','cam_right_wrist'):
                image=sample['images'][key]
                if image.dtype!=torch.uint8 or tuple(image.shape)!=(1,3,240,320):
                    raise ValueError('recorded camera shape/dtype disagrees with the declared profile')
            normalizer.normalize(sample['action']['default'].numpy(),fail_on_clip=True)
            cached[(episode['dataset_index'],episode['recorded_episode'],frame)]=sample
    write_data=dict(status='qualified',recorded_prefixes=len(cached),catalog_sha256=sha256_file(cfg.data.val.episode_catalog_path))
    (output/'data_qualification.json').write_text(json.dumps(write_data,indent=2))
    if args.data_only:
        print(json.dumps(write_data),flush=True)
        return
    print('stage=load_online_policy', flush=True)
    os.chdir(bundle_args.rmbench_root)
    policy = RMBenchWarmPolicy(runtime)
    bank = policy.retriever._bank
    manifest = dict(checkpoint_sha256=policy.checkpoint_sha256,
                    online_contract_sha256=policy.contract.sha256,
                    bank_content_sha256=policy.contract.bank_content_sha256,
                    train_config_sha256=sha256_file(args.train_config),
                    task=task, gpu=torch.cuda.get_device_name(), precision='bf16', batch=1,
                    nfe=policy.num_inference_steps, horizon=policy.action_horizon,
                    replan_steps=policy.replan_steps, bank_events=len(bank), top_k=bundle_args.top_k,
                    bank_storage='CPU numpy/mmap with exact CPU cosine search; gathered tensors transferred to model',
                    scope='recorded DEV observations and recorded executed actions; model outputs never executed; no simulator or SR',
                    timing='CUDA synchronized around entire deployed _replan, including preprocessing and telemetry; excludes disk decoding')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows = []
    schedule = [(True,episodes[0],args.warmup)] + [(False,ep,args.queries_per_episode) for ep in episodes]
    try:
        for warmup, episode, count in schedule:
            policy.reset()
            instruction = None
            for q in range(count):
                frame = q*policy.replan_steps
                sample = cached[(episode['dataset_index'],episode['recorded_episode'],frame)]
                actual_episode = int(torch.as_tensor(sample['episode_index']).reshape(-1)[0])
                actual_frame = int(torch.as_tensor(sample['frame_index']).reshape(-1)[0])
                if (actual_episode,actual_frame)!=(episode['recorded_episode'],frame):
                    raise ValueError('recorded prefix identity mismatch')
                if bool(sample['action_is_pad'].any()):
                    raise ValueError('recorded action prefix is padded')
                current_instruction = str(sample['task'])
                if instruction is None:
                    instruction = current_instruction
                if instruction != current_instruction:
                    raise ValueError('instruction changed in recorded episode')
                class RecordedInstruction:
                    def get_instruction(self):
                        return instruction
                cameras = {}
                for key, target in [('cam_high','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]:
                    cameras[target] = dict(rgb=np.ascontiguousarray(sample['images'][key][0].permute(1,2,0).numpy()))
                observation = dict(observation=cameras,
                    joint_action=dict(vector=np.ascontiguousarray(sample['state']['default'][0].numpy(),dtype=np.float32)))
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                policy._replan(observation, RecordedInstruction())
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-tick
                row = dict(dataset_index=episode['dataset_index'],recorded_episode=actual_episode,
                           recorded_frame=actual_frame,warmup=warmup,end_to_end_s=elapsed,
                           peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                           peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
                append_record(output/'timings.jsonl',row)
                rows.append(row)
                # Replay commands actually recorded in the demonstration. Never
                # claim the generated model commands caused the next observation.
                policy.queue.clear()
                for command in sample['action']['default'].numpy():
                    native=np.ascontiguousarray(command,dtype=np.float32)
                    normalized=policy.action_normalizer.normalize(native[None,:],fail_on_clip=True)[0]
                    policy.controller.note_executed_action(native,model_space_action=normalized)
                    policy._executed_policy_actions += 1
            print(json.dumps(dict(episode=episode['recorded_episode'],warmup=warmup,queries=count)),flush=True)
            policy.seal_active(reason='recorded_replay_end_no_outcome',success=False)
    finally:
        policy.close()
    report = summarize(rows,expected_count=args.episodes*args.queries_per_episode)
    report['elapsed_s']=time.perf_counter()-started
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (output/'RESULTS.md').write_text(
        '# Recorded DEV online-pipeline profile\n\n'+json.dumps(report,indent=2)+
        '\n\nNo task SR. Recorded action history, smoke300 checkpoint; excludes simulator and disk decoding.\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
