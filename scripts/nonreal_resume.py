#!/usr/bin/env python3
"""Prepare and execute a bounded four-H100 continuation, with durable reports.

Preparation verifies parent bytes and the unchanged training recipe on CPU.
Live distributed resume is validated by the original trainer on the ACP node.
No network or repository synchronization occurs in this launcher.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def continuation_config(parent, *, state, output, parent_step, target_step):
    from copy import deepcopy
    from fastwam.models.warm.training_attestation import training_config_hashes
    if not parent_step < target_step <= int(parent['max_steps']):
        raise ValueError('target must advance the parent within its original schedule')
    child=deepcopy(parent)
    child.update(output_dir=str(output),resume=str(state),run_steps=target_step-parent_step)
    if training_config_hashes(child)[1] != training_config_hashes(parent)[1]:
        raise ValueError('continuation changed the scientific training recipe')
    return child


def prepare(args):
    from omegaconf import OmegaConf
    from fastwam.models.warm.training_attestation import (
        training_config_hashes, verify_training_attestation, sha256_training_state_tree)
    from nonreal_job import file_hash, source_identity
    root=args.output.resolve()
    if root.is_relative_to(ROOT):
        raise ValueError('run root must be outside source checkout')
    root.mkdir(parents=True,exist_ok=False)
    cfg=OmegaConf.to_container(OmegaConf.load(args.parent_config),resolve=True)
    state=args.state.resolve()
    state_meta=json.loads((state/'trainer_state.json').read_text())
    step=int(state_meta['global_step'])
    weights=state.parent.parent/'weights'/f'step_{step:06d}.pt'
    print('stage=verify_parent_weights',flush=True)
    parent=verify_training_attestation(weights,weights.with_suffix('.training.json'))
    full,shared=training_config_hashes(cfg)
    if full!=parent.resolved_train_config_sha256 or shared!=parent.shared_recipe_sha256:
        raise ValueError('supplied parent config is not the attested config')
    if parent.world_size!=4 or parent.effective_batch_size!=128 or parent.checkpoint_step!=step:
        raise ValueError('expected the original four-rank global-batch-128 state')
    for rank in range(4):
        for path in (state/f'random_states_{rank}.pkl',
                     state/'pytorch_model'/f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'):
            if not path.is_file():
                raise ValueError(f'missing rank state: {path}')
    if not (state/'scheduler.bin').is_file():
        raise ValueError('missing scheduler state')
    child=continuation_config(cfg,state=state,output=root/'training',parent_step=step,target_step=args.target_step)
    config_path=root/'resume_train.yaml'
    OmegaConf.save(OmegaConf.create(child),config_path)
    print('stage=hash_parent_state',flush=True)
    state_hash=sha256_training_state_tree(state)
    plan=dict(schema='warm.nonreal.resume.v1',parent_step=step,target_step=args.target_step,
              config=str(config_path),config_sha256=file_hash(config_path),parent_state=str(state),
              parent_state_sha256=state_hash,parent_weights=str(weights),parent_weights_sha256=parent.checkpoint_sha256,
              parent_attestation=str(weights.with_suffix('.training.json')),
              parent_attestation_sha256=file_hash(weights.with_suffix('.training.json')),
              shared_recipe_sha256=shared,training_output=str(root/'training'),
              expected_runtime=json.loads(parent.training_runtime_json),source_sha256=source_identity(ROOT)['sha256'],
              max_seconds=22*3600,qualification='CPU parent identity and recipe checks; live four-rank resume pending')
    write_json(root/'plan.json',plan)
    print(json.dumps(plan),flush=True)


def live_preflight(plan):
    import torch
    expected=plan['expected_runtime']
    actual=dict(python_version=platform.python_version(),platform=platform.platform(),
                torch_version=str(torch.__version__),torch_cuda_version=str(torch.version.cuda),
                cudnn_version=torch.backends.cudnn.version(),
                accelerate_version=importlib.metadata.version('accelerate'),
                deepspeed_version=importlib.metadata.version('deepspeed'))
    different=[key for key,value in actual.items() if value!=expected[key]]
    if different:
        raise ValueError('ACP runtime differs from original resume contract: '+', '.join(different))
    if torch.cuda.device_count()!=4:
        raise ValueError('this continuation requires exactly four visible H100 GPUs')
    for index, original in enumerate(expected['gpu_devices']):
        prop=torch.cuda.get_device_properties(index)
        if prop.name!=original['name'] or list(torch.cuda.get_device_capability(index))!=original['capability'] or prop.total_memory<75*1024**3:
            raise ValueError(f'GPU {index} differs from parent hardware')
    return actual


def collect_metrics(path):
    from qualify_warm_rmbench_training_smoke import load_training_metrics
    if not path.exists():
        return []
    return load_training_metrics(path)


def summarize(plan, *, exit_code, reason=None):
    from nonreal_job import file_hash
    from fastwam.models.warm.training_attestation import verify_training_attestation
    root=Path(plan['training_output'])
    rows=collect_metrics(root/'training_metrics.jsonl')
    checkpoints=[]
    for path in sorted((root/'checkpoints/state').glob('step_*/trainer_state.json')):
        meta=json.loads(path.read_text())
        tag=path.parent.name
        weights=root/'checkpoints/weights'/f'{tag}.pt'
        sidecar=weights.with_suffix('.training.json')
        if weights.is_file() and sidecar.is_file():
            checkpoints.append(dict(step=int(meta['global_step']),state=str(path.parent),weights=str(weights),attestation=str(sidecar)))
    target=next((item for item in checkpoints if item['step']==plan['target_step']),None)
    complete=exit_code==0 and reason is None and bool(rows) and rows[-1]['step']==plan['target_step'] and target is not None
    if complete:
        # Verify final published bytes, not just a successful launcher exit.
        attestation=verify_training_attestation(target['weights'],target['attestation'])
        if attestation.resume_step!=plan['parent_step'] or attestation.parent_checkpoint_sha256!=plan['parent_weights_sha256'] or attestation.shared_recipe_sha256!=plan['shared_recipe_sha256']:
            raise ValueError('final checkpoint continuation lineage mismatch')
        target['verified_weights_sha256']=attestation.checkpoint_sha256
    report=dict(status='complete' if complete else 'incomplete',exit_code=exit_code,reason=reason,
                parent_step=plan['parent_step'],target_step=plan['target_step'],metric_records=len(rows),
                last_metrics=rows[-1] if rows else None,checkpoints=checkpoints,
                metrics_sha256=file_hash(root/'training_metrics.jsonl') if rows else None,
                limitation='Training continuation evidence only; no task SR, paired CI, or independently labelled gate acceptance.')
    destination=Path(plan['config']).parent
    write_json(destination/'summary.json',report)
    lines=['# WARM four-H100 continuation','',f"Status: {report['status']}",
           f"Parent / target step: {plan['parent_step']} / {plan['target_step']}",
           f"Exit: {exit_code}; reason: {reason}",'',
           '| Step | Loss | Gate loss | Gate gradient | Learned gate | LR | Steps/s |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m=row['metrics']
        lines.append('| '+ ' | '.join(str(v) for v in [row['step'],row['loss'],m.get('loss_warm_gate'),m.get('warm_grad_source_gate'),m.get('warm_learned_gate_mean'),row['learning_rate'],row['steps_per_second']])+' |')
    lines += ['',report['limitation'],'',f"Exact source SHA256: {plan['source_sha256']}",f"Config SHA256: {plan['config_sha256']}",f"Raw outputs: {root}"]
    (destination/'EXPERIMENT_RECORD.generated.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def run(args):
    import fcntl
    from nonreal_job import file_hash,source_identity,run as run_logged
    from fastwam.models.warm.training_attestation import sha256_training_state_tree
    plan=json.loads(args.plan.read_text())
    root=args.plan.resolve().parent
    if plan.get('schema')!='warm.nonreal.resume.v1':
        raise ValueError('unsupported resume plan')
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'started.json').exists():
            raise ValueError('run already started; preserve it and prepare a new continuation from its last complete state')
        code=2
        reason=None
        try:
            source=source_identity(ROOT)
            if source['sha256']!=plan['source_sha256']:
                raise ValueError('source snapshot changed after preparation')
            for path,key in [(plan['config'],'config_sha256'),(plan['parent_weights'],'parent_weights_sha256'),(plan['parent_attestation'],'parent_attestation_sha256')]:
                if file_hash(Path(path))!=plan[key]:
                    raise ValueError('prepared input changed: '+path)
            if sha256_training_state_tree(plan['parent_state'])!=plan['parent_state_sha256']:
                raise ValueError('parent optimizer state changed')
            runtime=live_preflight(plan)
            write_json(root/'started.json',dict(start_unix=time.time(),runtime_preflight=runtime))
            command=[sys.executable,'-m','accelerate.commands.launch','--config_file',str(ROOT/'scripts/accelerate_configs/accelerate_zero1_ds.yaml'),
                     '--num_processes','4','--num_machines','1','--main_process_port',str(args.port),str(ROOT/'scripts/train.py'),
                     '--config-path',str(root),'--config-name','resume_train']
            spec=dict(max_seconds=plan['max_seconds'],evidence_type='training',claim='Continue the original shared recipe from its complete optimizer state')
            code=run_logged(spec,command,root/'job',source=source)
        except Exception as error:
            reason=f'{type(error).__name__}: {error}'
            print(reason,file=sys.stderr,flush=True)
        finally:
            try:
                report=summarize(plan,exit_code=code,reason=reason)
                print(json.dumps(report),flush=True)
                if report['status']!='complete':
                    code=code or 3
            except Exception as error:
                write_json(root/'summary_error.json',dict(error=f'{type(error).__name__}: {error}',exit_code=code))
                code=code or 4
        return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare')
    p.add_argument('--parent-config',type=Path,required=True)
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target-step',type=int,required=True)
    p=commands.add_parser('run')
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--port',type=int,default=29622)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args)
        return 0
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
