#!/usr/bin/env python3
"""Two real training microbatches plus isolated gate optimizer diagnostics.

No production checkpoint is changed. Full-model backward verifies the actual
gradient path. Fresh gate-only AdamW comparisons are NOT a distributed resume.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def native_bf16_training_loss(model, sample):
    """Match Accelerate+DeepSpeed BF16, which disables native AMP autocast."""
    import torch
    with torch.autocast(device_type=torch.device(model.device).type, enabled=False):
        return model.training_loss(sample)


def compare_updates(state, gradients, *, learning_rate, weight_decay):
    import torch
    output={}
    for name,dtype in (('bf16_direct',torch.bfloat16),('fp32_master',torch.float32)):
        params={key:torch.nn.Parameter(value.detach().cpu().to(dtype).clone()) for key,value in state.items()}
        optimizer=torch.optim.AdamW(list(params.values()),lr=learning_rate,weight_decay=weight_decay,betas=(.9,.95),foreach=False)
        before={key:value.detach().clone() for key,value in params.items()}
        for key,value in params.items():
            value.grad=gradients[key].detach().cpu().to(dtype).clone()
        optimizer.step()
        output[name]={key:dict(max_abs_update=float((value.detach().float()-before[key].float()).abs().max()),
                              changed_elements=int((value.detach()!=before[key]).sum()),
                              changed_after_bf16_cast=int((value.detach().bfloat16()!=before[key].bfloat16()).sum()))
                      for key,value in params.items()}
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--asset-root',type=Path,required=True)
    parser.add_argument('--training-metrics',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError('output must be outside source checkout')
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      DIFFSYNTH_SKIP_DOWNLOAD='true',DIFFSYNTH_MODEL_BASE_PATH=str(args.asset_root),WANDB_MODE='disabled')
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import default_collate
    from fastwam.runtime import _wrap_warm_candidate_dataset
    from fastwam.trainer import Wan22Trainer
    from fastwam.utils.misc import register_work_dir
    from fastwam.memory.manifest import sha256_file
    from fastwam.research.evidence import append_record
    from probe_nonreal_full_null import select_prefixes

    started=time.perf_counter()
    register_work_dir(args.output)
    cfg=OmegaConf.load(args.train_config)
    if not cfg.data.train.is_training_set or cfg.data.train.episode_split != 'train':
        raise ValueError('requires an explicit TRAIN dataset, not DEV')
    tasks=['press_button','put_back_block']
    cfg.data.train.episode_task_allowlist=tasks
    (args.output/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
    history=[json.loads(line) for line in args.training_metrics.read_text().splitlines()]
    last=history[-1]
    learning_rate=float(last['learning_rate'])
    print('stage=build_train_dataset',flush=True)
    dataset=_wrap_warm_candidate_dataset(instantiate(cfg.data.train),cfg.data.warm_candidates.train,expected_query_split='train')
    prefixes=select_prefixes(dataset.sampling_query_records(),tasks,1,1)
    (args.output/'frozen_prefixes.json').write_text(json.dumps(prefixes,indent=2))
    torch.manual_seed(int(cfg.seed))
    print('stage=load_model',flush=True)
    model=instantiate(cfg.model,model_dtype=torch.bfloat16,device='cuda:0')
    print('stage=load_checkpoint',flush=True)
    model.load_checkpoint(str(args.checkpoint))
    print('stage=validate_dataset',flush=True)
    model.validate_training_dataset(dataset)
    parameters=Wan22Trainer._apply_dit_only_train_mode(model,training_stage='shared')
    parameter_ids={id(value) for value in parameters}
    gate=model.source_confidence_gate
    gate_parameters=dict(gate.named_parameters())
    registered=all(value.requires_grad and id(value) in parameter_ids for value in gate_parameters.values())
    if not registered:
        raise RuntimeError('gate parameter missing from trainer optimizer parameter set')
    initial={key:value.detach().cpu().clone() for key,value in gate_parameters.items()}
    manifest=dict(checkpoint_sha256=model._warm_loaded_checkpoint_sha256,config_sha256=sha256_file(args.train_config),
                  training_metrics_sha256=sha256_file(args.training_metrics),historical_step=last['step'],
                  historical_learning_rate=learning_rate,historical_gate_gradient=last['metrics'].get('warm_grad_source_gate'),
                  bank_sha256=dataset.resolver.bank_content_sha256,query_corpus_sha256=dataset.resolver.query_corpus_sha256,
                  registered_gate_parameters=registered,trainable_parameters=sum(p.numel() for p in parameters),
                  batch_size=1,gradient_accumulation=1,precision='bf16',native_autocast=False,seed=int(cfg.seed),gpu=torch.cuda.get_device_name(0),
                  scope='engineering gradient-path and fresh gate-only optimizer probe; no model update or distributed resume')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    for number,prefix in enumerate(prefixes):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(int(cfg.seed)+number)
        sample=default_collate([dataset[prefix['dataset_sample_index']]])
        sample={key:value.to('cuda:0') if isinstance(value,torch.Tensor) else value for key,value in sample.items()}
        torch.cuda.reset_peak_memory_stats()
        tick=time.perf_counter()
        print(f'stage=forward task={prefix["task"]}',flush=True)
        loss,metrics=native_bf16_training_loss(model,sample)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('nonfinite training loss')
        print(f'stage=backward task={prefix["task"]}',flush=True)
        loss.backward()
        torch.cuda.synchronize()
        missing=[key for key,value in gate_parameters.items() if value.grad is None]
        if missing:
            raise RuntimeError(f'gate gradients missing: {missing}')
        gradients={key:value.grad.detach().cpu().clone() for key,value in gate_parameters.items()}
        if any(not bool(torch.isfinite(value).all()) for value in gradients.values()):
            raise RuntimeError('nonfinite gate gradients')
        global_norm=float(torch.nn.utils.clip_grad_norm_(parameters,float(cfg.max_grad_norm)))
        if not __import__('math').isfinite(global_norm):
            raise RuntimeError('nonfinite full gradient norm')
        clipped={key:value.grad.detach().cpu().clone() for key,value in gate_parameters.items()}
        comparisons=compare_updates(initial,clipped,learning_rate=learning_rate,weight_decay=float(cfg.weight_decay))
        checkpoint=args.output/f'gate_probe_{number}.pt'
        torch.save(dict(state=initial,gradients=gradients,clipped_gradients=clipped),checkpoint)
        restored=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if not all(torch.equal(initial[key],restored['state'][key]) for key in initial):
            raise RuntimeError('gate diagnostic save/load mismatch')
        row=dict(prefix,loss=float(loss.detach()),metrics={key:float(value) for key,value in metrics.items()},
                 gate_gradients={key:dict(norm=float(value.float().norm()),nonzero=int(torch.count_nonzero(value))) for key,value in gradients.items()},
                 global_grad_norm_before_clip=global_norm,update_comparison=comparisons,gate_tensor_roundtrip=True,
                 seconds=time.perf_counter()-tick,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                 artifact_sha256=sha256_file(checkpoint))
        append_record(args.output/'batch_results.jsonl',row)
        rows.append(row)
        print(json.dumps(row),flush=True)
        del loss,sample,metrics
    unchanged=all(torch.equal(value.detach().cpu(),initial[key]) for key,value in gate_parameters.items())
    if not unchanged:
        raise RuntimeError('diagnostic unexpectedly changed model gate')
    summary=dict(status='complete',batches=len(rows),elapsed_seconds=time.perf_counter()-started,
                 gate_registered=registered,model_gate_unchanged=unchanged,
                 gate_nonzero_gradient_batches=sum(any(v['nonzero'] for v in row['gate_gradients'].values()) for row in rows),
                 limitation='Two TRAIN microbatches; fresh gate-only AdamW states, not restored DeepSpeed state or full training resume; no paper SR')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
