#!/usr/bin/env python3
"""Freeze matched single-H100 Full/simple-prior training segment configs.

Uses the byte-identified author recipe without inventing a resumed optimizer.
New runs start at the common release weights and share the original30k-step
scheduler, data, effective batch128 and a declared3000-update checkpoint rule.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
AUTHOR_CONFIG_SHA256='34f5dc0a6f4cc354da61c0ba7bdc394b6c9652c7ef7706e72e8cbb683ec9a00c'


def make_config(original,*,kind,output,seed,segment_updates):
    if kind not in ('full','simple_prior','conditioned_prior') or seed not in (3407,3408,3409) or not 1<=segment_updates<=250:
        raise ValueError('declared model, seed and bounded segment required')
    cfg=deepcopy(original)
    if cfg['batch_size']!=8 or cfg['gradient_accumulation_steps']!=4 or cfg['max_steps']!=30000:
        raise ValueError('author recipe differs from the inspected4-GPU schedule')
    cfg.update(output_dir=str(output),overwrite=False,resume=None,initialization_checkpoint=None,
        initialization_fork_manifest=None,seed=seed,gradient_accumulation_steps=16,
        run_steps=segment_updates,save_every=segment_updates,log_every=1,num_workers=4,
        # Both arms use the same independent DEV procedure. The control has
        # a distinct checkpoint schema and does not enter full online eval.
        eval_every=500)
    cfg['model']['_target_']=('fastwam.runtime.create_warm_retrospection' if kind=='full'
                            else 'fastwam.research.simple_prior.create_'+kind)
    cfg['model']['retrospection']['corruption_seed']=seed
    cfg['data']['train']['instruction_variant_seed']=seed
    if kind!='full':
        for name in ('loss_retrieval','loss_effect','loss_gate','loss_adaptation'):
            cfg['model']['retrospection'][name]=0.0
    return cfg


def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--author-config',type=Path,required=True)
    p.add_argument('--config-output',type=Path,required=True)
    p.add_argument('--training-output-root',type=Path,required=True)
    p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--segment-updates',type=int,default=5)
    p.add_argument('--control-kind',choices=('simple_prior','conditioned_prior'),default='simple_prior')
    a=p.parse_args()
    raw=a.author_config.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=AUTHOR_CONFIG_SHA256:raise ValueError('author recipe bytes differ')
    for path in (a.config_output,a.training_output_root):
        if not path.is_absolute() or path.resolve().is_relative_to(ROOT):raise ValueError('absolute output paths outside source required')
    a.config_output.mkdir(parents=True,exist_ok=False)
    source=yaml.safe_load(raw);records={}
    for kind in ('full',a.control_kind):
        cfg=make_config(source,kind=kind,output=a.training_output_root/f'{kind}-s{a.seed}',
            seed=a.seed,segment_updates=a.segment_updates)
        target=a.config_output/f'{kind}-s{a.seed}.yaml';target.write_text(yaml.safe_dump(cfg,sort_keys=False))
        records[kind]=dict(config=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    manifest=dict(schema='warm.rebuttal.matched-training-plan.v1',author_config_sha256=AUTHOR_CONFIG_SHA256,
        initialization='Same immutable FastWAM release; new optimizers in both arms. No claim of bitwise equivalence to the historical4-GPU resumed run.',
        world_size=1,micro_batch=8,accumulation=16,effective_batch=128,scheduler_max_steps=30000,
        final_checkpoint_updates=3000,checkpoint_selection='Fixed3000 optimizer updates; DEV reporting only, no TEST selection.',
        preflight_updates=a.segment_updates,seed=a.seed,control_kind=a.control_kind,configs=records)
    (a.config_output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))


if __name__=='__main__':main()
