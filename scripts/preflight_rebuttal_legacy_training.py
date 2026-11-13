"""Run one bounded, factual GPU optimizer step for the new legacy control.

This engineering check is deliberately not a matched training run. Its single
sample and updated weights must not seed the registered specialist comparison.
Run inside the separately frozen original-v3 source assembly only.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,time,traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixed-index',type=int,default=0)
    a=p.parse_args()
    if a.fixed_index!=0:raise ValueError('This qualification fixes original training sample zero')
    a.output.mkdir(parents=True,exist_ok=False)
    utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();start=time.monotonic()
    def sha(f):
        h=hashlib.sha256()
        with Path(f).open('rb') as q:
            for b in iter(lambda:q.read(8*1024**2),b''):h.update(b)
        return h.hexdigest()
    def write(name,value):
        f=a.output/name;t=f.with_suffix(f.suffix+'.partial')
        t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(f)
    state=dict(started_utc=utc(),status='starting',optimizer_updates=0,task_commands=0,
        scientific_training_updates=0,success_rate_eligible=False,matched_training_qualified=False,
        GPUs=1,maximum_optimizer_updates=1,training_index=0,microbatch=1,accumulation=1,
        updated_weights_must_not_seed_matched_training=True)
    def update(s):state.update(status=s,heartbeat_utc=utc());write('operator.json',state)
    update('checking_assets')
    try:
        if not a.model_root.is_dir():raise ValueError('Local-only model root is required')
        if os.statvfs(a.output).f_bavail*os.statvfs(a.output).f_frsize<4*1024**3:
            raise ValueError('AFS admission below the unchanged 4 GiB threshold')
        os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            DIFFSYNTH_SKIP_DOWNLOAD='true',DIFFSYNTH_MODEL_BASE_PATH=str(a.model_root),
            TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
        import torch
        from omegaconf import OmegaConf
        from fastwam.runtime import build_datasets
        from fastwam.research.legacy_pooled_memory_prior import legacy_api,create_legacy_pooled_memory_prior,REMOVED_LOSSES
        legacy_api()
        from fastwam.utils.misc import register_work_dir
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError('Exactly one visible GPU is required for this engineering check')
        if torch.cuda.get_device_properties(0).total_memory<75*1024**3:
            raise ValueError('This qualification requires the confirmed 80 GB GPU class')
        recipe=OmegaConf.load(a.recipe)
        if sha(a.recipe)!='32b8d67832c49894138fbe8808b51f8feca91a65741fb4f7056935da58cccae6':
            raise ValueError('The original PB recipe identity changed')
        torch.manual_seed(int(recipe.seed));torch.cuda.manual_seed_all(int(recipe.seed));torch.set_num_threads(4)
        model_cfg=OmegaConf.to_container(recipe.model,resolve=True);params=dict(model_cfg);params.pop('_target_')
        params['retrospection']=dict(params['retrospection'])
        for name in REMOVED_LOSSES:params['retrospection'][name]=0.0
        write('declared_config.json',dict(original_recipe=str(a.recipe),recipe_sha256=sha(a.recipe),
            derived_model=params,changes=['new_control_factory']+list(REMOVED_LOSSES),
            engineering_only_batch_override='1x1x1;original matched target remains 8x4x4=128',
            original_training_budget=12000,optimizer_peak_lr=float(recipe.learning_rate),
            full_training_scheduler_not_replaced=True,GPUs=[dict(name=torch.cuda.get_device_name(0),
                total_memory=torch.cuda.get_device_properties(0).total_memory)]))
        update('constructing_original_backbone_and_new_control')
        model=create_legacy_pooled_memory_prior(**params,model_dtype=torch.bfloat16,device='cuda')
        register_work_dir(a.output/'work')
        update('loading_fixed_original_training_sample')
        dataset,_=build_datasets(recipe.data,build_validation=False);model.validate_training_dataset(dataset)
        batch=next(iter(torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)))
        def tensor_identity(t):
            c=t.detach().cpu().contiguous();data=c.view(torch.uint8).numpy().tobytes()
            return dict(shape=list(c.shape),dtype=str(c.dtype),sha256=hashlib.sha256(data).hexdigest())
        write('sample_identity.json',dict(index=0,tensors={k:tensor_identity(v) for k,v in batch.items() if torch.is_tensor(v)},
            dataset_class=type(dataset).__name__,no_synthetic_phase_or_proprio=True))
        trainable=model.configure_trainable_modules();assert trainable
        optimizer=torch.optim.AdamW(trainable,lr=float(recipe.learning_rate),betas=(.9,.95),eps=1e-8,
            weight_decay=float(recipe.weight_decay))
        optimizer.zero_grad(set_to_none=True);update('actual_action_and_video_training_forward')
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):loss,metrics=model.training_loss(batch)
        if not torch.isfinite(loss):raise ValueError('Nonfinite actual training loss')
        update('actual_joint_training_backward');loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable):
            raise ValueError('Nonfinite parameter gradient')
        groups={}
        for name in ['action_expert','semantic_bridge','retrospective_gist','retrospective_event_adapter','video_layer_adapters']:
            module=getattr(model,name);grads=[p.grad for p in module.parameters() if p.grad is not None]
            groups[name]=dict(tensors_with_grad=len(grads),nonzero=any(torch.count_nonzero(g).item()>0 for g in grads),
                norm=float(sum(g.detach().float().square().sum() for g in grads).sqrt().item()) if grads else 0.)
            if not grads or not groups[name]['nonzero']:raise ValueError('Missing actual gradient in '+name)
        norm=torch.nn.utils.clip_grad_norm_(trainable,float(recipe.max_grad_norm),error_if_nonfinite=True)
        before=[p.detach().clone() for p in model.retrospective_event_adapter.parameters()]
        update('one_engineering_optimizer_update');optimizer.step();state['optimizer_updates']=1
        changed=sum(torch.count_nonzero(old!=new.detach()).item() for old,new in zip(before,model.retrospective_event_adapter.parameters(),strict=True))
        if not changed:raise ValueError('Optimizer did not change any pooled parameter')
        if not all(torch.isfinite(p).all() for p in trainable):raise ValueError('Nonfinite updated parameter')
        scalar_metrics={k:float(v.detach().item() if torch.is_tensor(v) else v) for k,v in metrics.items()}
        write('training_step_qualification.json',dict(loss=float(loss.detach()),metrics=scalar_metrics,gradients=groups,
            unclipped_gradient_norm=float(norm),pool_changed_elements=changed,optimizer_updates=1,
            sample_index=0,action_and_video_losses_present=True,full_sampler_already_qualified_separately=True,
            matched_training_qualified=False,success_rate_eligible=False,no_updated_checkpoint_promoted=True,
            CUDA_peak_allocated=torch.cuda.max_memory_allocated(),CUDA_peak_reserved=torch.cuda.max_memory_reserved()))
        # Retain only the small changed control module, clearly isolated from all
        # production checkpoints. No full-weight or optimizer-state file is made.
        torch.save(dict(schema='warm.engineering.legacy-one-step-pool.v1',not_a_matched_training_checkpoint=True,
            original_recipe_sha256=sha(a.recipe),state={k:v.detach().cpu() for k,v in model.retrospective_event_adapter.state_dict().items()}),
            a.output/'engineering_pool_after_one_step.pt')
        state.update(status='complete',exit_code=0)
    except BaseException as exc:
        state.update(status='failed',exit_code=1,error=str(exc),traceback=traceback.format_exc())
    finally:
        state.update(ended_utc=utc(),elapsed_seconds=time.monotonic()-start);write('operator.json',state)
    print(json.dumps(state));return state['exit_code']


if __name__=='__main__':raise SystemExit(main())
