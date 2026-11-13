"""Qualify a new legacy pooled control on sealed factual inputs, without rollout.

Run only from an immutable assembly of the audited original v3 source plus the
new research factory. A finite action is an engineering check, never a success
rate. The existing trained Full/Pooled checkpoints are not loaded as this model.
"""
from pathlib import Path
import argparse,dataclasses,datetime,hashlib,json,os,time,traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--sealed-dir',type=Path,required=True)
    p.add_argument('--infer-kwargs',type=Path,required=True)
    p.add_argument('--model-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sample',action='store_true')
    p.add_argument('--resume-qualification-from',type=Path)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    start=time.monotonic();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
    def sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    state=dict(status='starting',started_utc=utc(),scope='New untrained control:actual input and production sampler qualification only',
               optimizer_updates=0,task_rollout_commands=0,success_rate_eligible=False,actual_policy_generations=0,policy_generation_attempts=0,new_GPU_hours=0)
    def write(name,value):
        path=a.output/name;temp=path.with_suffix(path.suffix+'.partial');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)
    def update(status):state.update(status=status,heartbeat_utc=utc());write('operator.json',state)
    update('checking_factual_inputs')
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('This intake is explicitly CPU-only')
        if not a.model_root.is_dir():raise ValueError('The confirmed local model root must exist')
        os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            DIFFSYNTH_SKIP_DOWNLOAD='true',DIFFSYNTH_MODEL_BASE_PATH=str(a.model_root),
            TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
        import torch,numpy as np
        from omegaconf import OmegaConf
        # The original package exports warm classes while importing FastWAM.
        # Bind its runtime first, as in the original production entrypoint.
        from fastwam.runtime import build_datasets
        from fastwam.research.legacy_pooled_memory_prior import create_legacy_pooled_memory_prior,legacy_api,REMOVED_LOSSES
        legacy,_=legacy_api()
        from fastwam.models.wan22.fastwam import FastWAM
        from fastwam.utils.misc import register_work_dir
        torch.set_num_threads(4);torch.manual_seed(3407)
        recipe=OmegaConf.load(a.recipe)
        original=OmegaConf.to_container(recipe.model,resolve=True)
        params=dict(original);params.pop('_target_')
        params['retrospection']=dict(params['retrospection'])
        for key in REMOVED_LOSSES:params['retrospection'][key]=0.0
        write('derived_control_model_config.json',dict(original=original,derived=params,
             declared_changes=['factory_target']+list(REMOVED_LOSSES),same_data_and_normalization=True))
        context_file=a.sealed_dir/'actual_q8_source_context.pt'
        input_file=a.sealed_dir/'actual_q8_inputs.npz'
        history_file=a.sealed_dir/'actual_q8_history.npz'
        witness_file=a.sealed_dir/'original_Full_source_witness.pt'
        factual=torch.load(context_file,map_location='cpu',weights_only=True)
        infer=torch.load(a.infer_kwargs,map_location='cpu',weights_only=True)
        step=infer['online_step'];prompt=step['prompt'];seed=int(step['derived_seed'])
        cache_root=Path(recipe.data.train.text_embedding_cache_dir)
        text_file=cache_root/(hashlib.sha256(prompt.encode('utf-8')).hexdigest()+f'.t5_len{original["tokenizer_max_len"]}.wan22ti2v5b.pt')
        text=torch.load(text_file,map_location='cpu',weights_only=True)
        paths=[a.recipe,context_file,input_file,history_file,witness_file,a.infer_kwargs,text_file]
        before={str(f):dict(bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns,sha256=sha(f)) for f in paths}
        write('input_identities_before.json',before)
        previous=a.resume_qualification_from
        if previous is not None:
            old=json.loads((previous/'operator.json').read_text())
            if old.get('status')!='failed' or old.get('actual_policy_generations')!=1 or "result['action'].shape" not in old.get('traceback',''):
                raise ValueError('Only the known post-sampler action-shape validator failure can resume here')
            assert json.loads((previous/'input_identities_before.json').read_text())==before
            native_output=torch.load(previous/'actual_sampler_output.pt',map_location='cpu',weights_only=True)
            assert native_output['action'].shape==(32,14) and torch.isfinite(native_output['action']).all()
            old_manifest=json.loads((previous.parent/'SOURCE_MANIFEST_V1.json').read_text())
            core=Path(__file__).resolve().parents[1]/'src/fastwam/research/legacy_pooled_memory_prior.py'
            assert sha(core)==old_manifest['src/fastwam/research/legacy_pooled_memory_prior.py']
            state.update(upstream_policy_generations=1,upstream_results=str(previous),sampler_not_repeated=True)
        forbidden={'target_action','target_effect','current_semantic_teacher','future_valid_mask','target_action_valid_mask'}
        if any(factual.get(k) is not None for k in forbidden):raise ValueError('An inference record contains a future teacher')
        names={f.name for f in dataclasses.fields(legacy.RetrospectiveSourceContext)}
        if set(factual)-names:raise ValueError('Unexpected fields in the original factual context')
        ctx=legacy.RetrospectiveSourceContext(**{k:v.to(dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) if torch.is_tensor(v) else v for k,v in factual.items()})
        with np.load(input_file,allow_pickle=False) as z:
            image=torch.from_numpy(z['model_input'].copy());proprio=torch.from_numpy(z['proprio'].copy())
        update('constructing_original_backbone_and_new_control_on_CPU')
        model=create_legacy_pooled_memory_prior(**params,model_dtype=torch.bfloat16,device='cpu')
        assert not torch.cuda.is_initialized()
        model._validate_retrospection_state(model._retrospection_state())
        write('construction.json',dict(model_class=type(model).__name__,checkpoint_state=model._retrospection_state(),
            parameters=sum(v.numel() for v in model.parameters()),pool_parameters=sum(v.numel() for v in model.retrospective_event_adapter.parameters()),
            common_base_checkpoint=original['base_checkpoint_path'],Full_specialist_checkpoint_loaded=False,
            existing_shared_Pooled_checkpoint_loaded=False,CUDA_initialized=False))
        torch.save(dict(schema='warm.engineering.untrained-legacy-control-compact.v1',is_trained=False,
            checkpoint_state=model._retrospection_state(),modules={name:getattr(model,name).state_dict() for name in [
                'semantic_bridge','retrospective_gist','retrospective_event_adapter','episode_action_projection','gist_to_text','action_context_to_text']}),
            a.output/'untrained_compact_modules.pt')
        if previous is None:
            update('reading_fixed_original_training_index_zero')
            register_work_dir(a.output/'work')
            dataset,_=build_datasets(recipe.data,build_validation=False)
            model.validate_training_dataset(dataset)
            sample=dataset[0]
            batch={k:v[None] if torch.is_tensor(v) else v for k,v in sample.items()}
            train_ctx=model._training_source_context(batch)
            write('actual_training_context.json',dict(fixed_index=0,
                fields={f.name:dict(shape=list(getattr(train_ctx,f.name).shape),dtype=str(getattr(train_ctx,f.name).dtype)) for f in dataclasses.fields(train_ctx) if torch.is_tensor(getattr(train_ctx,f.name))},
                actual_proprio_shape=list(batch['proprio'].shape),new_phase_fields_synthesized=False))
        else:
            reference=previous/'actual_training_context.json';assert reference.is_file()
            write('actual_training_context_reused.json',dict(original_file=str(reference),sha256=sha(reference),
                previously_completed_fixed_index=0,source_and_recipe_unchanged=True,no_new_training_sample=True))
        if a.sample:
            if previous is None:state['policy_generation_attempts']=1
            update('one_production_sampler_call_on_sealed_q8_input' if previous is None else 'resume_source_checks_from_completed_factual_sampler')
            from types import MethodType
            calls=[];pool_inputs=[];raw_gists=[]
            original_source=model._resolve_action_source
            def cpu(value):
                if torch.is_tensor(value):return value.detach().cpu().clone()
                if dataclasses.is_dataclass(value):return {f.name:cpu(getattr(value,f.name)) for f in dataclasses.fields(value)}
                if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
                if isinstance(value,(list,tuple)):return [cpu(v) for v in value]
                return value
            def capture(_self,**kwargs):
                out=original_source(**kwargs);calls.append((kwargs,out));return out
            def observe_pool(_m,args,kwargs):pool_inputs.append(cpu(kwargs))
            def observe_gist(_m,args,kwargs,out):raw_gists.append(cpu(out.gist_tokens))
            handles=[model.retrospective_event_adapter.register_forward_pre_hook(observe_pool,with_kwargs=True),
                     model.retrospective_gist.register_forward_hook(observe_gist,with_kwargs=True)]
            model._resolve_action_source=MethodType(capture,model)
            try:
                if previous is None:
                    result=FastWAM.infer_action(model,prompt=None,input_image=image,proprio=proprio,
                        context=text['context'],context_mask=text['mask'],action_horizon=32,seed=seed,
                        action_source_context=ctx,memory_sigma=.2,rand_device='cpu',
                        text_cfg_scale=float(infer['text_cfg_scale']),num_inference_steps=int(infer['num_inference_steps']),
                        sigma_shift=infer['sigma_shift'],tiled=bool(infer['tiled']))
                    state['actual_policy_generations']=1
                else:
                    saved=torch.load(previous/'actual_source_inputs.pt',map_location='cpu',weights_only=True)
                    replay_ctx=legacy.RetrospectiveSourceContext(**saved['action_source_context'])
                    for f in dataclasses.fields(ctx):
                        left,right=getattr(ctx,f.name),getattr(replay_ctx,f.name)
                        assert torch.equal(left,right) if torch.is_tensor(left) else left==right
                    saved['action_source_context']=replay_ctx;model.eval()
                    with torch.no_grad():model._resolve_action_source(**saved)
                    result=native_output
            finally:
                del model._resolve_action_source
                for handle in handles:handle.remove()
            assert len(calls)==len(pool_inputs)==len(raw_gists)==1
            if previous is not None:
                assert torch.equal(raw_gists[0],torch.load(previous/'actual_required_gist.pt',map_location='cpu',weights_only=True))
                state['required_gist_reconstruction_bit_exact']=True
            kw,out=calls[0];saved=cpu(kw);saved['action_source_context']={k:v for k,v in saved['action_source_context'].items() if k not in forbidden}
            torch.save(saved,a.output/'actual_source_inputs.pt')
            torch.save(pool_inputs[0],a.output/'actual_pool_inputs.pt')
            torch.save(raw_gists[0],a.output/'actual_required_gist.pt')
            torch.save(cpu(result),a.output/('actual_sampler_output.pt' if previous is None else 'upstream_sampler_output.pt'))
            old_witness=torch.load(witness_file,map_location='cpu',weights_only=True)
            assert torch.equal(out.base_gaussian,old_witness['base_gaussian'])
            assert result['action'].shape==(32,14) and torch.isfinite(result['action']).all()
            assert out.source_gate.item()==.25
            model.eval()
            with torch.no_grad():
                full=original_source(**kw)
                null_kw=dict(kw,action_source_context=dataclasses.replace(ctx,forced_candidate_indices=torch.full((1,),-1,dtype=torch.long)))
                null=original_source(**null_kw)
                repeat=original_source(**kw)
            assert torch.equal(full.source,repeat.source) and torch.equal(full.conditioning_tokens,repeat.conditioning_tokens)
            assert torch.equal(null.source,null.base_gaussian) and not torch.count_nonzero(null.conditioning_tokens[:,8:])
            assert torch.equal(full.conditioning_tokens[:,:8],null.conditioning_tokens[:,:8])
            adapter=model.retrospective_event_adapter
            adapter.requires_grad_(True);adapter.zero_grad(set_to_none=True)
            differentiable=dict(pool_inputs[0])
            for key in ['gist_tokens','proprio','query_context']:
                differentiable[key]=differentiable[key].detach().clone().requires_grad_(True)
            with torch.enable_grad():
                _,tokens=adapter(**differentiable)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(137);probe=(tokens.float()*torch.randn_like(tokens.float())).sum()
                probe.backward()
            grads=[x.grad for x in adapter.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            assert all(differentiable[key].grad is not None and differentiable[key].grad.abs().sum()>0 for key in ['gist_tokens','proprio','query_context'])
            write('actual_sampler_qualification.json',dict(untrained_control=True,actual_policy_generations=state['actual_policy_generations'],
                upstream_actual_policy_generations=1 if previous is not None else 0,upstream_results=str(previous) if previous is not None else None,
                full_sampler_output_finite=True,action_shape=[32,14],source_shape=[1,32,14],fixed_source_weight=.25,
                actual_raw_gist_shape=list(raw_gists[0].shape),actual_proprio_shape=list(pool_inputs[0]['proprio'].shape),
                original_Gaussian_exact=True,source_Full_null_Full_repeat_exact=True,
                common_eight_history_tokens_retained=True,null_candidate_tokens_zero=True,
                actual_input_pool_gradients_finite=True,history_proprio_query_gradients_nonzero=True,synthetic_history_or_proprio=False,
                task_rollout_commands=0,training_step_qualified=False,trained_control_available=False,success_rate_eligible=False))
        after={str(f):dict(bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns,sha256=sha(f)) for f in paths}
        assert before==after and not torch.cuda.is_initialized()
        write('input_identities_after.json',after);state.update(status='complete',exit_code=0,original_inputs_unchanged=True)
    except BaseException as exc:
        state.update(status='failed',exit_code=1,error=str(exc),traceback=traceback.format_exc())
    finally:
        state.update(ended_utc=utc(),elapsed_seconds=time.monotonic()-start);write('operator.json',state)
    print(json.dumps(state));return state['exit_code']


if __name__=='__main__':raise SystemExit(main())
