"""Exactly reproduce original BT q8 before sealing an explicit null action.

No simulator or historical online capability is constructed. Full reproduction
must pass before any null inference. Synthetic candidate perturbation is only
an engineering invariance check, never a replacement scientific sample.
"""
from pathlib import Path
from contextlib import contextmanager
from types import MethodType
import argparse,dataclasses,datetime,hashlib,importlib.util,json,os,signal,sys,traceback


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence-root',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--admission',type=Path);ap.add_argument('--expected-admission-sha256')
    cli=ap.parse_args();P=cli.evidence_root.resolve(strict=True);out=cli.output.resolve();ROOT=Path(__file__).resolve().parents[1]
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if out.exists() or out.parent!=P:raise ValueError('fresh flat Git-external output required')
    C=P/'historical_specialist_source_f77c633/code';S=P/'original_BT12000_cohort_case0_native1000_v1';R=P/'ITER30_ORIGINAL_Q8_GPU_RECOVERY_V2'
    identity=load('identity',P/'controller_eb70b41/code/scripts/run_rebuttal_training_with_runtime_trace.py')
    source_sha='ff32342d286929ddc5904d55814c66e965269f6b749c1e1b6df1060aaef0b7fe'
    assert identity.source_digest(C)==source_sha
    binding=read(R/'recovery_binding.json');recovery=read(R/'operator.json')
    assert recovery['full_factual_context_recovered'] and recovery['original_scientific_input_payload_hashes_exact']
    assert sha(R/'recovered_source_context.pt')==binding['source_context_sha256'] and sha(R/'recovered_query_inputs.npz')==binding['query_inputs_sha256']
    assert binding['query']==8 and binding['frame']==32 and binding['checkpoint_sha256']=='27c137d689007b38f6204d3dd12bb038550b1e58403506d7a9894bd6fcd43a9f'
    os.environ.update(TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        DIFFSYNTH_SKIP_DOWNLOAD='true',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',WARM_IMMUTABLE_ASSET_HASH_CACHE='1')
    os.environ.pop('NVIDIA_TF32_OVERRIDE',None)
    if cli.preflight_only:os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path[:0]=[str(C/'src'),str(C)]
    import numpy as np,torch
    from fastwam.memory.manifest import sha256_array,sha256_canonical_json
    from fastwam.models.warm.retrospection_model import RetrospectiveSourceContext
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.memory.runtime_fingerprint import current_encoder_runtime
    from fastwam.models.warm.online_contract import WarmOnlineRunContract
    payload=torch.load(R/'recovered_source_context.pt',map_location='cpu',weights_only=True)
    forbidden={'target_action','target_effect','current_semantic_teacher','target_action_valid_mask','future_valid_mask'}
    assert not forbidden.intersection(payload) and all(torch.is_tensor(v) for v in payload.values())
    context=RetrospectiveSourceContext(**payload)
    assert tuple(context.candidate_actions.shape)==(1,32,32,14) and tuple(context.episode_tokens.shape)==(1,16,768)
    assert all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in payload.values())
    with np.load(R/'recovered_query_inputs.npz',allow_pickle=False) as arrays:image=arrays['model_input'].copy();proprio=arrays['proprio'].copy()
    assert sha256_array(image)==binding['original_model_input_sha256'] and sha256_array(proprio)==binding['original_proprio_sha256']
    original=torch.load(S/'probes/q0008/model_output.pt',map_location='cpu',weights_only=True)
    expected_action=original['action'].float().numpy().reshape(32,14)
    args=read(S/'runtime_args.json');contract=WarmOnlineRunContract.from_dict(read(Path(args['warm_online_contract_path'])))
    admission=None;end=None;interrupted=False
    def stop(*_):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if not cli.preflight_only:
        if cli.admission is None or sha(cli.admission)!=cli.expected_admission_sha256:raise ValueError('authenticated new generation-stage admission required')
        admission=read(cli.admission);start=datetime.datetime.fromisoformat(admission['first_admitted_utc']);end=datetime.datetime.fromisoformat(admission['absolute_deadline_utc'])
        assert admission['schema']=='warm.original-q8-full-null-generation.v1' and 0<(end-start).total_seconds()<=3000
        assert start<=datetime.datetime.now(datetime.timezone.utc)<end and admission['maximum_policy_inferences']==4 and admission['physical_branches']==0
        assert admission['recovery_binding_sha256']==sha(R/'recovery_binding.json') and Path(admission['output']).resolve()==out
        assert admission['entry_sha256']==sha(Path(__file__)) and admission['controller_source_sha256']==identity.source_digest(ROOT)
    out.mkdir();state=dict(status='authenticated_original_context_ready',query=8,frame=32,policy_inferences=0,physical_branches=0,
        original_source_sha256=source_sha,recovery_binding_sha256=sha(R/'recovery_binding.json'),Full_replay_exact=False,
        explicit_null_qualified=False,synthetic_variation_used_for_physical_evidence=False)
    def save():
        state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(out/'operator.json').write_text(json.dumps(state,indent=2))
    save()
    if cli.preflight_only:
        assert not torch.cuda.is_initialized();state.update(status='complete_CPU_original_full_null_input_preflight',CUDA_initialized=False);save();print(json.dumps(state));return
    (out/'admission_snapshot.json').write_bytes(cli.admission.read_bytes())
    def budget(before_load=False):
        if interrupted or datetime.datetime.now(datetime.timezone.utc)>=end:raise TimeoutError('generation-stage window ended;never reset')
        if os.statvfs(out).f_bavail*os.statvfs(out).f_frsize<4*1024**3:raise OSError('preserve concurrent evidence:storage guard')
        if torch.cuda.mem_get_info()[0]<(32 if before_load else 8)*1024**3:raise OSError('protect active policy:free GPU memory guard')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>32*1024**2:raise OSError('generation artifact budget exceeded')
    policy=None
    try:
        torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.45);budget(True)
        actual=current_encoder_runtime(args['warm_dino_device']);assert sha256_canonical_json(actual)==contract.encoder_runtime_sha256
        (out/'actual_encoder_runtime.json').write_text(json.dumps(actual,indent=2))
        import experiments.rmbench.warm_policy.deploy_policy as deployment
        helpers=load('asset_helpers',ROOT/'src/fastwam/research/native_null_replay.py')
        withdraw=load('legacy_withdrawal',ROOT/'src/fastwam/research/legacy_gate_intervention.py').legacy_candidate_withdrawal
        asset_root=helpers.asset_working_directory(args);os.environ['DIFFSYNTH_MODEL_BASE_PATH']=str(asset_root/'checkpoints');os.chdir(asset_root)
        args['warm_telemetry_path']=str(out/'policy_loader_telemetry.jsonl')
        old_git=deployment._git_identity
        def frozen_identity():
            assert identity.source_digest(C)==source_sha
            return 'f77c63385c747fdc1424386489cbf9c7ea57ddc5',False
        deployment._git_identity=frozen_identity
        state['status']='loading_original_checkpoint_policy';save()
        try:
            with helpers.bound_asset_metadata(deployment,args,lambda x:(out/'asset_metadata.json').write_text(json.dumps(x,indent=2))):policy=deployment.RMBenchWarmPolicy(args)
        finally:deployment._git_identity=old_git
        model=policy.model;assert model._warm_loaded_checkpoint_sha256==binding['checkpoint_sha256'] and model.torch_dtype==torch.bfloat16
        versions=[(n,p,int(p._version)) for n,p in list(model.named_parameters())+list(model.named_buffers())]
        ctx=RetrospectiveSourceContext(**{k:v.detach().clone().to(model.device) for k,v in payload.items()})
        @contextmanager
        def capture_full():
            assert '_resolve_action_source' not in model.__dict__
            method=model._resolve_action_source;cache={}
            def observe(_self,*a,**kw):
                assert not cache and kw['phase']=='infer'
                value=method(*a,**kw)
                for key in ['source','base_gaussian','conditioning_tokens','source_gate']:cache[key]=getattr(value,key).detach().cpu().clone()
                return value
            model._resolve_action_source=MethodType(observe,model)
            try:yield cache
            finally:del model._resolve_action_source
        def generate(context):
            budget()
            output=FastWAM.infer_action(model,prompt=binding['prompt'],input_image=torch.tensor(image,device=model.device,dtype=model.torch_dtype),
                proprio=torch.tensor(proprio,device=model.device,dtype=model.torch_dtype),action_horizon=32,action_source_context=context,
                seed=int(binding['derived_seed']),memory_sigma=.2,**binding['sampler'])
            state['policy_inferences']+=1
            return output,output['action'].detach().float().cpu().numpy().reshape(32,14).copy()
        results={};witnesses={}
        with torch.inference_mode():
            state['status']='reproducing_original_Full';save()
            with capture_full() as witness:output,action=generate(ctx)
            results['Full']=action;witnesses['Full']=witness;np.save(out/'Full_model_actions.npy',action,allow_pickle=False);torch.save(witness,out/'Full_source_witness.pt')
            assert np.array_equal(action,expected_action),'Original Full H32 differs;stop before null'
            for key in ['source_component','source_memory_mask','source_gate']:
                actual=output[key].detach().cpu();expected=original[key]
                assert actual.shape==expected.shape and torch.equal(actual.float(),expected.float()),key
            state['Full_replay_exact']=True;save()
            for role,current in [('candidate-null',ctx),('candidate-null-input-invariance',dataclasses.replace(ctx,
                    candidate_actions=ctx.candidate_actions*1.07,candidate_effect_pre=ctx.candidate_effect_pre*.97,
                    candidate_effect_delta=ctx.candidate_effect_delta*1.03,candidate_context=ctx.candidate_context*.95))]:
                state['status']='generating_'+role;save()
                with withdraw(model) as witness:_,action=generate(current)
                assert torch.equal(witness['base_gaussian'],witnesses['Full']['base_gaussian'])
                results[role]=action;witnesses[role]=witness;np.save(out/(role+'_model_actions.npy'),action,allow_pickle=False);torch.save(witness,out/(role+'_source_witness.pt'))
            assert np.array_equal(results['candidate-null'],results['candidate-null-input-invariance']),'Withdrawn action still depends on candidate content'
            state['status']='reproducing_final_Full';save()
            with capture_full() as witness:_,action=generate(ctx)
            assert np.array_equal(action,expected_action) and all(torch.equal(witness[k],witnesses['Full'][k]) for k in witness)
            results['Full-final']=action;np.save(out/'Full-final_model_actions.npy',action,allow_pickle=False)
        assert state['policy_inferences']==4 and all(int(p._version)==v for n,p,v in versions)
        after=dict(list(model.named_parameters())+list(model.named_buffers()))
        assert set(after)=={n for n,p,v in versions} and all(after[n] is p for n,p,v in versions), 'Model parameter or buffer object changed'
        native={k:np.asarray(policy.action_normalizer.denormalize(v),dtype=np.float32).copy() for k,v in results.items() if k!='candidate-null-input-invariance'}
        telemetry=next(x for x in (json.loads(s) for s in (S/'full-telemetry.jsonl').read_text().splitlines()) if x['kind']=='replan' and x['frame_index']==32)
        assert sha256_array(native['Full'])==telemetry['environment_action_chunk_sha256']
        np.savez_compressed(out/'sealed_native_actions.npz',**native)
        seal=dict(schema='warm.original-q8-explicit-null-action-seal.v1',checkpoint_sha256=binding['checkpoint_sha256'],original_source_sha256=source_sha,
            query=8,frame=32,recovery_binding_sha256=sha(R/'recovery_binding.json'),Full_actions_and_wrapper_source_fields_exact=True,
            Full_null_Gaussian_exact=True,withdrawn_generator_candidate_invariance_exact=True,final_Full_actions_and_source_exact=True,
            model_parameters_and_buffers_unchanged=True,explicit_intervention=True,natural_withdrawal_established=False,
            policy_inferences=4,physical_branches=0,synthetic_variation_has_no_physical_branch=True,
            original_hidden_source_cache_unavailable=True,before_any_new_null_truth=True,
            sealed_native_actions_sha256=sha(out/'sealed_native_actions.npz'),controller_source_sha256=admission['controller_source_sha256'])
        (out/'sealed_null.json').write_text(json.dumps(seal,indent=2));state.update(status='complete_original_Full_null_Full_action_qualification',explicit_null_qualified=True,seal_sha256=sha(out/'sealed_null.json'));save()
    except BaseException as error:
        state.update(status='failed_or_interrupted_original_Full_null_generation',error=repr(error),traceback=traceback.format_exc());save();raise
    finally:
        if policy is not None:policy.close()
        assert identity.source_digest(C)==source_sha
        state['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()


if __name__=='__main__':main()
