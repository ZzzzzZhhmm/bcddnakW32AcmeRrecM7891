"""Recover the original BT q8 factual input through a fresh owned retrieval.

This reads existing observations and exact reconstructed past history. It never
loads a policy, predicts an action, constructs a historical capability, or runs
a simulator. The original immutable retriever validates its own fresh step.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,dataclasses,datetime,hashlib,importlib.util,json,os,signal,sys,traceback


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence-root',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--preflight-only',action='store_true')
    ap.add_argument('--admission',type=Path)
    ap.add_argument('--expected-admission-sha256')
    cli=ap.parse_args();P=cli.evidence_root.resolve(strict=True);out=cli.output.resolve()
    if out.exists() or out.parent!=P:raise ValueError('fresh Git-external evidence output required')
    read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    C=P/'historical_specialist_source_f77c633/code';S=P/'original_BT12000_cohort_case0_native1000_v1';H=P/'ITER30_ORIGINAL_Q8_HISTORY_CPU_V1'
    expected='ff32342d286929ddc5904d55814c66e965269f6b749c1e1b6df1060aaef0b7fe'
    spec=importlib.util.spec_from_file_location('identity',P/'controller_eb70b41/code/scripts/run_rebuttal_training_with_runtime_trace.py')
    identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
    assert identity.source_digest(C)==expected
    history_proof=read(H/'operator.json')
    assert history_proof['status']=='complete_original_q8_history_reconstruction_CPU_only' and history_proof['all32_past_action_hashes_exact']
    assert sha(H/'recovered_history.npz')==history_proof['recovered_history_sha256']
    os.environ.update(TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        DIFFSYNTH_SKIP_DOWNLOAD='true',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',WARM_IMMUTABLE_ASSET_HASH_CACHE='1')
    os.environ.pop('NVIDIA_TF32_OVERRIDE',None)
    if cli.preflight_only:os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path[:0]=[str(C/'src'),str(C)]
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fastwam.memory.manifest import sha256_array,sha256_canonical_json
    from fastwam.memory.online_retrieval import FrozenDinoOnlineRetriever
    from fastwam.memory.processor_contract import load_m1_data_config,extract_m1_robotwin_processor_recipe,validate_processor_instance
    from fastwam.memory.runtime_fingerprint import current_encoder_runtime
    from fastwam.models.warm.online_contract import WarmOnlineRunContract
    from fastwam.models.warm.source_contract import WarmSourceRunContract
    from fastwam.models.warm.retrospection_config import WarmRetrospectionConfig
    from fastwam.models.warm.retrospection_model import WarmRetrospectionFastWAM
    import experiments.rmbench.warm_policy.deploy_policy as deployment
    args=read(S/'runtime_args.json');contract=WarmOnlineRunContract.from_dict(read(Path(args['warm_online_contract_path'])))
    assert args['task_name']=='battery_try' and args['seed']==3407 and contract.root_seed==3407
    assert contract.warm_checkpoint_sha256=='27c137d689007b38f6204d3dd12bb038550b1e58403506d7a9894bd6fcd43a9f'
    cfg=deployment._compose_runtime_config(args,checkpoint=Path(args['ckpt_setting']),stats=Path(args['dataset_stats_path']),task_name='battery_try',task_id=contract.task_id)
    projection=read(P/'original_BT12000_online_bundle_s3407_v3/runtime_projections/full_warm/battery_try.json')
    rconfig=WarmRetrospectionConfig.from_dict(projection['model_behavior']['retrospection'])
    assert (rconfig.context_dim,rconfig.semantic_dim,rconfig.action_dim,rconfig.proprio_dim,rconfig.timing_dim)==(777,768,14,14,4)
    stats,stats_sha=deployment._stable_stats(Path(args['dataset_stats_path']));assert stats_sha==contract.normalization_stats_sha256
    assert args['mixed_precision']=='bf16'
    processor=instantiate(cfg.data.train.processor).eval();processor.set_normalizer_from_stats(stats)
    recipe,digest=load_m1_data_config(args['warm_m1_data_config_path'],profile='robotwin')
    assert digest==contract.m1_data_config_sha256 and extract_m1_robotwin_processor_recipe(OmegaConf.to_container(cfg,resolve=True))==recipe
    validate_processor_instance(processor,recipe)
    old=torch.load(S/'probes/q0008/infer_kwargs.pt',map_location='cpu',weights_only=True)
    original=old['online_step'];assert original['query_id']==history_proof['query_id']
    with np.load(S/'replan-0008.npz',allow_pickle=False) as arrays:
        cameras={k:arrays[k].copy() for k in arrays.files if k!='native_qpos'};qpos=arrays['native_qpos'].copy()
    proprio=deployment._normalize_state(qpos,processor).numpy()
    assert sha256_array(proprio)==original['proprio_sha256']
    prompt=original['prompt']
    source_inputs=[S/'runtime_args.json',S/'probes/q0008/infer_kwargs.pt',S/'probes/q0008/model_output.pt',S/'replan-0008.npz',H/'recovered_history.npz',H/'operator.json']
    input_manifest={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in source_inputs}
    now=lambda:datetime.datetime.now(datetime.timezone.utc)
    end=None;interrupted=False
    def stop(*_):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    if not cli.preflight_only:
        if cli.admission is None or sha(cli.admission)!=cli.expected_admission_sha256:raise ValueError('authenticated bounded retrieval admission required')
        admission=read(cli.admission);start=datetime.datetime.fromisoformat(admission['first_admitted_utc']);end=datetime.datetime.fromisoformat(admission['absolute_deadline_utc'])
        assert admission['schema']=='warm.original-q8-retrieval-reconstruction.v1' and 0<(end-start).total_seconds()<=1200 and start<=now()<end
        assert admission['policy_inferences']==admission['physical_commands']==0 and admission['maximum_retrievals']==1
        assert Path(admission['output']).resolve()==out and admission['history_receipt_sha256']==sha(H/'operator.json')
        assert admission['entry_file_sha256']==sha(Path(__file__))
    out.mkdir();state=dict(status='original_inputs_static_qualified',started_utc=now().isoformat(),query=8,frame=32,
        history_arrays_exactly_reconstructed=True,normalized_proprio_exact=True,policy_inferences=0,physical_commands=0,
        policy_checkpoint_loaded=False,retriever_created=False,historical_BoundOnlineStep_constructed=False,
        full_factual_context_recovered=False,real_checkpoint_GPU_replay_qualified=False)
    def save():
        state['updated_utc']=now().isoformat();(out/'operator.json').write_text(json.dumps(state,indent=2))
    save()
    def budget():
        if interrupted or now()>=end:raise TimeoutError('original reconstruction admission expired;never reset')
        free=os.statvfs(out).f_bavail*os.statvfs(out).f_frsize
        if free<4*1024**3:raise OSError('preserve concurrent evidence:storage guard')
        if torch.cuda.mem_get_info()[0]<8*1024**3:raise OSError('protect active policy:free GPU guard')
    try:
        if cli.preflight_only:
            assert not torch.cuda.is_initialized();state.update(status='complete_CPU_original_query_static_preflight',CUDA_initialized=False);(out/'inputs_manifest.json').write_text(json.dumps(input_manifest,indent=2));save();print(json.dumps(state));return
        (out/'admission_snapshot.json').write_bytes(cli.admission.read_bytes())
        torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.cuda.set_per_process_memory_fraction(.08)
        budget();runtime=current_encoder_runtime(args['warm_dino_device'])
        assert sha256_canonical_json(runtime)==contract.encoder_runtime_sha256,'Actual encoder runtime differs;no contract editing'
        (out/'actual_encoder_runtime.json').write_text(json.dumps(runtime,indent=2))
        source_contract=WarmSourceRunContract.from_dict(read(Path(args['warm_training_run_contract_path'])))
        retriever=FrozenDinoOnlineRetriever.from_artifacts(args['warm_bank_directory'],source_run_contract=source_contract,online_run_contract=contract,
            normalizer_contract_path=args['warm_normalizer_contract_path'],encoder_contract_path=args['warm_encoder_contract_path'],
            camera_contract_path=args['warm_camera_contract_path'],normalization_stats_path=args['dataset_stats_path'],
            catalog_path=args['warm_catalog_path'],audit_report_path=args['warm_audit_report_path'],dino_checkpoint_path=args['warm_dino_checkpoint_path'],
            processor=processor,dino_device=args['warm_dino_device'],dino_torch_dtype=deployment._dino_dtype(Path(args['warm_encoder_contract_path']),expected_sha256=contract.encoder_contract_sha256),
            dino_batch_size=args['warm_dino_batch_size'],benchmark_profile='robotwin')
        state.update(status='fresh_owned_retrieval_ready',retriever_created=True);save();budget()
        retriever.begin_episode(0);qid=retriever.make_query_id(32);assert dataclasses.asdict(qid)==original['query_id']
        step=retriever.retrieve(qid,cameras,task_description='battery_try',prompt=prompt,proprio=proprio)
        for key in ['step_sha256','context_key_sha256','candidate_payload_sha256','model_input_sha256','proprio_sha256','prompt_sha256','derived_seed']:
            assert getattr(step,key)==original[key],key
        telemetry=torch.load(S/'probes/q0008/model_output.pt',map_location='cpu',weights_only=True)['warm_online_telemetry']['retrieval']
        assert step.bank_rows.tolist()==telemetry['bank_rows'] and step.candidate_valid_mask.tolist()==telemetry['candidate_valid_mask']
        # This consumes only the fresh step created above, through its public
        # original ownership validator. No historical capability is rebuilt.
        facts=retriever.validate_and_gather_candidate_facts(step,prompt=step.prompt,proprio=step.proprio,input_image=step.model_input)
        with np.load(H/'recovered_history.npz',allow_pickle=False) as arrays:history={k:arrays[k].copy() for k in arrays.files}
        # The original conversion routine contains no model computation. Use its
        # validated original dimensions and dtype without allocating a policy.
        conversion=SimpleNamespace(device='cpu',torch_dtype=torch.bfloat16,_require_retrospection=lambda:rconfig)
        ctx=WarmRetrospectionFastWAM._context_from_online_facts(conversion,online_step=step,facts=facts,**history)
        payload={f.name:getattr(ctx,f.name).detach().cpu().clone() for f in dataclasses.fields(ctx) if torch.is_tensor(getattr(ctx,f.name))}
        forbidden={'target_action','target_effect','current_semantic_teacher','target_action_valid_mask','future_valid_mask'}
        assert not forbidden.intersection(payload)
        assert all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in payload.values())
        torch.save(payload,out/'recovered_source_context.pt')
        np.savez_compressed(out/'recovered_query_inputs.npz',model_input=np.array(step.model_input,copy=True),proprio=np.array(step.proprio,copy=True),context_key=np.array(step.context_key,copy=True))
        binding=dict(schema='warm.original-q8-authenticated-context-recovery.v1',query=8,frame=32,
            original_source_sha256=expected,checkpoint_sha256=contract.warm_checkpoint_sha256,
            original_online_step_sha256=step.step_sha256,original_model_input_sha256=step.model_input_sha256,
            original_proprio_sha256=step.proprio_sha256,original_context_key_sha256=step.context_key_sha256,
            original_candidate_payload_sha256=step.candidate_payload_sha256,derived_seed=int(step.derived_seed),prompt=step.prompt,
            history_receipt_sha256=sha(H/'operator.json'),history_arrays_sha256=history_proof['history_arrays'],
            original_facts_restored=True,historical_capability_forged=False,new_owned_step_consumed_by_original_validator=True,
            source_context_sha256=sha(out/'recovered_source_context.pt'),query_inputs_sha256=sha(out/'recovered_query_inputs.npz'),
            sampler={k:old[k] for k in ['negative_prompt','text_cfg_scale','num_inference_steps','sigma_shift','rand_device','tiled']},
            policy_inferences=0,physical_commands=0,exact_Full_GPU_replay_qualified=False)
        (out/'recovery_binding.json').write_text(json.dumps(binding,indent=2))
        budget();assert identity.source_digest(C)==expected
        assert all(sha(Path(n))==v['sha256'] for n,v in input_manifest.items())
        (out/'inputs_manifest.json').write_text(json.dumps(input_manifest,indent=2))
        assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<32*1024**2
        assert torch.cuda.max_memory_allocated()<6*1024**3
        state.update(status='complete_original_q8_authenticated_factual_context_recovery',full_factual_context_recovered=True,
            original_step_and_all_input_payload_hashes_exact=True,original_source_unchanged=True,
            recovery_binding_sha256=sha(out/'recovery_binding.json'),peak_allocated_bytes=torch.cuda.max_memory_allocated(),ended_utc=now().isoformat())
        save();print(json.dumps(state))
    except BaseException as error:
        state.update(status='failed_or_interrupted_factual_context_recovery',error=repr(error),traceback=traceback.format_exc());save();raise


if __name__=='__main__':main()
