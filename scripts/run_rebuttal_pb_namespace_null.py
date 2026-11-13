"""Reproduce two frozen original PB q8 contexts, then execute Full/null/Full.

This independent allocation does not modify/adopt the running K32 pools. It
reconstructs actual past inputs by exact live policy replay, not a fabricated
historical online capability. All rejected and unknown endpoints are retained.
"""
from pathlib import Path
from contextlib import contextmanager
from types import MethodType
import argparse, dataclasses, datetime, gc, hashlib, importlib, importlib.util
import json, os, pickle, random, shutil, signal, subprocess, sys, tempfile, time, traceback

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


BIND = load('PB_namespace_null_bind', ROOT/'src/fastwam/research/pb_namespace_null.py')
B = load('PB_namespace_null_labels', ROOT/'src/fastwam/research/pb_mechanism.py')
ARCH = load('PB_namespace_null_archive', ROOT/'scripts/run_rebuttal_original_PB_stage.py')
sha, rd = BIND.sha, BIND.read
now = lambda: datetime.datetime.now(datetime.timezone.utc)


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)); temp.replace(path)


def run_case(plan, case_id, admission):
    row = plan['cases'][case_id]; out = Path(row['output']); source = Path(row['source'])
    assert row['case_id'] == case_id and not out.exists()
    import fcntl
    lease = Path(row['lease']).open('a'); fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out.mkdir()
    end = datetime.datetime.fromisoformat(admission['absolute_deadline_utc'])
    state = dict(status='binding_original_q8_reproduction', started_utc=now().isoformat(),
                 case_id=case_id, physical_seed=row['physical_seed'], query=8, frame=32,
                 source_sha256=BIND.CORE, checkpoint_sha256=BIND.WEIGHT,
                 original_DEV_selected_step='unverified', original_accuracy_reproduction=False,
                 live_reproduction_queries=0, extra_policy_queries=0, reproduced_native_commands=0,
                 new_branch_commands=0, model_updates=0, natural_absence_certified=False,
                 success_rate_eligible=False, branches={})
    def save():
        state['updated_utc']=now().isoformat(); write(out/'operator.json',state)
    def budget():
        if now() >= end: raise TimeoutError('Original null-stage deadline; never reset')
        if shutil.disk_usage(out).free < plan['runtime_free_space_guard_bytes']:
            raise OSError('Preserve concurrent raw evidence: AFS guard')
    save(); policy=None; env=None; observer=None; old_git=None; teacher=None
    try:
        original = rd(BIND.reference(row['factual_plan']))
        C=Path(original['original_source']); H=Path(original['reference_controller']); runtime=Path(row['runtime'])
        identity=load('PB_null_source_identity',H/'scripts/run_rebuttal_training_with_runtime_trace.py')
        setup=load('PB_null_native_setup',H/'scripts/qualify_rebuttal_rmbench_task.py')
        assert identity.source_digest(C)==BIND.CORE and setup.source_identity(runtime)['sha256']==BIND.SIM
        assert identity.source_digest(H)=='7a0ce8943818b02a7ee493fc3b6d666ba6b55b5362df7ab5f3a6be4c760000ae'
        assert sha(C/'src/fastwam/models/warm/retrospection_model.py')=='d89796cc5da67d431d911e5f9191c5d60a14599bd3763cc64198e1384dc0b67d'
        args=rd(source/'runtime_args.json'); args['warm_telemetry_path']=str(out/'replay-telemetry.jsonl')
        asset=rd(Path(original['evidence_root'])/'resumable_rb_cases0003_0009_v1_plan.json')
        os.environ.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            DIFFSYNTH_SKIP_DOWNLOAD='true',DIFFSYNTH_MODEL_BASE_PATH=asset['model_root'],
            WARM_IMMUTABLE_ASSET_HASH_CACHE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
            TOKENIZERS_PARALLELISM='false',TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0')
        os.environ.pop('NVIDIA_TF32_OVERRIDE',None)
        sys.path[:0]=[str(C/'src'),str(C),asset['curobo_source']]
        import numpy as np, torch
        import fastwam
        fastwam.__path__.append(str(ROOT/'src/fastwam'))
        import fastwam.research
        fastwam.research.__path__.insert(0,str(ROOT/'src/fastwam/research'))
        import experiments.rmbench.warm_policy.deploy_policy as deployment
        from experiments.rmbench.warm_policy.policy_core import factual_cameras, factual_joint_state
        from fastwam.memory.manifest import sha256_array, sha256_canonical_json
        from fastwam.models.wan22.fastwam import FastWAM
        from fastwam.models.warm.retrospection_model import _mean_effect
        from fastwam.research.factual_effect_reader import FrozenFactualEffectReader
        from fastwam.research.fresh_branches import OfficialQposGuard
        from fastwam.research.branches import InvalidBranchExecution
        from fastwam.research.native_replay import native_parent, state_hashes, require_observation
        from fastwam.research.fresh_pool import seal_branch
        withdraw=load('PB_null_original_gate',ROOT/'src/fastwam/research/legacy_gate_intervention.py').legacy_candidate_withdrawal
        assert Path(deployment.__file__).resolve().is_relative_to(C)
        torch.set_num_threads(4); torch.set_float32_matmul_precision('highest'); torch.cuda.set_per_process_memory_fraction(.48)
        if torch.cuda.mem_get_info()[0] < 34*1024**3: raise OSError('Original model free-GPU guard')
        contract=rd(args['warm_online_contract_path']); assert contract['warm_checkpoint_sha256']==BIND.WEIGHT
        assert sha256_canonical_json(deployment.current_encoder_runtime(args['warm_dino_device']))==contract['encoder_runtime_sha256']
        write(out/'actual_encoder_runtime.json',deployment.current_encoder_runtime(args['warm_dino_device']))
        trace=[json.loads(x) for x in (source/'native_trace.jsonl').read_text().splitlines()]
        source_records={int(r['frame_index']):r for r in (json.loads(x) for x in (source/'full-telemetry.jsonl').read_text().splitlines()) if r['kind']=='replan'}
        cfg=setup.resolved_config(runtime,'press_button',row['physical_seed']); cfg['now_ep_num']=case_id
        assert cfg==rd(source/'native_config.json')
        expert=rd(source/'expert_setup_metadata.json'); initial=rd(source/'initial_identity.json')
        observer_path=Path(original['native_TOPP_observer_source'])
        assert sha(observer_path)=='49888eb2854b29d6f5f384e394aaf91847dd9ebfdf9e89f6bc51fb1f8152dd4d'
        TOPP=load('PB_null_recorded_TOPP',observer_path).TOPPObserver
        old_git=deployment._git_identity
        def frozen_identity():
            assert identity.source_digest(C)==BIND.CORE
            return original['original_source_commit'],False
        deployment._git_identity=frozen_identity
        os.chdir(runtime); sys.path[:0]=[str(runtime),str(runtime/'description/utils')]
        from generate_episode_instructions import generate_episode_descriptions
        cls=getattr(importlib.import_module('envs.press_button'),'press_button')
        def seed_reset():
            random.seed(row['physical_seed']); np.random.seed(row['physical_seed'])
            torch.manual_seed(row['physical_seed']); torch.cuda.manual_seed_all(row['physical_seed'])
        def measured():
            return [dict(name=a.name,qpos=a.get_qpos().tolist(),qvel=a.get_qvel().tolist()) for a in env.scene.get_all_articulations()]
        def source_obs(path,obs):
            with np.load(path,allow_pickle=False) as z:
                require_observation({k:z[k] for k in z.files},{**factual_cameras(obs),'native_qpos':factual_joint_state(obs)})
        def instruction():
            text=str(np.random.choice(generate_episode_descriptions('press_button',[expert['info']],100)[0]['unseen']))
            assert text==initial['instruction']; env.set_instruction(instruction=text)
        def compare_native(i,before,after,p0,p1,s0,s1,errors):
            r=trace[i]
            assert np.array_equal(before,np.asarray(r['before'],dtype=before.dtype)) and np.array_equal(after,np.asarray(r['after'],dtype=after.dtype))
            assert p0==r['physical_before'] and p1==r['physical_after'] and s0==r['task_state_before'] and s1==r['task_state_after']
            assert errors==r['observed_TOPP_errors'] and int(env.take_action_cnt)==i+1
        # Match the original collector's pre-loader setup, then fresh learned reset.
        env=cls(); seed_reset(); env.setup_demo(**cfg); assert env.step_lim==1500; env.close_env()
        state['status']='loading_original_policy_for_exact_prefix_reproduction';save();budget()
        policy=deployment.RMBenchWarmPolicy(args)
        assert policy.model._warm_loaded_checkpoint_sha256==BIND.WEIGHT
        seed_reset(); env.setup_demo(**cfg); instruction(); source_obs(source/'initial_observation.npz',env.get_obs())
        BIND.namespace_module().require_initial(original,env.get_obs(),initial['instruction'],measured(),B.observe(env)['native_state'])
        model=policy.model; old_infer=model.infer_action; query_now=0; sealed={}; full_witness={}
        def cpu(x):
            if torch.is_tensor(x):return x.detach().cpu().clone()
            if isinstance(x,np.ndarray):return torch.from_numpy(np.array(x,copy=True))
            if dataclasses.is_dataclass(x):return {f.name:cpu(getattr(x,f.name)) for f in dataclasses.fields(x)}
            if isinstance(x,dict):return {str(k):cpu(v) for k,v in x.items()}
            if isinstance(x,(list,tuple)):return [cpu(v) for v in x]
            if isinstance(x,(str,int,float,bool,type(None))):return x
            raise TypeError('Incomplete context serialization is forbidden: '+str(type(x)))
        @contextmanager
        def capture_source():
            assert '_resolve_action_source' not in model.__dict__
            original_resolve=model._resolve_action_source; cache={}
            def observe(_self,*a,**kw):
                assert not cache and kw['phase']=='infer'
                cache['context']=kw['action_source_context']
                value=original_resolve(*a,**kw)
                for key in ['source','base_gaussian','conditioning_tokens','source_gate']:cache[key]=cpu(getattr(value,key))
                return value
            model._resolve_action_source=MethodType(observe,model)
            try:yield cache
            finally:del model._resolve_action_source
        def infer_observer(*a,**kw):
            budget();q=query_now
            if q==8:
                with capture_source() as witness:value=old_infer(*a,**kw)
            else:value=old_infer(*a,**kw)
            action=value['action'].detach().float().cpu().numpy().reshape(32,14).copy()
            expected=np.load(source/f'generations/q{q:04d}/model_actions.npy',allow_pickle=False)
            assert action.dtype==expected.dtype and np.array_equal(action,expected),'Full factual action differs before null'
            state['live_reproduction_queries']+=1
            write(out/f'replay-q{q:04d}.json',dict(original_H32_exact=True,model_action_sha256=sha256_array(action)))
            if q==8:
                step=kw['online_step']; oldkw=torch.load(source/'probes/q0008/infer_kwargs.pt',map_location='cpu',weights_only=True)
                for key in ['context_key_sha256','candidate_payload_sha256','model_input_sha256','proprio_sha256','prompt_sha256','derived_seed']:
                    assert getattr(step,key)==oldkw['online_step'][key],key
                assert dataclasses.asdict(step.query_id)==oldkw['online_step']['query_id']
                ctx=witness['context']; payload=BIND.inference_context_payload(ctx)
                torch.save(payload,out/'actual_q8_source_context.pt')
                np.savez_compressed(out/'actual_q8_history.npz',**{k:np.array(kw[k],copy=True) for k in ['episode_tokens','episode_mask','episode_action_summaries','episode_action_mask']})
                np.savez_compressed(out/'actual_q8_inputs.npz',model_input=np.array(step.model_input,copy=True),proprio=np.array(step.proprio,copy=True))
                sampler={k:kw[k] for k in ['negative_prompt','text_cfg_scale','num_inference_steps','sigma_shift','rand_device','tiled']}
                full_witness.update({k:v for k,v in witness.items() if k!='context'})
                torch.save(full_witness,out/'original_Full_source_witness.pt')
                versions=[(n,p,int(p._version)) for n,p in list(model.named_parameters())+list(model.named_buffers())]
                def generate():
                    budget()
                    y=FastWAM.infer_action(model,prompt=step.prompt,input_image=torch.tensor(step.model_input,device=model.device,dtype=model.torch_dtype),
                        proprio=torch.tensor(step.proprio,device=model.device,dtype=model.torch_dtype),action_horizon=32,
                        seed=int(step.derived_seed),action_source_context=ctx,memory_sigma=model._require_retrospection().source_sigma_min,**sampler)
                    state['extra_policy_queries']+=1
                    return y['action'].detach().float().cpu().numpy().reshape(32,14).copy()
                with torch.inference_mode():
                    with capture_source() as check:a_full=generate()
                    assert np.array_equal(a_full,action) and all(torch.equal(check[k],full_witness[k]) for k in full_witness)
                    with withdraw(model) as null_witness:a_null=generate()
                    assert torch.equal(null_witness['base_gaussian'],full_witness['base_gaussian'])
                    torch.save(cpu(null_witness),out/'candidate_null_source_witness.pt')
                    with capture_source() as final:a_final=generate()
                    assert np.array_equal(a_final,action) and all(torch.equal(final[k],full_witness[k]) for k in full_witness)
                after=dict(list(model.named_parameters())+list(model.named_buffers()))
                assert set(after)=={n for n,p,v in versions} and all(after[n] is p and int(p._version)==v for n,p,v in versions)
                sealed.update({'Full-pre':np.asarray(policy.action_normalizer.denormalize(a_full),dtype=np.float32),
                               'candidate-null':np.asarray(policy.action_normalizer.denormalize(a_null),dtype=np.float32),
                               'Full-post':np.asarray(policy.action_normalizer.denormalize(a_final),dtype=np.float32)})
                assert sha256_array(sealed['Full-pre'])==source_records[32]['environment_action_chunk_sha256']
                np.savez_compressed(out/'sealed_native_actions.npz',**sealed)
                write(out/'action_seal.json',dict(schema='warm.PB-namespace2-q8-null-action-seal.v1',case_id=case_id,physical_seed=row['physical_seed'],query=8,
                    source_sha256=BIND.CORE,checkpoint_sha256=BIND.WEIGHT,source_factual_plan=row['factual_plan'],
                    Full_original_and_final_actions_exact=True,same_context_and_Gaussian=True,explicit_null_only=True,
                    factual_recollection_retained=True,natural_absence_certified=False,model_parameters_and_buffers_unchanged=True,
                    before_any_new_null_truth=True,actions_sha256=sha(out/'sealed_native_actions.npz'),
                    action_array_sha256={k:sha256_array(v) for k,v in sealed.items()},source_context_sha256=sha(out/'actual_q8_source_context.pt')))
            save();return value
        model.infer_action=infer_observer
        policy._next_episode_index=case_id;policy.reset()
        state['status']='reproducing_all_original_queries_and_first36_commands';save()
        observer=TOPP(env);observer.__enter__()
        with (out/'reproduced_native_prefix.jsonl').open('x',buffering=1) as stream:
            for i in range(36):
                budget();obs=env.get_obs();query_now=i//4
                if not policy.queue:
                    source_obs(source/f'replan-{query_now:04d}.npz',obs);policy._replan(obs,env)
                    records=[json.loads(x) for x in (out/'replay-telemetry.jsonl').read_text().splitlines()]
                    recent=[x for x in records if x['kind']=='replan'][-1]
                    assert recent['history_before_replan']==source_records[i]['history_before_replan']
                before=factual_joint_state(obs).copy();p0=measured();s0=B.observe(env)['native_state'];observer.errors.clear()
                policy.step(env,obs);after=factual_joint_state(env.get_obs()).copy();p1=measured();s1=B.observe(env)['native_state']
                compare_native(i,before,after,p0,p1,s0,s1,observer.errors)
                stream.write(json.dumps(dict(index=i,original_native_step_exact=True,physical_before=p0,physical_after=p1,task_before=s0,task_after=s1,TOPP_errors=observer.errors))+'\n')
                state['reproduced_native_commands']+=1;save()
        observer.__exit__(None,None,None);observer=None
        assert state['live_reproduction_queries']==9 and state['extra_policy_queries']==3 and len(sealed)==3
        env.close_env(clear_cache=True);env=None;policy.close();policy=None;gc.collect();torch.cuda.empty_cache()
        state['status']='action_seal_qualified_no_null_truth_yet';save();budget()
        torch.cuda.set_per_process_memory_fraction(.12)
        teacher=FrozenFactualEffectReader.from_source_runtime(args,contract);write(out/'reader_receipt.json',teacher.receipt)
        rng=pickle.loads((source/'probes/q0008/sampling_rng_before.pkl').read_bytes())
        commands=np.asarray([r['command'] for r in trace[:32]],dtype=np.float32)
        def reset_parent(label):
            nonlocal env
            budget();seed_reset();env=cls();env.setup_demo(**cfg);instruction();source_obs(source/'initial_observation.npz',env.get_obs())
            observed=TOPP(env)
            with observed:
                for i,command in enumerate(commands):
                    budget();before=factual_joint_state(env.get_obs()).copy();p0=measured();s0=B.observe(env)['native_state'];observed.errors.clear()
                    env.take_action(command.copy(),action_type='qpos')
                    compare_native(i,before,factual_joint_state(env.get_obs()),p0,measured(),s0,B.observe(env)['native_state'],observed.errors)
                    if (i+1)%4==0:source_obs(source/f'replan-{(i+1)//4:04d}.npz',env.get_obs())
            source_obs(source/'replan-0008.npz',env.get_obs())
            random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch_cpu']);torch.cuda.set_rng_state_all(rng['torch_cuda'])
            parent=native_parent(env,env.get_obs());ph=state_hashes(parent)
            write(out/(label+'_prefix_qualification.json'),dict(commands=32,all_original_TOPP_and_physics_exact=True,parent_hashes=ph))
            return parent,ph
        parents=[]
        for i in range(2):
            _,ph=reset_parent(f'qualification-{i}');parents.append(ph);env.close_env(clear_cache=True);env=None
        assert parents[0]==parents[1]
        write(out/'two_prefixes_qualified.json',dict(two_exact_prefixes_before_null_truth=True,parent_hashes=parents[0]))
        rows={}
        for role in ['Full-pre','candidate-null','Full-post']:
            budget();attempt=out/'branches'/role/'attempt-001';attempt.mkdir(parents=True,exist_ok=False)
            parent,ph=reset_parent(role);assert ph==parents[0]
            (attempt/'parent.pickle').write_bytes(pickle.dumps(parent,protocol=4))
            pre=B.observe(env);before=teacher.encode(env.get_obs());np.save(attempt/'parent_dino_spatial.npy',before,allow_pickle=False)
            assert state_hashes(native_parent(env,env.get_obs()))==ph
            n=0;status='complete_horizon';error=None;prefix4=None
            state.update(status='executing_sealed_q8_withdrawal_branch',active_branch=role);save()
            with (attempt/'commands.jsonl').open('x',buffering=1) as stream,OfficialQposGuard(env) as guard:
                for i,command in enumerate(sealed[role]):
                    budget();p0=measured();s0=B.observe(env);qpos=factual_joint_state(env.get_obs()).copy()
                    try:accepted=guard.step(command)
                    except InvalidBranchExecution as exc:status='invalid_branch';error=str(exc);break
                    if not accepted:status='invalid_branch';error='official executor stopped';break
                    n+=1;p1=measured();s1=B.observe(env)
                    if role.startswith('Full') and i<4:
                        compare_native(32+i,qpos,factual_joint_state(env.get_obs()),p0,p1,s0['native_state'],s1['native_state'],[])
                    stream.write(json.dumps(dict(index=i,command=command.tolist(),physical_before=p0,physical_after=p1,task_before=s0,task_after=s1))+'\n')
                    state['new_branch_commands']+=1;save()
                    if i==3:prefix4=dict(progress=s1,label=B.label(pre,s1,valid_branch=True))
                    if env.eval_success and n<32:status='early_success';break
            endpoint=native_parent(env,env.get_obs());eh=state_hashes(endpoint);after=teacher.encode(env.get_obs());post=B.observe(env)
            assert state_hashes(native_parent(env,env.get_obs()))==eh
            effect=_mean_effect(torch.from_numpy(after-before)[None])[0].numpy() if status=='complete_horizon' else None
            result=dict(query_id=f'original-pb12000-ns2-null-case{case_id}-seed{row["physical_seed"]}-q8',branch_id=role,
                parent_recreated=True,parent_hashes=ph,endpoint_hashes=eh,endpoint_status=status,executed_steps=n,error=error,
                parent_progress=pre,endpoint_progress=post,observed_effect=effect.tolist() if effect is not None else None,
                secondary_prefix4=prefix4,independent_label=B.label(pre,post,valid_branch=status!='invalid_branch'),
                sealed_query_sha256=sha(out/'action_seal.json'))
            (attempt/'endpoint.pickle').write_bytes(pickle.dumps(endpoint,protocol=4));np.save(attempt/'endpoint_dino_spatial.npy',after,allow_pickle=False)
            seal_branch(attempt,result);rows[role]=result;state['branches'][role]=result;save()
            env.close_env(clear_cache=True);env=None
        keys=['parent_hashes','endpoint_hashes','endpoint_status','executed_steps','error','observed_effect','parent_progress','endpoint_progress','secondary_prefix4','independent_label']
        assert all(rows['Full-pre'][key]==rows['Full-post'][key] for key in keys),'Repeated Full physical endpoints differ'
        state.update(status='complete_two_sided_Full_null_Full_fixed_q8',Full_physical_repeat_exact=True,
                     Full_H32_effect_available=rows['Full-pre']['endpoint_status']=='complete_horizon',
                     null_H32_effect_available=rows['candidate-null']['endpoint_status']=='complete_horizon',
                     population_repair_rate_available=False,final_task_attribution_available=False,
                     source_sha256_after=identity.source_digest(C),ended_utc=now().isoformat())
        assert state['source_sha256_after']==BIND.CORE
    except BaseException as exc:
        state.update(status='failed_or_interrupted_q8_null_preserve_all',error=repr(exc),traceback=traceback.format_exc(),ended_utc=now().isoformat());save();raise
    finally:
        if observer is not None:observer.__exit__(None,None,None)
        if policy is not None:policy.close()
        if env is not None:env.close_env(clear_cache=True)
        if old_git is not None:deployment._git_identity=old_git
        save()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--expected-plan-sha256',required=True)
    ap.add_argument('--validate-only',action='store_true');ap.add_argument('--case',type=int,choices=[0,1]);ap.add_argument('--admission',type=Path);ap.add_argument('--expected-admission-sha256')
    cli=ap.parse_args();assert sha(cli.plan)==cli.expected_plan_sha256;p=BIND.validate_plan(rd(cli.plan))
    if cli.validate_only:
        print(json.dumps(dict(status='namespace_q8_null_CPU_input_preflight',policy_forwards=0,physical_commands=0,cases=[0,1])));return
    if cli.case is not None:
        assert cli.admission and sha(cli.admission)==cli.expected_admission_sha256
        admission=rd(cli.admission);assert admission['plan_sha256']==cli.expected_plan_sha256 and admission['case_id']==cli.case
        start=datetime.datetime.fromisoformat(admission['first_admitted_utc']);end=datetime.datetime.fromisoformat(admission['absolute_deadline_utc'])
        assert start<=now()<end and 0<(end-start).total_seconds()<=p['maximum_seconds']
        assert end<=datetime.datetime.fromisoformat(p['latest_end_utc'])
        run_case(p,cli.case,admission);return
    import fcntl
    root=Path(p['root']);lock=(root/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    op=root/'operator.json';assert not op.exists(),'Unknown dispatch is not an automatic retry'
    assert shutil.disk_usage(root).free>=p['minimum_admission_free_bytes']
    first=now();end=BIND.deadline(first,datetime.datetime.fromisoformat(p['latest_end_utc']))
    state=dict(status='admitted_two_fixed_q8_withdrawal_cases',first_admitted_utc=first.isoformat(),absolute_deadline_utc=end.isoformat(),
               plan_sha256=cli.expected_plan_sha256,case_ids=[0,1],cases={},automatic_restarts=0,original_windows_unchanged=True)
    child=None
    def save():state['updated_utc']=now().isoformat();write(op,state)
    save()
    try:
        cache=Path(tempfile.mkdtemp(prefix='warm-pb-ns2-null-'));os.environ.update(TMPDIR=str(cache),TORCH_EXTENSIONS_DIR=str(cache/'torch'),TRITON_CACHE_DIR=str(cache/'triton'),CUDA_CACHE_PATH=str(cache/'cuda'))
        for case in [0,1]:
            if (end-now()).total_seconds()<1980:
                state['cases'][str(case)]=dict(status='not_started_original_window_insufficient');save();continue
            deadline=BIND.case_deadline(now(),end)
            admission=dict(case_id=case,first_admitted_utc=first.isoformat(),absolute_deadline_utc=deadline.isoformat(),plan_sha256=cli.expected_plan_sha256)
            af=root/f'case{case:02d}.admission.json';assert not af.exists();write(af,admission)
            argv=[p['python'],'-B',str(Path(__file__).resolve()),'--plan',str(cli.plan),'--expected-plan-sha256',cli.expected_plan_sha256,
                  '--case',str(case),'--admission',str(af),'--expected-admission-sha256',sha(af)]
            with (root/f'case{case:02d}.console.log').open('x') as stream:
                child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status=f'running_fixed_case{case}',actual_child_pid=child.pid);save();bounded=False
                while child.poll() is None:
                    if now()>=deadline or shutil.disk_usage(root).free<p['runtime_free_space_guard_bytes']:
                        bounded=True;os.killpg(child.pid,signal.SIGINT)
                        try:child.wait(timeout=120)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        break
                    save();time.sleep(5)
                result=dict(exit_code=child.returncode,bounded_termination=bounded,ended_utc=now().isoformat())
                child=None
            d=Path(p['cases'][case]['output'])
            if d.is_dir():result['archive']=ARCH.archive(d,root/'archives',f'case{case:02d}')
            state['cases'][str(case)]=result;save()
        state.update(status='terminal_fixed_q8_withdrawal_all_existing_output_archived',ended_utc=now().isoformat());save()
    except BaseException as exc:
        state.update(status='failed_or_interrupted_preserve_all',error=repr(exc),traceback=traceback.format_exc(),ended_utc=now().isoformat());save();raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:child.wait(timeout=120)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


if __name__=='__main__':main()
