"""Typed, outcome-independent admission and receipts for Full/pooled evaluation.

This module never fetches Git, constructs a simulator or changes model settings.
Recorded DEV engineering and fixed-final native policy trials are distinct.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROLES=('full','pooled_memory_prior')
ROLE_TYPES={
    'full':('sim_rmbench.yaml','WarmRetrospectionFastWAM','warm.retrospection-checkpoint'),
    'pooled_memory_prior':('sim_rmbench_pooled_memory.yaml','PooledMemoryPriorFastWAM','warm.research.pooled-event-prior-with-recollection.v1'),
}
DESIGN_SHA256='3484ad1477a868607a916638c631c80df25501db59dd82edbf6651c768238e33'
INFERENCE=dict(horizon=32,execution_prefix=4,nfe=10,top_k=32,recent_capacity=20,summary_capacity=8)
COMMON_CONTRACT_FIELDS=(
    'training_run_contract_sha256','validation_run_contract_sha256','bank_manifest_sha256','bank_content_sha256',
    'encoder_contract_sha256','encoder_runtime_sha256','camera_contract_sha256','m1_data_config_sha256',
    'dino_checkpoint_tree_sha256','normalization_stats_sha256','action_space_contract_sha256','catalog_sha256',
    'audit_sha256','vae_checkpoint_sha256','text_encoder_tree_sha256','tokenizer_tree_sha256',
    'evaluation_namespace_sha256','task_suite','task_id','task_description','root_seed',
    'initial_states_sha256','bddl_sha256','retrieval_implementation','top_k','source_policy',
    'memory_sigma','action_horizon','action_dim',
)

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic_json(path,value):
    import os,uuid
    path=Path(path);temp=path.with_name(path.name+'.atomic-'+uuid.uuid4().hex)
    with temp.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)

def require_deadline(plan,now=None):
    now=now or datetime.now(timezone.utc)
    first=datetime.fromisoformat(plan['first_admitted_utc']);end=datetime.fromisoformat(plan['absolute_deadline_utc'])
    if first.tzinfo is None or end.tzinfo is None or now.tzinfo is None:raise ValueError('Timezone-aware resource window required')
    ceiling=5400 if plan['phase']=='recorded_dev' else 36*3600
    if not 0<(end-first).total_seconds()<=ceiling:raise ValueError('Persistent resource window exceeds the declared phase')
    if now<first:raise ValueError('Admission precedes the frozen phase')
    if now>=end:raise TimeoutError('Original resource window expired;do not reset it')
    if plan['phase']=='formal' and end>datetime(2026,11,12,18,tzinfo=timezone.utc):raise ValueError('Formal phase exceeds the fixed campaign ceiling')
    return end.timestamp()

def validate_plan(plan):
    if plan.get('schema')!='warm.rebuttal.matched-native.v1' or plan.get('phase') not in ('recorded_dev','formal'):
        raise ValueError('Unknown matched-native protocol')
    if plan.get('design_sha256')!=DESIGN_SHA256 or plan.get('inference')!=INFERENCE or plan.get('root_seed')!=3407:
        raise ValueError('Fixed scientific design or inference budget differs')
    if set(plan.get('roles',{}))!=set(ROLES):raise ValueError('Independent Full and pooled bindings required')
    if plan['roles']['full']['checkpoint_sha256']==plan['roles']['pooled_memory_prior']['checkpoint_sha256']:
        raise ValueError('Full and pooled must have independent typed checkpoints')
    for role in ROLES:
        binding=plan['roles'][role]
        if (binding.get('config_name'),binding.get('model_class'),binding.get('checkpoint_schema'))!=ROLE_TYPES[role]:
            raise ValueError('Role factory/schema differs: '+role)
        expected_update=3000 if plan['phase']=='formal' else (200 if role=='full' else 100)
        if type(binding.get('checkpoint_update')) is not int or binding['checkpoint_update']!=expected_update:
            raise ValueError('Role checkpoint update differs from the fixed phase')
    if plan['phase']=='recorded_dev':
        if (plan.get('task')!='press_button' or plan.get('query_frames')!=list(range(0,40,4))
            or plan.get('recorded_episode')!=435 or plan.get('episode_id')!=0):
            raise ValueError('Earliest fixed PB DEV prefixes required')
    else:
        if plan.get('task') not in ('rearrange_blocks','battery_try'):raise ValueError('Unplanned formal task')
        expected_limit={'rearrange_blocks':700,'battery_try':1000}[plan['task']]
        if plan.get('official_step_limit')!=expected_limit or plan.get('cohort_case_ids')!=list(range(50)):
            raise ValueError('Full native task and fixed fifty-pair cohort required')
        if len(plan.get('case_ids',[]))!=10 or plan['case_ids']!=list(range(min(plan['case_ids']),min(plan['case_ids'])+10)) or min(plan['case_ids']) not in (0,10,20,30,40):
            raise ValueError('Exactly one predetermined ten-case shard required')
        if not 0<plan.get('shard_max_seconds',0)<=43200 or not 0<plan.get('worker_max_seconds',0)<=plan['shard_max_seconds']:
            raise ValueError('Bounded shard and worker windows required')
        if plan.get('native_concurrency_qualified') is not True:raise ValueError('Per-worker native writable-path qualification is pending')
    return plan

def validate_training_attestation(binding,attestation):
    if (attestation.get('schema')!='warm.training-attestation'
        or attestation.get('actual_global_step')!=binding['checkpoint_update']
        or attestation.get('checkpoint_step')!=binding['checkpoint_update']
        or attestation.get('checkpoint_sha256')!=binding['checkpoint_sha256']
        or attestation.get('root_seed')!=3407):
        raise ValueError('Checkpoint lineage/update differs from its explicit role')

def verify_formal_prerequisites(plan,training,dev):
    """Engineering integration never substitutes for complete final training."""
    if plan['phase']!='formal':raise ValueError('Formal prerequisites requested for a different phase')
    strict=dev.get('status')=='complete_actual_dual_role_DEV_matching'
    causal=(dev.get('schema')=='warm.rebuttal.matched-external-input-qualification.v1'
        and dev.get('status')=='qualified_current_inputs_noise_and_causal_retrieval_feedback'
        and plan.get('DEV_qualification_scope')==dev.get('qualification_scope')=='matched_external_inputs_with_causal_selection_feedback'
        and plan.get('DEV_scope_amendment',{}).get('sha256')==dev.get('scope_amendment_sha256')
        and dev.get('scope_amendment_sha256') is not None
        and dev.get('current_input_proprio_and_query_keys_exact') is True
        and dev.get('initial_candidate_pool_exact') is True
        and dev.get('all_later_retrieval_differences_causally_reconstructed') is True
        and dev.get('common_top31_at_differing_queries_exact') is True
        and dev.get('strict_all_query_candidate_equality_passed') is False
        and bool(dev.get('raw_evidence_archive_sha256')) and bool(dev.get('independent_reconstruction_proof_sha256')))
    if (not (strict or causal) or dev.get('evaluation_source_sha256')!=plan['evaluation_source_sha256']
        or dev.get('actual_noise_pairs_verified')!=10 or dev.get('simulator_loaded') is not False
        or dev.get('generated_commands_executed')!=0 or dev.get('role_types_verified') is not True):
        raise ValueError('Actual two-role DEV input/noise/history qualification is required')
    for role in ROLES:
        row=training[role];binding=plan['roles'][role]
        if (row.get('schema')!='warm.rebuttal.final-training-evidence.v1'
            or row.get('status')!='verified_complete_final_state_and_checkpoint'
            or row.get('role')!=role or row.get('actual_global_step')!=3000
            or row.get('checkpoint_sha256')!=binding['checkpoint_sha256']
            or row.get('training_source_sha256')!=binding['training_source_sha256']
            or row.get('all_state_files_verified') is not True
            or row.get('byte_exact_state_archive_verified') is not True
            or not row.get('raw_state_manifest_sha256') or not row.get('byte_exact_state_archive_sha256')):
            raise ValueError('Complete byte-exact final training evidence required: '+role)

def verify_inputs(plan):
    validate_plan(plan)
    for item in plan['inputs']:
        if file_sha(item['path'])!=item['sha256']:raise ValueError('Frozen input changed: '+item['path'])
    for role in ROLES:
        b=plan['roles'][role]
        if file_sha(b['bundle_args'])!=b['bundle_args_sha256']:raise ValueError('Typed bundle arguments changed')
        if file_sha(b['training_attestation'])!=b['training_attestation_sha256']:raise ValueError('Training attestation changed')
        validate_training_attestation(b,read_json(b['training_attestation']))
    if plan['phase']=='formal':
        training={}
        for role in ROLES:
            ref=plan['roles'][role]['complete_training_state_receipt']
            if file_sha(ref['path'])!=ref['sha256']:raise ValueError('Final training state receipt changed')
            training[role]=read_json(ref['path'])
        ref=plan['dual_role_DEV_qualification']
        if file_sha(ref['path'])!=ref['sha256']:raise ValueError('Actual two-role DEV receipt changed')
        dev=read_json(ref['path']);verify_formal_prerequisites(plan,training,dev)
        if dev.get('qualification_scope')=='matched_external_inputs_with_causal_selection_feedback':
            scope=plan['DEV_scope_amendment']
            if file_sha(scope['path'])!=scope['sha256']:raise ValueError('Explicit pre-TEST DEV scope amendment changed')
            amendment=read_json(scope['path'])
            if (amendment.get('pre_TEST') is not True or amendment.get('unchanged_scientific_design_sha256')!=DESIGN_SHA256
                or amendment.get('scientific_primary_metrics_samples_weights_algorithms_or_deadlines_changed') is not False
                or datetime.fromisoformat(amendment['created_utc'])>datetime.fromisoformat(plan['first_admitted_utc'])):
                raise ValueError('DEV scope must be declared before the original formal admission')
            proof_ref=plan['DEV_reconstruction_proof']
            if file_sha(proof_ref['path'])!=proof_ref['sha256'] or proof_ref['sha256']!=dev['independent_reconstruction_proof_sha256']:
                raise ValueError('Independent causal retrieval reconstruction proof differs')
            proof=read_json(proof_ref['path'])
            if (proof.get('all10_current_input_proprio_query_key_and_Gaussian_pairs_exact') is not True
                or proof.get('initial_K32_exact') is not True or proof.get('strict_all_query_pair_gate_passed') is not False
                or proof.get('archive_sha256')!=dev['raw_evidence_archive_sha256']
                or proof.get('causally_reconstructed_continuation_tail_differences')!=dev.get('continuation_tail_difference_frames')):
                raise ValueError('Observed input or retrieval difference is not fully accounted for')
        q=plan['native_concurrency_qualification']
        if file_sha(q['path'])!=q['sha256']:raise ValueError('Native concurrency qualification receipt changed')
        receipt=read_json(q['path'])
        if (receipt.get('status')!='qualified_disjoint_writable_native_runtimes_and_exact_resets'
            or receipt.get('runtime_source_sha256')!=plan['runtime_source_sha256']
            or plan['runtime_isolation'] not in receipt.get('worker_isolations',[])):
            raise ValueError('Actual disjoint native worker qualification is missing')
        if file_sha(plan['roster'])!=plan['roster_sha256']:raise ValueError('Fixed outcome-independent reset roster changed')
        roster=read_json(plan['roster']);entries=roster['entries']
        if (roster.get('status')!='qualified_first50_expert_admitted_reset_roster' or roster.get('task')!=plan['task']
            or roster.get('runtime_source_sha256')!=plan['runtime_source_sha256']
            or [r.get('episode_id') for r in entries]!=list(range(50))):raise ValueError('Exactly the planned fifty expert-admitted resets required')
        seeds=[r['seed'] for r in entries]
        if seeds!=sorted(set(seeds)) or not all(type(x) is int and 340800000<=x<340802000 for x in seeds):
            raise ValueError('Original fixed seed namespace/order differs')

def load_runtime(plan,role,output):
    """Use the already frozen common model/deployment source, never this helper."""
    if role not in ROLES:raise ValueError('Unknown policy role')
    from scripts import build_warm_rmbench_contract_bundle as builder
    from scripts.run_warm_rmbench_matrix import load_matrix,select_experiments
    core=Path(plan['evaluation_code']).resolve(strict=True)
    if Path(builder.__file__).resolve()!=core/'scripts/build_warm_rmbench_contract_bundle.py':raise RuntimeError('Wrong common core imported')
    b=plan['roles'][role];ba=builder.build_parser().parse_args(read_json(b['bundle_args']))
    if ba.task!=[plan['task']] or ba.experiment!=['full_warm'] or ba.sim_config.name!=b['config_name']:
        raise ValueError('Role was passed through a foreign native constructor')
    if ba.evaluation_namespace!=plan['evaluation_namespace']:raise ValueError('Role-independent paired sampling namespace required')
    builder.validate_contract_bundle(ba.output_root,experiment_id='full_warm',task_names=[plan['task']])
    matrix=load_matrix(ba.matrix);experiment,=select_experiments(matrix,ba.experiment)
    if matrix.root_seed!=3407:raise ValueError('Common root seed differs')
    artifacts=builder._artifact_paths(ba)
    runtime=builder._runtime_args(ba,artifacts,experiment=experiment,task_name=plan['task'],
        contract_path=ba.output_root/'full_warm'/f'{plan["task"]}.json',
        seed_protocol_path=ba.output_root/'seeds'/f'{plan["task"]}.seed_protocol.npy',
        task_definition_path=ba.rmbench_root/'envs'/f'{plan["task"]}.py',root_seed=3407)
    fields=dict(action_horizon='horizon',replan_steps='execution_prefix',num_inference_steps='nfe',warm_top_k='top_k',
                warm_recent_event_capacity='recent_capacity',warm_action_summary_capacity='summary_capacity')
    if any(runtime[k]!=plan['inference'][v] for k,v in fields.items()):raise ValueError('Native runtime inference budget changed')
    if file_sha(artifacts['warm_checkpoint'])!=b['checkpoint_sha256']:raise ValueError('Wrong independent checkpoint artifact')
    runtime['warm_telemetry_path']=str(Path(output)/'policy_telemetry.jsonl')
    return runtime

def typed_policy_receipt(policy,plan,role):
    b=plan['roles'][role]
    if (type(policy.model).__name__!=b['model_class'] or policy.model._retrospection_state()['schema']!=b['checkpoint_schema']
        or policy.checkpoint_sha256!=b['checkpoint_sha256']):raise ValueError('Loaded native model has wrong role type/weights')
    control=getattr(policy.model,'_warm_research_probe',None)
    if control is not None and any(bool(getattr(control,k,False)) for k in ('capture','force_null')):
        raise ValueError('Matched policies must retain their production mechanism modes')
    return dict(role=role,model_class=type(policy.model).__name__,checkpoint_schema=b['checkpoint_schema'],checkpoint_update=b['checkpoint_update'],
        checkpoint_sha256=policy.checkpoint_sha256,common_contract={k:getattr(policy.contract,k) for k in COMMON_CONTRACT_FIELDS})

class PolicyOutputError(ValueError):
    """A returned model action is missing, nonfinite or has a foreign shape."""

def tensor_record(value):
    import torch
    value=value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
                finite=bool(torch.isfinite(value).all()),nonzero=int(torch.count_nonzero(value)))

@contextmanager
def capture_replan(policy,directory):
    """Seal actual H32 before validation/execution; preserve both mode and output."""
    import numpy as np
    from fastwam.research.query_capture import clone_inference_context
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    model=policy.model;infer=model.infer_action;source=model._resolve_action_source
    had_i='infer_action' in vars(model);had_s='_resolve_action_source' in vars(model)
    result=dict(infer_calls=0,source_calls=0)
    def observed_source(*args,**kwargs):
        result['source_calls']+=1
        result['context']=clone_inference_context(kwargs['action_source_context'])
        result['base_gaussian']=kwargs['base_gaussian'].detach().clone()
        original=source(*args,**kwargs)
        result['source']=original.source.detach().clone()
        result['conditioning_tokens']=None if original.conditioning_tokens is None else original.conditioning_tokens.detach().clone()
        return original
    def observed_infer(*args,**kwargs):
        result['infer_calls']+=1;result['online_step']=kwargs['online_step']
        original=infer(*args,**kwargs);value=original.get('action')
        if value is None:
            atomic_json(directory/'generation.json',dict(returned_action_missing=True,before_consumer_validation=True))
            raise PolicyOutputError('Returned action is missing')
        action=value.detach().float().cpu().numpy().copy() if hasattr(value,'detach') else np.array(value,copy=True)
        with (directory/'model_actions.npy').open('xb') as stream:np.save(stream,action,allow_pickle=False)
        metadata=dict(shape=list(action.shape),all_finite=bool(np.isfinite(action).all()),action_file_sha256=file_sha(directory/'model_actions.npy'),before_consumer_validation=True)
        atomic_json(directory/'generation.json',metadata);result['action']=action
        if action.shape not in ((32,14),(1,32,14)) or not metadata['all_finite']:raise PolicyOutputError('Returned model action must be finite H32x14')
        return original
    model.infer_action=observed_infer;model._resolve_action_source=observed_source
    try:
        yield result
        if result['infer_calls']!=1 or result['source_calls']!=1:raise RuntimeError('One native source resolution per actual replan required')
    finally:
        import sys
        error=sys.exc_info()[1]
        if had_i:model.infer_action=infer
        else:del model.infer_action
        if had_s:model._resolve_action_source=source
        else:del model._resolve_action_source
        atomic_json(directory/'consumer_status.json',dict(infer_calls=result['infer_calls'],source_calls=result['source_calls'],
            consumer_completed=error is None,error=None if error is None else repr(error),model_outputs_unmodified=True))

def replan_receipt(capture):
    context=capture['context'];step=capture['online_step']
    return dict(derived_seed=int(step.derived_seed),base_gaussian=tensor_record(capture['base_gaussian']),
        actual_source=tensor_record(capture['source']),conditioning=None if capture['conditioning_tokens'] is None else tensor_record(capture['conditioning_tokens']),
        model_input_sha256=step.model_input_sha256,proprio_sha256=step.proprio_sha256,bank_rows=step.bank_rows.tolist(),
        valid_candidates=int(step.candidate_valid_mask.sum()),candidate_payload_sha256=step.candidate_payload_sha256,
        context_key_sha256=step.context_key_sha256,history_observation_tokens=int(context.episode_mask.sum()),
        history_action_slots=int(context.episode_action_mask.sum()),mode_unmodified=True,legacy_Full_probe_enabled=False)

def invalid_action_proof(worker):
    """A failed process or a JSON flag alone is not an invalid policy action."""
    import numpy as np
    found=False
    for path in Path(worker).glob('generations/*/generation.json'):
        row=read_json(path)
        if row.get('before_consumer_validation') is not True:raise ValueError('Unsealed invalid-action evidence')
        if row.get('returned_action_missing') is True:
            found=True;continue
        action_file=path.parent/'model_actions.npy'
        if file_sha(action_file)!=row['action_file_sha256']:raise ValueError('Returned action artifact hash differs')
        action=np.load(action_file,allow_pickle=False)
        if list(action.shape)!=row['shape'] or bool(np.isfinite(action).all())!=row['all_finite']:
            raise ValueError('Returned action disagrees with its failure metadata')
        found|=tuple(action.shape) not in ((32,14),(1,32,14)) or not bool(np.isfinite(action).all())
    return found

def pair_receipts(left,right,*,recorded_dev=False):
    if left['role']!='full' or right['role']!='pooled_memory_prior':raise ValueError('Full and pooled receipts required in canonical role order')
    for key in ('plan_sha256','task','episode_id','phase','initial_identity','common_contract'):
        if left[key]!=right[key]:raise ValueError('Paired initial input or common binding differs: '+key)
    queries=left['queries'] if recorded_dev else left['queries'][:1]
    other=right['queries'] if recorded_dev else right['queries'][:1]
    if not queries or len(queries)!=len(other):raise ValueError('Actual paired sampler noise evidence unavailable')
    for a,b in zip(queries,other,strict=True):
        for key in ('frame','derived_seed','base_gaussian','model_input_sha256','proprio_sha256','bank_rows'):
            if a[key]!=b[key]:raise ValueError('Paired input/initial Gaussian differs: '+key)
    return dict(status='paired_DEV_engineering_only' if recorded_dev else 'paired_native_known_outcomes' if all(x['status'] in ('complete_native','identifiable_policy_failure') for x in (left,right)) else 'paired_native_unknown',
        actual_noise_pairs_verified=len(queries),initial_identity_exact=True,common_information_contract_exact=True,
        phase=left['phase'],episode_id=left['episode_id'],task=left['task'],
        full_minus_pooled=None if recorded_dev or left['success'] is None or right['success'] is None else left['success']-right['success'])

def next_request(records,case_ids,*,known_terminal_attempts=()):
    """Never skip unknown attempts or choose among completed results."""
    grouped={}
    for row in records:
        key=(row['episode_id'],row['role'])
        grouped.setdefault(key,[]).append(row)
    terminal=set(known_terminal_attempts)
    accepted={}
    for key,rows in grouped.items():
        known=[r for r in rows if r['status'] in ('complete_native','identifiable_policy_failure','complete_recorded_DEV')]
        if len(known)>1:raise ValueError('Duplicate role evidence;never select by outcome')
        if any(r.get('attempt_id') not in terminal for r in rows if r not in known):
            raise RuntimeError('Unknown or active attempt must be adjudicated before retry')
        if known:accepted[key]=known[0]
    for index in case_ids:
        for role in (ROLES if index%2==0 else tuple(reversed(ROLES))):
            row=accepted.get((index,role))
            if row is None:return index,role
    return None


NATIVE_PRIVATE_ENVIRONMENTS=('TMPDIR','TMP','TEMP','HOME','XDG_CACHE_HOME','TORCH_EXTENSIONS_DIR',
    'PYTORCH_KERNEL_CACHE_PATH','CUDA_CACHE_PATH','TRITON_CACHE_DIR','NUMBA_CACHE_DIR','MPLCONFIGDIR',
    '__GL_SHADER_DISK_CACHE_PATH')


def validate_native_isolations(isolations):
    """Require actual distinct private paths, not an asserted concurrency flag."""
    paths=[];ids=[]
    for item in isolations:
        root=Path(item['writable_root']).resolve();runtime=Path(item['runtime_root']).resolve(strict=True)
        if Path(item['writable_root']).is_symlink() or len(str(item['environment'].get('TMPDIR','')).encode())>100:raise ValueError('Unsafe or NVRTC-incompatible private temp root')
        if Path(item['persistent_lease_path']).resolve()!=runtime.parent/'native-worker.lock':raise ValueError('Native worker lease must remain persistent beside its runtime')
        if item.get('ipc_kind')!='anonymous_multiprocessing_Pipe':raise ValueError('Unqualified planner IPC')
        if root==runtime or root.is_relative_to(runtime) or runtime.is_relative_to(root):raise ValueError('Writable/runtime roots overlap')
        if item.get('worker_id') in ids:raise ValueError('Duplicate native worker identity')
        ids.append(item['worker_id'])
        if set(item['environment'])!=set(NATIVE_PRIVATE_ENVIRONMENTS):raise ValueError('Incomplete private native environment')
        for value in item['environment'].values():
            p=Path(value)
            if p.is_symlink() or not p.is_absolute() or not p.resolve().is_relative_to(root):
                raise ValueError('Cache/temp/home path escapes the private worker')
        if any(p==q or p.is_relative_to(q) or q.is_relative_to(p) for p in (root,runtime) for q in paths):
            raise ValueError('Concurrent worker writable/runtime paths overlap')
        paths.extend((root,runtime))
    return True


def activate_native_isolation(plan):
    import fcntl,os,sys,tempfile
    item=plan['runtime_isolation'];validate_native_isolations([item])
    runtime=Path(plan['native_runtime']).resolve(strict=True)
    if runtime!=Path(item['runtime_root']).resolve(strict=True) or read_json(runtime/'MATCHED_WORKER_ISOLATION.json')!=item:
        raise ValueError('Actual worker runtime marker differs')
    if 'torch' in sys.modules:raise RuntimeError('Private caches must be selected before importing Torch')
    # CUDA/NVRTC needs a short container-local temp name. Scientific outputs and
    # cross-node leases remain on AFS; host-local caches may be recreated.
    root=Path(item['writable_root']);root.mkdir(parents=True,exist_ok=True)
    owner=root/'WARM_NATIVE_CACHE_OWNER.json'
    if owner.exists() and read_json(owner)!=item:raise ValueError('Existing short cache belongs to another worker')
    if not owner.exists():atomic_json(owner,item)
    for value in item['environment'].values():Path(value).mkdir(parents=True,exist_ok=True)
    fd=os.open(item['persistent_lease_path'],os.O_CREAT|os.O_RDWR,0o600)
    try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:os.close(fd);raise
    os.environ.update(item['environment']);tempfile.tempdir=None
    return fd
