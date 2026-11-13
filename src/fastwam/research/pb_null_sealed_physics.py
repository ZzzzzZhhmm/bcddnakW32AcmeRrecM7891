"""Strict known-OOM handoff of original sealed Full/null actions, without resampling."""
from pathlib import Path
import datetime,hashlib,json

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def reference(r):
    p=Path(r['path'])
    if sha(p)!=r['sha256']:raise ValueError('Sealed continuation input changed: '+str(p))
    return p

def validate_known_failure(op,runner,case):
    expected=dict(status='failed_or_interrupted_q8_null_preserve_all',case_id=case,
                  physical_seed=340800000+case,query=8,frame=32,live_reproduction_queries=9,
                  extra_policy_queries=3,reproduced_native_commands=36,new_branch_commands=0,
                  model_updates=0,branches={})
    if any(op.get(k)!=v for k,v in expected.items()):raise ValueError('Require complete sealed generation and zero physical intervention')
    if not str(op.get('error','')).startswith("OutOfMemoryError(") or 'FrozenFactualEffectReader.from_source_runtime' not in op.get('traceback',''):
        raise ValueError('Only the known post-seal effect-reader allocation error is eligible')
    if runner.get('status')!='failed' or runner.get('exit_code')!=1:raise ValueError('Previous process must be known terminal')
    return True

def validate(spec,case,now):
    if spec.get('schema')=='warm.PB-null-unstarted-physical-roles.v1':return validate_unstarted_roles(spec,case,now)
    if spec.get('schema')!='warm.PB-null-sealed-physical-continuation.v1' or case not in [0,1]:raise ValueError('Fixed continuation schema/case required')
    first=datetime.datetime.fromisoformat(spec['original_first_admitted_utc']);end=datetime.datetime.fromisoformat(spec['original_end_utc'])
    if first.isoformat()!='2026-11-13T07:47:40.544934+00:00' or end!=datetime.datetime(2026,11,13,11,tzinfo=datetime.timezone.utc):raise ValueError('Do not reset the original window')
    if now.tzinfo is None or not first<now<end-datetime.timedelta(seconds=180):raise ValueError('Original active window expired')
    if spec.get('new_GPU_hours_reserved')!=0 or spec.get('policy_generations')!=0:raise ValueError('Physical-only original allocation required')
    row=spec['cases'][case]
    if row['case_id']!=case:raise ValueError('No case substitution')
    validate_known_failure(read(reference(row['operator'])),read(reference(row['runner'])),case)
    for r in row['files'].values():reference(r)
    check_output_readiness(spec['root'],case,row['output'])
    if not read(reference(row['platform_terminal'])).get('known_terminal'):raise ValueError('Previous cloud owner must be terminal')
    return row

def load_sealed_actions(recovery,case,original):
    import numpy as np,torch
    from fastwam.memory.manifest import sha256_array
    files=recovery['files'];seal=read(reference(files['action_seal.json']))
    wanted=dict(schema='warm.PB-namespace2-q8-null-action-seal.v1',case_id=case,physical_seed=340800000+case,query=8,
                Full_original_and_final_actions_exact=True,same_context_and_Gaussian=True,explicit_null_only=True,
                factual_recollection_retained=True,model_parameters_and_buffers_unchanged=True,before_any_new_null_truth=True)
    if any(seal.get(k)!=v for k,v in wanted.items()):raise ValueError('Incomplete original action seal')
    if seal['source_sha256']!='4a1d05cb548f16bf6a9b239082b02d4697aa8e8c1f43c2f5d8ffbaec196462fd' or seal['checkpoint_sha256']!='a765d987b2df0d228782980592c39aaed0e837040c7f6f48c905e4641f79c285':raise ValueError('Original policy identity changed')
    if seal['source_factual_plan']!=original['factual_plan']:raise ValueError('Original factual source differs')
    actions=reference(files['sealed_native_actions.npz'])
    if sha(actions)!=seal['actions_sha256'] or sha(reference(files['actual_q8_source_context.pt']))!=seal['source_context_sha256']:raise ValueError('Action/context seal differs')
    with np.load(actions,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
    if set(values)!={'Full-pre','candidate-null','Full-post'}:raise ValueError('Three original roles required')
    for key,value in values.items():
        if value.shape!=(32,14) or value.dtype!=np.float32 or not np.isfinite(value).all() or sha256_array(value)!=seal['action_array_sha256'][key]:raise ValueError('Original action bytes differ')
    if not np.array_equal(values['Full-pre'],values['Full-post']):raise ValueError('Original two-sided Full mismatch')
    full=torch.load(reference(files['original_Full_source_witness.pt']),map_location='cpu',weights_only=True)
    null=torch.load(reference(files['candidate_null_source_witness.pt']),map_location='cpu',weights_only=True)
    if not torch.equal(full['base_gaussian'],null['base_gaussian']):raise ValueError('Sampling randomness differs')
    return values


SEALED_INPUT_NAMES=(
    'actual_q8_source_context.pt','actual_q8_history.npz','actual_q8_inputs.npz',
    'original_Full_source_witness.pt','candidate_null_source_witness.pt',
    'sealed_native_actions.npz','action_seal.json')

def check_output_readiness(root,case,output):
    root=Path(root);out=Path(output)
    if out!=root/f'case{case:02d}'/'results' or not out.parent.is_dir() or out.exists():
        raise ValueError('Prepare the fixed case parent and retain a fresh absent results directory')

def materialize_sealed_inputs(recovery,output):
    out=Path(output);result={}
    if any((out/name).exists() for name in SEALED_INPUT_NAMES):raise ValueError('Never overwrite sealed inputs')
    for name in SEALED_INPUT_NAMES:
        source=reference(recovery['files'][name]);data=source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=recovery['files'][name]['sha256']:raise ValueError('Input changed while copying')
        destination=out/name
        with destination.open('xb') as stream:stream.write(data)
        if sha(destination)!=recovery['files'][name]['sha256']:raise ValueError('Copied seal differs')
        result[name]=dict(original=recovery['files'][name],local_sha256=sha(destination))
    return result


def physical_roles(recovery):
    roles=recovery.get('physical_roles',['Full-pre','candidate-null','Full-post']) if recovery else ['Full-pre','candidate-null','Full-post']
    if roles not in [['Full-pre','candidate-null','Full-post'],['candidate-null','Full-post']]:raise ValueError('Keep original order and never repeat the unknown Full-pre')
    return roles

def validate_unstarted_failure(op,runner,case):
    required=dict(status='failed_or_interrupted_q8_null_preserve_all',case_id=case,
                  mode='sealed_action_physics_only_continuation',new_policy_loaded=False,
                  live_reproduction_queries=0,extra_policy_queries=0,model_updates=0,
                  active_branch='Full-pre',branches={})
    if any(op.get(k)!=v for k,v in required.items()):raise ValueError('Require only the known first-Full seal-writer failure')
    if not 1<=op.get('new_branch_commands',0)<=32 or not str(op.get('error','')).startswith('FileNotFoundError(') or 'action_seal.json' not in op.get('traceback',''):raise ValueError('Unknown or different physical execution failure')
    if runner.get('status')!='failed' or runner.get('exit_code')!=1:raise ValueError('Old writer process must be known terminal')

def validate_unstarted_roles(spec,case,now):
    if case not in [0,1] or spec.get('case_ids')!=[0,1]:raise ValueError('Both original cases required')
    old_end=datetime.datetime.fromisoformat(spec['original_end_utc']);first=datetime.datetime.fromisoformat(spec['phase_first_admitted_utc']);end=datetime.datetime.fromisoformat(spec['phase_hard_end_utc']);active=datetime.datetime.fromisoformat(spec['active_end_utc'])
    if old_end!=datetime.datetime(2026,11,13,11,tzinfo=datetime.timezone.utc) or first<old_end or not 0<(end-first).total_seconds()<=720 or active!=end-datetime.timedelta(seconds=90):raise ValueError('Old window stays closed;new phase must be prospectively bounded')
    if now.tzinfo is None or not first<=now<active:raise ValueError('New bounded phase expired')
    required=dict(policy_generations=0,new_external_GPU_hours_reserved=0,reused_closed_GPU_hours=.4,cumulative_reserved_GPU_hours=143.8054115725,lifetime_cap_GPU_hours=144,source_clock_reset=False,original_Full_pre_unknown_preserved=True)
    if any(spec.get(k)!=v for k,v in required.items()):raise ValueError('Reuse only certified closed balance;no unknown outcome replacement')
    credit=read(reference(spec['resource_transfer']))
    if credit.get('credit_id')!='PB_null_known_writer_closed_balance_20261113_v1' or credit.get('GPU_hours')!=.4 or credit.get('used_for')!=spec['root'] or not credit.get('all_prior_allocations_terminal'):raise ValueError('One explicit closed-budget transfer required')
    row=spec['cases'][case]
    if row['case_id']!=case or row.get('physical_roles')!=['candidate-null','Full-post']:raise ValueError('Only two previously unstarted roles may execute')
    validate_unstarted_failure(read(reference(row['operator'])),read(reference(row['runner'])),case)
    if not read(reference(row['platform_terminal'])).get('known_terminal'):raise ValueError('Previous cloud owner must be terminal')
    prior=Path(row['operator']['path']).parent
    if any((prior/'branches'/name/'attempt-001').exists() for name in ['candidate-null','Full-post']):raise ValueError('Never repeat started or unknown roles')
    for r in row['files'].values():reference(r)
    check_output_readiness(spec['root'],case,row['output'])
    return row

def full_control_summary(rows):
    if 'Full-pre' in rows:
        keys=['parent_hashes','endpoint_hashes','endpoint_status','executed_steps','error','observed_effect','parent_progress','endpoint_progress','secondary_prefix4','independent_label']
        if not all(rows['Full-pre'][key]==rows['Full-post'][key] for key in keys):raise ValueError('Repeated Full physical endpoints differ')
        return rows['Full-pre'],True
    if set(rows)!={'candidate-null','Full-post'}:raise ValueError('Unstarted role pair incomplete')
    return rows['Full-post'],False


def reader_memory_fraction(policy_was_loaded):
    """Do not shrink a process limit below still-referenced policy allocations.

    Closing the online wrapper is not a guarantee that local model references
    have been freed. Keep its original allowance in the combined process;
    a physical-only fresh process retains the smaller reader-only allowance.
    """
    if type(policy_was_loaded) is not bool:raise ValueError('Explicit policy lifetime state required')
    return .48 if policy_was_loaded else .12
