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
    if spec.get('schema')!='warm.PB-null-sealed-physical-continuation.v1' or case not in [0,1]:raise ValueError('Fixed continuation schema/case required')
    first=datetime.datetime.fromisoformat(spec['original_first_admitted_utc']);end=datetime.datetime.fromisoformat(spec['original_end_utc'])
    if first.isoformat()!='2026-11-13T07:47:40.544934+00:00' or end!=datetime.datetime(2026,11,13,11,tzinfo=datetime.timezone.utc):raise ValueError('Do not reset the original window')
    if now.tzinfo is None or not first<now<end-datetime.timedelta(seconds=180):raise ValueError('Original active window expired')
    if spec.get('new_GPU_hours_reserved')!=0 or spec.get('policy_generations')!=0:raise ValueError('Physical-only original allocation required')
    row=spec['cases'][case]
    if row['case_id']!=case:raise ValueError('No case substitution')
    validate_known_failure(read(reference(row['operator'])),read(reference(row['runner'])),case)
    for r in row['files'].values():reference(r)
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
