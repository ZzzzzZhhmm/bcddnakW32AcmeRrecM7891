"""Admission and descriptive pairing for a separate native development pilot.

Family removal is an availability intervention. It never supplies relevance
labels or changes the registered formal non-inferiority experiment.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json

CONDITIONS=('clean_size_matched','target_skill_family_holdout')
ROLES=('full','candidate_null')
PHASE='native_family_holdout_development_pilot'
INFERENCE=dict(horizon=32,execution_prefix=4,nfe=10,top_k=32,recent_capacity=20,summary_capacity=8)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())


def snapshot_bank_rows(value):
    """Copy the actual NumPy online-step bank rows before evidence serialization."""
    import numpy as np
    rows=np.asarray(value)
    if rows.shape!=(32,) or rows.dtype.kind not in 'iu' or np.any(rows<0):
        raise ValueError('Expected the actual K32 nonnegative integer bank-row vector')
    return rows.copy()

def validate(plan):
    if plan.get('schema')!='warm.rebuttal.native-family-dev-pilot.v1' or plan.get('phase')!=PHASE:
        raise ValueError('Separate development-pilot identity required')
    if plan.get('task')!='rearrange_blocks' or plan.get('official_step_limit')!=700 or plan.get('inference')!=INFERENCE:
        raise ValueError('Fixed native RB and deployed inference budget required')
    if plan.get('registered_pilot_seeds')!=[3407,3408] or plan.get('admitted_case_ids') not in ([0],[1]):
        raise ValueError('Only the two registered development seeds,one bounded stage at a time')
    if plan.get('conditions')!=list(CONDITIONS) or plan.get('intervention_roles')!=list(ROLES):
        raise ValueError('Both predeclared nonempty universes and both roles required')
    if plan.get('root_seed')!=3407 or plan.get('formal_TEST') is not False or plan.get('primary_NI_protocol_replaced') is not False:
        raise ValueError('Pilot cannot replace formal performance or NI evidence')
    if set(plan.get('roles',{}))!={'full'}:raise ValueError('Exactly one frozen Full checkpoint is shared by both interventions')
    b=plan['roles']['full']
    if (b.get('model_class'),b.get('checkpoint_schema'),b.get('checkpoint_update'))!=('WarmRetrospectionFastWAM','warm.retrospection-checkpoint',3000):
        raise ValueError('Fixed final3000 Full factory and schema required for both roles')
    if plan.get('candidate_null_checkpoint_sha256')!=b.get('checkpoint_sha256') or not b.get('checkpoint_sha256'):
        raise ValueError('Null is the same model,not a separately trained baseline')
    if set(plan.get('universe_rows',{}))!=set(CONDITIONS):raise ValueError('Both search-universe files required')
    if any(r.get('rows')!=19540 for r in plan['universe_rows'].values()):raise ValueError('Frozen equal nonempty universe sizes required')
    if plan.get('worker_max_seconds')!=2400 or plan.get('stage_max_seconds')!=10800:
        raise ValueError('Original bounded CCI stage and worker limits required')
    return plan

def require_window(plan,now=None):
    validate(plan);now=now or datetime.now(timezone.utc)
    first=datetime.fromisoformat(plan['first_admitted_utc']);end=datetime.fromisoformat(plan['absolute_deadline_utc'])
    if any(x.tzinfo is None for x in [first,end,now]) or (end-first).total_seconds()!=10800:
        raise ValueError('One original three-hour stage window required')
    if now<first:raise ValueError('Admission has not begun')
    if now>=end:raise TimeoutError('Original pilot stage expired;never reset its window')
    return end.timestamp()

def requests(plan):
    validate(plan);out=[]
    for case in plan['admitted_case_ids']:
        roles=ROLES if case%2==0 else tuple(reversed(ROLES))
        conditions=CONDITIONS if case%2==0 else tuple(reversed(CONDITIONS))
        out.extend((case,c,r) for c in conditions for r in roles)
    return out

def pair(left,right):
    if (left.get('role'),right.get('role'))!=ROLES:raise ValueError('Canonical Full/null role order required')
    for k in ['schema','phase','plan_sha256','task','episode_id','seed','search_condition','checkpoint_sha256','initial_identity','initial_measured_physical','common_contract']:
        if left.get(k)!=right.get(k):raise ValueError('Paired initial state or binding differs: '+k)
    if left.get('phase')!=PHASE:raise ValueError('Not a development-pilot pair')
    if left.get('search_condition') not in CONDITIONS:raise ValueError('Unknown search intervention')
    for r in [left,right]:
        if r.get('status')!='complete_native' or type(r.get('success')) is not int or r['success'] not in (0,1):raise ValueError('Complete native outcomes required;unknown is not failure')
        if not r['success'] and r.get('executed_commands')!=700:raise ValueError('A failed pilot must reach the native cap')
        if not r.get('queries'):raise ValueError('Actual policy query evidence required')
    for q in right['queries']:
        if (q.get('declared_force_null') is not True or q.get('effective_gate')!=0 or q.get('source_gate')!=0
            or q.get('actual_source')!=q.get('base_gaussian')):raise ValueError('Actual candidate-null boundary was not verified')
    for k in ['derived_seed','base_gaussian','model_input_sha256','proprio_sha256','context_key_sha256']:
        if left['queries'][0].get(k)!=right['queries'][0].get(k):raise ValueError('Initial input/Gaussian differs: '+k)
    return dict(episode_id=left['episode_id'],seed=left['seed'],condition=left['search_condition'],
        full_success=left['success'],null_success=right['success'],full_minus_null=left['success']-right['success'],
        scope='Prespecified single development seed per stage;descriptive only,not formal100pairs,absence certification or NI.',
        independently_measured_missing_skill_coverage=None,false_activation_rate=None,noninferiority_claimed=False)
