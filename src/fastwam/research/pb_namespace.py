"""Bind supplementary PB sources to expert-admitted official seed namespaces.

A protocol row is (episode ordinal, minimum candidate seed), never two seeds.
The already recorded physical-seed-zero diagnostics keep their old identities.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

CORE_SHA256='4a1d05cb548f16bf6a9b239082b02d4697aa8e8c1f43c2f5d8ffbaec196462fd'
WEIGHT_SHA256='a765d987b2df0d228782980592c39aaed0e837040c7f6f48c905e4641f79c285'
NAMESPACE_SHA256='ee29037b74c4ee65358ccbab91549e1aa0e5e05e75226b646cf07b4aca0be221'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def array_sha(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode('ascii')+b'\0'+json.dumps(list(value.shape),separators=(',',':')).encode('ascii')+b'\0'+value.tobytes()).hexdigest()

def observation_identity(observation):
    values={key:np.asarray(value['rgb']) for key,value in observation['observation'].items() if 'rgb' in value}
    values['native_qpos']=np.asarray(observation['joint_action']['vector'])
    if values['native_qpos'].shape!=(14,) or not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError('Nonfinite or malformed PB initial observation')
    return {key:array_sha(value) for key,value in values.items()}

def namespace_bounds(values,root_seed=3407):
    values=np.asarray(values)
    if type(root_seed) is not int or root_seed!=3407:raise ValueError('Pinned PB root seed required')
    if values.shape!=(100,2) or values.dtype!=np.int64:raise ValueError('Official namespace must be int64 [100,2]')
    ordinal=np.arange(100,dtype=np.int64);bounds=100000*(1+root_seed)+ordinal
    if not np.array_equal(values[:,0],ordinal) or not np.array_equal(values[:,1],bounds):
        raise ValueError('Episode ordinals or namespace lower bounds differ')
    return bounds.copy()

def first_admitted(records,values,count=2):
    bounds=namespace_bounds(values)
    if count!=2 or type(count) is not int:raise ValueError('Exactly the first two expert-admitted sources are frozen')
    accepted=[]
    for index,row in enumerate(records):
        if len(accepted)==count:raise ValueError('Admission must stop at the second accepted source')
        if row.get('seed')!=int(bounds[0])+index or type(row.get('seed')) is not int:
            raise ValueError('Candidate seeds must follow the original namespace without gaps')
        if row.get('learned_policy_queries')!=0 or type(row.get('learned_policy_queries')) is not int:
            raise ValueError('Expert admission cannot depend on learned-policy outcomes')
        if row.get('status') not in ('accepted','expert_rejected','unstable_seed'):
            raise ValueError('Runtime failures are unknown and cannot be skipped as setup rejections')
        if row['status']=='accepted':
            accepted.append(dict(case_id=len(accepted),seed=row['seed'],namespace_lower_bound=int(bounds[len(accepted)])))
    if len(accepted)!=2:raise ValueError('Both prospectively fixed admitted sources are required')
    return accepted

def validate_registration(plan):
    expected=dict(schema='warm.original-PB-namespace2-registration.v1',root_seed=3407,case_ids=[0,1],queries=[8,48,96],
        checkpoint_step=12000,checkpoint_sha256=WEIGHT_SHA256,original_source_sha256=CORE_SHA256,
        namespace_file_sha256=NAMESPACE_SHA256,horizon=32,execution_prefix=4,top_k=32,
        original_diagnostic_seed0_preserved=True,original_windows_unchanged=True,policy_outcome_based_selection=False,
        roster_GPU_hours_max=.5,per_source_GPU_hours_max=10,additional_GPU_hours_max=20.5,lifetime_GPU_hours_cap=144,
        original_phase_GPU_hours_cap=120,prior_reserved_GPU_hours=119.8054115725,max_concurrent_GPUs=6)
    if any(type(plan.get(key)) is not type(value) or plan.get(key)!=value for key,value in expected.items()):
        raise ValueError('New cohort, model, measurements or resource registration differ')
    if plan['prior_reserved_GPU_hours']+plan['additional_GPU_hours_max']>plan['lifetime_GPU_hours_cap']:
        raise ValueError('Cumulative resource reservation exceeds the new bounded phase')
    if sha(plan['namespace_file'])!=NAMESPACE_SHA256:raise ValueError('Original namespace file changed')
    namespace_bounds(np.load(plan['namespace_file'],allow_pickle=False))
    return plan

def bind_case(binding):
    reference=binding['namespace2_registration']
    if sha(reference['path'])!=reference['sha256']:raise ValueError('Prospective namespace registration changed')
    registration=validate_registration(read(reference['path']))
    reference=binding['namespace2_roster']
    if sha(reference['path'])!=reference['sha256']:raise ValueError('Frozen expert roster changed')
    roster=read(reference['path']);case=binding['namespace_case_id']
    if type(case) is not int or case not in (0,1):raise ValueError('Only the first two admitted case IDs are allowed')
    if roster['status']!='qualified_original_PB_namespace_roster' or roster['learned_policy_queries']!=0 or roster['namespace_file_sha256']!=NAMESPACE_SHA256 or roster['registration_sha256']!=binding['namespace2_registration']['sha256']:
        raise ValueError('Not a closed expert-only roster')
    seeds=first_admitted(roster['admission_records'],np.load(registration['namespace_file'],allow_pickle=False))
    entries=roster['entries'];entry=entries[case]
    if any(entry[key]!=seeds[case][key] for key in seeds[case]) or binding['physical_seed']!=entry['seed']:
        raise ValueError('Fixed source case or admitted physical seed differs')
    return registration,entry


SCHEMAS=dict(qualification='warm.original-PB-namespace2-qualification.v1',
    factual='warm.original-specialist-factual-trace.namespace2-PB.v1',
    pool='warm.original-PB-namespace2-fresh-pool.v1')


def validate_execution_binding(plan,kind):
    """Resolve a case only from its pre-policy roster and private native paths."""
    if plan.get('schema')!=SCHEMAS[kind]:raise ValueError('Different namespace execution schema')
    registration,entry=bind_case(plan)
    private=Path(registration['phase_root'])/f"case{entry['case_id']:02d}"
    if Path(plan['rmbench_runtime'])!=private/'runtime' or Path(plan['native_worker_lease'])!=private/'native.worker.lock':
        raise ValueError('Parallel cases must own separate native runtimes and leases')
    leaf={'qualification':'qualification','factual':'factual','pool':f"pool-q{plan.get('query_replan',-1):04d}"}[kind]
    if Path(plan['output'])!=private/leaf:raise ValueError('Case output escaped its prospectively assigned directory')
    if kind=='pool':
        reference=plan['namespace2_factual_plan']
        if sha(reference['path'])!=reference['sha256'] or reference['sha256']!=plan['factual_source_plan_sha256']:
            raise ValueError('Original factual plan changed')
        factual=read(reference['path']);validate_execution_binding(factual,'factual')
        for key in ['namespace2_registration','namespace2_roster','namespace_case_id','physical_seed','rmbench_runtime','native_worker_lease']:
            if factual[key]!=plan[key]:raise ValueError('Pool and factual case bindings differ')
        if Path(plan['factual_source'])!=Path(factual['output']):raise ValueError('Pool consumes a different factual source')
    return registration,entry


def require_initial(plan,observation,instruction,physical,task_state):
    _,entry=bind_case(plan);expected=entry['initial_identity']
    actual=dict(observation_arrays=observation_identity(observation),instruction=instruction,
                physical_initial=physical,task_state_audit_only=task_state)
    if actual!=expected:raise ValueError('Learned initial state differs from the two expert-roster reset observations')
    return dict(namespace_case_id=entry['case_id'],namespace_roster_sha256=plan['namespace2_roster']['sha256'],
                namespace_registration_sha256=plan['namespace2_registration']['sha256'],
                original_namespace_initial_identity_exact=True)


def query_id(plan):
    if plan.get('schema')==SCHEMAS['pool']:
        return f"original-pb12000-ns2-case{plan['namespace_case_id']}-seed{plan['physical_seed']}-q{plan['query_replan']}"
    return f"original-pb12000-seed0-q{plan['query_replan']}"
