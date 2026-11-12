import numpy as np
import pytest
from fastwam.research.pb_namespace import first_admitted,namespace_bounds
from fastwam.research import pb_namespace as N
from pathlib import Path

def namespace():return np.stack((np.arange(100,dtype=np.int64),340800000+np.arange(100,dtype=np.int64)),axis=1)
def records():return [dict(seed=340800000,status='expert_rejected',learned_policy_queries=0),dict(seed=340800001,status='accepted',learned_policy_queries=0),dict(seed=340800002,status='accepted',learned_policy_queries=0)]

def test_second_column_is_a_candidate_bound_not_an_accepted_seed():
    values=namespace();assert values.reshape(-1)[0]==0
    assert namespace_bounds(values)[0]==340800000
    assert first_admitted(records(),values)==[dict(case_id=0,seed=340800001,namespace_lower_bound=340800000),dict(case_id=1,seed=340800002,namespace_lower_bound=340800001)]

@pytest.mark.parametrize('values',[np.arange(200,dtype=np.int64),namespace().astype(np.float32),namespace()[:2],namespace()[:,::-1]])
def test_ambiguous_or_wrong_namespace_is_rejected(values):
    with pytest.raises(ValueError):namespace_bounds(values)

@pytest.mark.parametrize('mutation',['gap','model_selection','runtime_failure','too_many','missing'])
def test_roster_cannot_hide_rejected_seeds_or_policy_results(mutation):
    rows=records()
    if mutation=='gap':rows[0]['seed']+=1
    if mutation=='model_selection':rows[1]['learned_policy_queries']=1
    if mutation=='runtime_failure':rows[0]['status']='runtime_failure'
    if mutation=='too_many':rows.append(dict(seed=340800003,status='accepted',learned_policy_queries=0))
    if mutation=='missing':rows.pop()
    with pytest.raises(ValueError):first_admitted(rows,namespace())

@pytest.mark.parametrize('root',[True,0,3408])
def test_wrong_root_seed_is_rejected(root):
    with pytest.raises(ValueError):namespace_bounds(namespace(),root)


def test_parallel_case_paths_cannot_share_runtime_or_output(monkeypatch,tmp_path):
    registration=dict(phase_root=str(tmp_path));entry=dict(case_id=1,seed=340800002)
    monkeypatch.setattr(N,'bind_case',lambda p:(registration,entry))
    private=tmp_path/'case01'
    p=dict(schema=N.SCHEMAS['factual'],rmbench_runtime=str(private/'runtime'),native_worker_lease=str(private/'native.worker.lock'),output=str(private/'factual'))
    assert N.validate_execution_binding(p,'factual')==(registration,entry)
    for key in ['rmbench_runtime','native_worker_lease','output']:
        changed=dict(p);changed[key]=str(tmp_path/'case00'/Path(p[key]).name)
        with pytest.raises(ValueError):N.validate_execution_binding(changed,'factual')


def test_initial_identity_includes_nonpolicy_hidden_state_and_every_camera(monkeypatch):
    obs=dict(observation=dict(head=dict(rgb=np.zeros((2,2,3),dtype=np.uint8)),left=dict(rgb=np.ones((2,2,3),dtype=np.uint8))),joint_action=dict(vector=np.zeros(14,dtype=np.float32)))
    identity=dict(observation_arrays=N.observation_identity(obs),instruction='fixed',physical_initial=[dict(qpos=[1],qvel=[0])],task_state_audit_only=dict(count=0))
    entry=dict(case_id=1,initial_identity=identity)
    monkeypatch.setattr(N,'bind_case',lambda p:({},entry))
    plan=dict(namespace2_roster=dict(sha256='roster'),namespace2_registration=dict(sha256='registration'))
    result=N.require_initial(plan,obs,'fixed',[dict(qpos=[1],qvel=[0])],dict(count=0))
    assert result['namespace_case_id']==1 and result['original_namespace_initial_identity_exact']
    for text,physical,state in [('other',[dict(qpos=[1],qvel=[0])],dict(count=0)),('fixed',[dict(qpos=[2],qvel=[0])],dict(count=0)),('fixed',[dict(qpos=[1],qvel=[0])],dict(count=1))]:
        with pytest.raises(ValueError):N.require_initial(plan,obs,text,physical,state)
    obs['observation']['left']['rgb'][0,0,0]=0
    with pytest.raises(ValueError):N.require_initial(plan,obs,'fixed',identity['physical_initial'],identity['task_state_audit_only'])


def test_query_identity_distinguishes_new_case_and_keeps_old_diagnostic():
    assert N.query_id(dict(schema='warm.original-PB-first-state-fresh-pool.v1',query_replan=8))=='original-pb12000-seed0-q8'
    assert N.query_id(dict(schema=N.SCHEMAS['pool'],query_replan=8,namespace_case_id=1,physical_seed=340800002))=='original-pb12000-ns2-case1-seed340800002-q8'
