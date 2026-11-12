import numpy as np
import pytest
from fastwam.research.pb_namespace import first_admitted,namespace_bounds

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
