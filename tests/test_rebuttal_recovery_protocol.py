from copy import deepcopy
import pytest
from fastwam.research.recovery_protocol import admit_formal_recovery


def setup(success=0,index=0):
    plan=dict(tasks=['t'],checkpoint_sha256='c',closed_loop=dict(reset_seed_start=100,candidate_seed_limit=20),
        local_recovery=dict(source_episodes_per_task=20,query_replan_ordinals=[0,8,24,48,80,120]),
        candidate_specific_effects=dict(source_episodes_per_task=10,query_replan_ordinals=[8,48]))
    identity=dict(task='t',phase='formal',checkpoint_sha256='c',plan_sha256='p',core_source_sha256='s',search_condition='full_bank')
    row={**identity,'episode_id':index,'seed':100+index,'roles':{'full':dict(status='complete',success=success)}}
    args=dict(task='t',seed=100+index,episode_index=index,plan_sha256='p',core_source_sha256='s',search_condition='full_bank')
    return plan,identity,[row],args


@pytest.mark.parametrize('success',[0,1])
def test_fixed_source_admission_does_not_select_policy_success(success):
    plan,identity,rows,args=setup(success)
    result=admit_formal_recovery(plan,identity,rows,**args)
    assert result['query_ordinals']==[0,8,24,48,80,120]
    assert result['candidate_pool_query_ordinals']==[8,48]


def test_pool_cohort_is_the_fixed_first_ten_sources():
    plan,identity,rows,args=setup(index=10)
    assert admit_formal_recovery(plan,identity,rows,**args)['candidate_pool_query_ordinals']==[]


@pytest.mark.parametrize('field',['checkpoint_sha256','core_source_sha256','plan_sha256','search_condition'])
def test_different_bound_source_is_rejected(field):
    plan,identity,rows,args=setup();identity[field]='other'
    with pytest.raises(ValueError,match='differs'):admit_formal_recovery(plan,identity,rows,**args)


def test_incomplete_duplicate_or_out_of_cohort_sources_are_rejected():
    plan,identity,rows,args=setup();rows[0]['roles']['full']['status']='invalid_runtime'
    with pytest.raises(ValueError,match='complete'):admit_formal_recovery(plan,identity,rows,**args)
    plan,identity,rows,args=setup()
    with pytest.raises(ValueError,match='exactly one'):admit_formal_recovery(plan,identity,rows+deepcopy(rows),**args)
    args['episode_index']=20
    with pytest.raises(ValueError,match='cohort'):admit_formal_recovery(plan,identity,rows,**args)
