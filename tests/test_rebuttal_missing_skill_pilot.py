"""CPU protocol fixtures,not robot outcomes or experimental measurements."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import pytest
from fastwam.research.missing_skill_pilot import CONDITIONS,PHASE,INFERENCE,validate,require_window,requests,pair

def plan():
    return dict(schema='warm.rebuttal.native-family-dev-pilot.v1',phase=PHASE,task='rearrange_blocks',official_step_limit=700,
        inference=INFERENCE.copy(),registered_pilot_seeds=[3407,3408],admitted_case_ids=[0],conditions=list(CONDITIONS),
        intervention_roles=['full','candidate_null'],root_seed=3407,formal_TEST=False,primary_NI_protocol_replaced=False,
        roles={'full':dict(model_class='WarmRetrospectionFastWAM',checkpoint_schema='warm.retrospection-checkpoint',checkpoint_update=3000,checkpoint_sha256='a'*64)},
        candidate_null_checkpoint_sha256='a'*64,universe_rows={c:dict(rows=19540) for c in CONDITIONS},worker_max_seconds=2400,stage_max_seconds=10800,
        first_admitted_utc='2026-11-11T06:00:00+00:00',absolute_deadline_utc='2026-11-11T09:00:00+00:00')

def test_fixed_order_and_counterbalance():
    p=plan();assert requests(p)==[(0,c,r) for c in CONDITIONS for r in ('full','candidate_null')]
    p['admitted_case_ids']=[1];assert requests(p)==[(1,c,r) for c in reversed(CONDITIONS) for r in ('candidate_null','full')]

@pytest.mark.parametrize('field,bad', [('phase','formal'),('registered_pilot_seeds',[340800000,340800001]),('admitted_case_ids',[0,1]),
    ('formal_TEST',True),('primary_NI_protocol_replaced',True),('official_step_limit',400),('worker_max_seconds',3600),('candidate_null_checkpoint_sha256','b'*64)])
def test_cannot_substitute_formal_sampling_scope_budget_or_weights(field,bad):
    p=plan();p[field]=bad
    with pytest.raises(ValueError):validate(p)

def test_nonempty_size_is_preserved():
    p=plan();p['universe_rows'][CONDITIONS[1]]['rows']=0
    with pytest.raises(ValueError):validate(p)

def test_original_window_cannot_be_extended_or_used_after_expiry():
    p=plan();start=datetime(2026,11,11,6,tzinfo=timezone.utc)
    require_window(p,start+timedelta(seconds=1))
    with pytest.raises(TimeoutError):require_window(p,start+timedelta(hours=3))
    p['absolute_deadline_utc']='2026-11-11T10:00:00+00:00'
    with pytest.raises(ValueError):require_window(p,start)

def rows():
    q=dict(derived_seed=8,base_gaussian={'sha256':'x'},actual_source={'sha256':'x'},model_input_sha256='im',proprio_sha256='p',context_key_sha256='c',
        declared_force_null=True,effective_gate=0,source_gate=0)
    base=dict(schema='warm.rebuttal.native-family-dev-role.v1',phase=PHASE,plan_sha256='p',task='rearrange_blocks',episode_id=0,seed=3407,
        search_condition=CONDITIONS[1],checkpoint_sha256='a'*64,initial_identity={'state':1},initial_measured_physical=[1],common_contract={'bank':'b'},
        status='complete_native',success=0,executed_commands=700,queries=[q])
    return dict(deepcopy(base),role='full'),dict(deepcopy(base),role='candidate_null')

def test_zero_development_outcomes_do_not_establish_absence_or_NI():
    r=pair(*rows());assert r['full_minus_null']==0 and r['independently_measured_missing_skill_coverage'] is None and r['noninferiority_claimed'] is False

@pytest.mark.parametrize('field,bad',[('initial_identity',{'state':2}),('seed',3408),('checkpoint_sha256','b'*64),('status','unknown_runtime_or_interruption'),('executed_commands',699)])
def test_bad_or_truncated_pair_is_never_imputed_failure(field,bad):
    a,b=rows();b[field]=bad
    with pytest.raises(ValueError):pair(a,b)

def test_actual_zero_source_boundary_is_required():
    a,b=rows();b['queries'][0]['source_gate']=.1
    with pytest.raises(ValueError):pair(a,b)
    a,b=rows();b['queries'][0]['base_gaussian']['sha256']='different'
    with pytest.raises(ValueError):pair(a,b)

def test_only_initial_common_noise_is_required_not_fictional_later_state_identity():
    a,b=rows();a['queries'].append(dict(a['queries'][0],model_input_sha256='causal_full_history'))
    b['queries'].append(dict(b['queries'][0],model_input_sha256='causal_null_history'))
    assert pair(a,b)['full_minus_null']==0
