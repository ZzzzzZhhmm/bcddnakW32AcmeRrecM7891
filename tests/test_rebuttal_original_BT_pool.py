from pathlib import Path
import importlib.util
import pytest
path=Path(__file__).resolve().parents[1]/'scripts/collect_rebuttal_original_fresh_pool.py'
spec=importlib.util.spec_from_file_location('BT_pool_plan',path);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
def plan():return dict(schema=M.BT_POOL_SCHEMA,task_name='battery_try',query_replan=8,selected_rank=4,physical_seed=340800000,root_seed=3407,cohort_case_id=0,checkpoint_step=12000,horizon=32,execution_prefix=4,checkpoint_sha256='27c137d689007b38f6204d3dd12bb038550b1e58403506d7a9894bd6fcd43a9f',roster_sha256='612cb5d1728b2bebef7c09ee61bd50933c64821be31e1df0a7aa475b61d666b8',factual_source_plan_sha256='9d479fa7d1dedcc120edff3e38561d84c46d8214a649a436d004ecbbce72aec1',next_interaction_rubric='warm.rmbench.physical-next-interaction.v1')
def source(count=1000):return dict(status='complete_fixed_original_factual_episode',candidate_seed=340800000,cohort_case_id=0,source_sha256='ff32342d286929ddc5904d55814c66e965269f6b749c1e1b6df1060aaef0b7fe',source_sha256_after='ff32342d286929ddc5904d55814c66e965269f6b749c1e1b6df1060aaef0b7fe',checkpoint_sha256=plan()['checkpoint_sha256'],initial_observation_exact_to_qualification=True,first_replan_exact_to_qualification=True,model_updates=0,native_commands=count,success_flag=False)
def trace(count=1000):return [dict(index=i,take_action_cnt=i+1,observed_TOPP_errors=[]) for i in range(count)]
def test_old_BRT_protocol_stays_fixed():
 for q,s in [(8,28),(48,0),(96,10)]:assert not M.validate_original_pool_plan(dict(schema='warm.original-specialist-fresh-pool.v1',query_replan=q,selected_rank=s))['is_BT']
 with pytest.raises(ValueError):M.validate_original_pool_plan(dict(schema='warm.original-specialist-fresh-pool.v1',query_replan=8,selected_rank=4))
def test_BT_selected_rank_is_bound_by_original_seal_not_hardcoded_BRT_rank():assert M.validate_original_pool_plan(plan())['physical_seed']==340800000
@pytest.mark.parametrize('key,value',[('physical_seed',0),('root_seed',17),('cohort_case_id',1),('query_replan',7),('checkpoint_step',14000),('roster_sha256','other'),('horizon',26),('execution_prefix',8),('selected_rank',32),('factual_source_plan_sha256','other'),('next_interaction_rubric','posthoc')])
def test_BT_cannot_borrow_diagnostic_or_change_fixed_protocol(key,value):
 p=plan();p[key]=value
 with pytest.raises(ValueError):M.validate_original_pool_plan(p)
def test_full_BT_native_source_and_genuine_early_success_are_complete():
 M.validate_BT_factual_source(source(),trace(),plan())
 s=source(100);s['success_flag']=True;M.validate_BT_factual_source(s,trace(100),plan())
def test_partial_BT_or_unreached_fixed_query_stays_unknown():
 with pytest.raises(ValueError):M.validate_BT_factual_source(source(100),trace(100),plan())
 p=plan();p['query_replan']=48;s=source(100);s['success_flag']=True
 with pytest.raises(ValueError):M.validate_BT_factual_source(s,trace(100),p)
def test_BT_native_counter_and_prefix_TOPP_drift_cannot_be_ignored():
 t=trace();t[2]['take_action_cnt']=9
 with pytest.raises(ValueError):M.validate_BT_factual_source(source(),t,plan())
 t=trace();t[2]['observed_TOPP_errors']=['TOPP failure']
 with pytest.raises(ValueError):M.validate_BT_factual_source(source(),t,plan())
def test_unknown_raw_label_is_not_imputed_as_failure_or_repair():
 rows={'anchor-a':dict(independent_label=dict(applicable=1),endpoint_status='complete_horizon'),'rank-4':dict(independent_label=dict(applicable=None),endpoint_status='invalid_branch')}
 r=M.paired_BT_repair_summary(rows,4);assert not r['paired_known'] and r['repaired'] is None and not r['population_effect_estimated']
def test_single_state_repair_and_harm_are_separate_from_population_success():
 rows={'anchor-a':dict(independent_label=dict(applicable=1),endpoint_status='complete_horizon'),'rank-4':dict(independent_label=dict(applicable=0),endpoint_status='complete_horizon')}
 r=M.paired_BT_repair_summary(rows,4);assert r['repaired'] and not r['harmed'] and r['independent_source_states']==1
