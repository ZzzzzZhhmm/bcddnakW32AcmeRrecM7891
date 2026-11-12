from pathlib import Path
import importlib.util
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/collect_rebuttal_original_specialist_trace.py'
spec=importlib.util.spec_from_file_location('original_trace_plan',path);M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def original():return dict(schema='warm.original-specialist-factual-trace.v1',qualified_original_run='blocks_ranking_try-s3407-v1',checkpoint_step=14000,max_native_commands=600,capture_replans=[8,48,96],physical_seed=0)
def battery():return dict(schema='warm.original-specialist-factual-trace.native-BT.v2',qualified_original_run='battery_try-s3407-v1',checkpoint_step=12000,max_native_commands=1000,capture_replans=[8,48,96],physical_seed=0,task_name='battery_try',root_seed=3407)

def test_original_BRT_source_protocol_stays_fixed():assert M.validate_trace_plan(original())=='blocks_ranking_try'
def test_BT_uses_its_full_official_horizon():assert M.validate_trace_plan(battery())=='battery_try'

@pytest.mark.parametrize('field,value',[('max_native_commands',600),('checkpoint_step',14000),('qualified_original_run','blocks_ranking_try-s3407-v1'),('physical_seed',1),('capture_replans',[0,8,48]),('task_name','rearrange_blocks'),('root_seed',17)])
def test_original_BT_cannot_borrow_a_different_run_seed_or_cap(field,value):
    plan=battery();plan[field]=value
    with pytest.raises(ValueError):M.validate_trace_plan(plan)

def test_BRT_cap_is_not_silently_extended_to_a_different_scientific_scope():
    plan=original();plan['max_native_commands']=1000
    with pytest.raises(ValueError):M.validate_trace_plan(plan)


def cohort():return battery()|dict(schema='warm.original-specialist-factual-trace.cohort-case0.v3',physical_seed=340800000,cohort_case_id=0,roster_sha256='612cb5d1728b2bebef7c09ee61bd50933c64821be31e1df0a7aa475b61d666b8')

def test_pre_frozen_cohort_has_separate_protocol_from_extra_diagnostic0():
    assert M.validate_trace_plan(cohort())=='battery_try'
    assert M.validate_trace_plan(battery())=='battery_try'

@pytest.mark.parametrize('key,value',[('physical_seed',0),('cohort_case_id',1),('roster_sha256','another-cohort')])
def test_original_cohort_trace_cannot_replace_its_fixed_start_or_roster(key,value):
    plan=cohort();plan[key]=value
    with pytest.raises(ValueError):M.validate_trace_plan(plan)


def press_button():
    return dict(schema='warm.original-specialist-factual-trace.native-PB.v4',
        task_name='press_button',qualified_original_run='press_button-s3407-v2',
        checkpoint_step=12000,max_native_commands=1500,root_seed=3407,
        physical_seed=0,capture_replans=[8,48,96],
        original_source_commit='47277e58624366506ee032f4ad27560c7e4bdd19',
        original_source_sha256='4a1d05cb548f16bf6a9b239082b02d4697aa8e8c1f43c2f5d8ffbaec196462fd',
        checkpoint_sha256='a765d987b2df0d228782980592c39aaed0e837040c7f6f48c905e4641f79c285')


def test_PB_is_separate_from_shared_and_BT_models():
    assert M.validate_trace_plan(press_button())=='press_button'


@pytest.mark.parametrize('key,value',[
    ('max_native_commands',1000),('checkpoint_step',3000),
    ('qualified_original_run','battery_try-s3407-v1'),('physical_seed',340800000),
    ('physical_seed',False),('root_seed',3408),('task_name','battery_try'),
    ('original_source_commit','f77c633'),('original_source_sha256','shared-stage'),
    ('checkpoint_sha256','b0669047'),('capture_replans',[0,8,48])])
def test_PB_cannot_borrow_a_different_model_or_select_a_different_state(key,value):
    plan=press_button();plan[key]=value
    with pytest.raises(ValueError):M.validate_trace_plan(plan)
