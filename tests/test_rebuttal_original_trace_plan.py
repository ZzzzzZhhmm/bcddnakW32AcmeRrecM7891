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
