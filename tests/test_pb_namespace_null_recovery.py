from pathlib import Path
import copy,datetime,importlib.util
import pytest
spec=importlib.util.spec_from_file_location('null_recovery_test',Path(__file__).resolve().parents[1]/'src/fastwam/research/pb_namespace_null_recovery.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def fixtures():
    parent=dict(status='terminal_fixed_q8_withdrawal_all_existing_output_archived',first_admitted_utc='2026-11-13T07:47:40.544934+00:00',absolute_deadline_utc='2026-11-13T11:00:00+00:00',case_ids=[0,1],cases={str(i):dict(exit_code=1,bounded_termination=False) for i in [0,1]})
    cases={i:dict(status='failed_or_interrupted_q8_null_preserve_all',case_id=i,physical_seed=340800000+i,query=8,frame=32,live_reproduction_queries=9,reproduced_native_commands=32,extra_policy_queries=0,new_branch_commands=0,model_updates=0,branches={},error=M.ERROR) for i in [0,1]}
    return parent,dict(status='complete',exit_code=0),cases,datetime.datetime(2026,11,13,9,tzinfo=datetime.timezone.utc)

def test_known_prebranch_errors_retain_original_start_and_end():
    p,r,c,n=fixtures();first,end=M.validate_failures(p,r,c,n)
    assert first.isoformat()==p['first_admitted_utc'] and end.isoformat()==p['absolute_deadline_utc']

@pytest.mark.parametrize('key,value',[('extra_policy_queries',1),('new_branch_commands',1),('branches',{'Full-pre':{}}),('error','unrelated error'),('physical_seed',0),('live_reproduction_queries',8)])
def test_observed_interventions_unknowns_or_foreign_cases_are_not_retried(key,value):
    p,r,c,n=fixtures();c[0][key]=value
    with pytest.raises(ValueError):M.validate_failures(p,r,c,n)

def test_running_or_unknown_parent_and_child_are_rejected():
    p,r,c,n=fixtures();r['status']='running'
    with pytest.raises(ValueError):M.validate_failures(p,r,c,n)
    p,r,c,n=fixtures();p['cases']['0']['exit_code']=None
    with pytest.raises(ValueError):M.validate_failures(p,r,c,n)

def test_original_window_cannot_be_extended_or_restarted():
    p,r,c,n=fixtures();p['absolute_deadline_utc']='2026-11-13T11:10:00+00:00'
    with pytest.raises(ValueError):M.validate_failures(p,r,c,n)
    p,r,c,n=fixtures();p['first_admitted_utc']='2026-11-13T10:00:00+00:00'
    with pytest.raises(ValueError):M.validate_failures(p,r,c,n)

def test_new_output_does_not_modify_original_private_runtime_or_input():
    original=dict(root='/evidence/old',cases=[dict(case_id=i,output=f'/evidence/old/case{i}/results',runtime=f'/private/{i}',lease=f'/private/{i}.lock',source=f'/source/{i}') for i in [0,1]])
    before=copy.deepcopy(original);new=M.execution_plan(original,'/evidence/PB_namespace2_null_recovery_20261113_v1')
    assert original==before
    for i in [0,1]:
        assert new['cases'][i]['runtime']==original['cases'][i]['runtime'] and new['cases'][i]['lease']==original['cases'][i]['lease']
        assert new['cases'][i]['source']==original['cases'][i]['source'] and new['cases'][i]['output']!=original['cases'][i]['output']
