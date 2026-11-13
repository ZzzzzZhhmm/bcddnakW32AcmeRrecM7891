import datetime as dt
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('namespace_remaining',Path(__file__).resolve().parents[1]/'src/fastwam/research/pb_namespace_remaining.py');B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)


def plan():
    return dict(schema=B.SCHEMA,query=96,maximum_seconds=14400,shutdown_margin_seconds=180,automatic_restarts=0,GPUs=1,CPUs=8,RAM_GiB=128,resource_type='idle',original_windows_unchanged=True,source_policy_forwards=0,model_updates=0,q48_interrupted_branches_never_retried=True,source_or_query_replacement=False,minimum_admission_free_bytes=4*1024**3,runtime_free_space_guard_bytes=512*1024**2,cumulative_reserved_GPU_hours=143.8054115725,lifetime_cap_GPU_hours=144,case_id=0,job_name=B.expected_name(0),old_UID=B.OLD_UIDS[0])


@pytest.mark.parametrize('key,value',[('query',48),('query',True),('maximum_seconds',18000),('shutdown_margin_seconds',0),('automatic_restarts',1),('GPUs',True),('GPUs',2),('resource_type','standard'),('source_policy_forwards',1),('model_updates',1),('q48_interrupted_branches_never_retried',False),('source_or_query_replacement',True),('minimum_admission_free_bytes',512*1024**2),('runtime_free_space_guard_bytes',0),('lifetime_cap_GPU_hours',145),('cumulative_reserved_GPU_hours',140),('case_id',True),('case_id',2),('job_name','other'),('old_UID',B.OLD_UIDS[1])])
def test_recovery_cannot_change_scope_or_resource_limits(key,value):
    p=plan();p[key]=value
    with pytest.raises(ValueError):B.validate_plan(p)


def test_both_frozen_cases_are_supported():
    for case in [0,1]:
        p=plan();p.update(case_id=case,job_name=B.expected_name(case),old_UID=B.OLD_UIDS[case]);assert B.validate_plan(p)==p


def test_only_unstarted_q96_may_follow_verified_termination():
    o=dict(first_admitted_utc='2026-11-13T00:35:00+00:00',absolute_deadline_utc='2026-11-13T10:35:00+00:00',queries={'8':{},'48':{}})
    r=dict(status='failed',exit_code=-15,end_utc='2026-11-13T03:49:00+00:00')
    first,end,stopped=B.require_old_terminal(o,r);now=dt.datetime(2026,11,13,5,tzinfo=dt.timezone.utc)
    assert B.bounded_deadline(now,first,end,stopped)==now+dt.timedelta(hours=4)
    assert B.bounded_deadline(end-dt.timedelta(minutes=30),first,end,stopped)==end
    with pytest.raises(ValueError):B.bounded_deadline(end-dt.timedelta(seconds=599),first,end,stopped)
    with pytest.raises(ValueError):B.bounded_deadline(stopped-dt.timedelta(seconds=1),first,end,stopped)
    with pytest.raises(ValueError):B.require_old_terminal(o|dict(queries={'96':{}}),r)
    for bad in [r|dict(status='running'),r|dict(exit_code=0),r|dict(end_utc=None)]:
        with pytest.raises(ValueError):B.require_old_terminal(o,bad)


def test_q96_plan_cannot_change_physics_or_model():
    old=dict(query_replan=48,selected_rank=3,output='/registered/case00/pool-q0048',physical_seed=340800000,weights='original',runtime='/original/private')
    new=old|dict(query_replan=96,selected_rank=4,output='/registered/case00/pool-q0096')
    assert B.require_q96_plan(old,new)=={'query_replan','selected_rank','output'}
    for key,value in [('weights','replacement'),('runtime','another'),('physical_seed',0),('output','/elsewhere'),('query_replan',8)]:
        with pytest.raises(ValueError):B.require_q96_plan(old,new|{key:value})


def test_authenticated_old_platform_receipt_is_required(tmp_path):
    p=plan();gate=tmp_path/'gate.json';source=tmp_path/'source.json';source.write_text('{}');p['cloud_terminal_gate']=str(gate)
    value=dict(schema='warm.PB-namespace2-old-platform-terminal.v1',case_id=0,old_UID=B.OLD_UIDS[0],old_platform_terminal=True,old_scientific_exit=-15,new_job_name=p['job_name'],authenticated_platform_receipt=dict(path=str(source),sha256=B.sha(source)))
    import json
    gate.write_text(json.dumps(value));assert B.validate_cloud_gate(p)==value
    gate.write_text(json.dumps(value|dict(old_platform_terminal=False)))
    with pytest.raises(ValueError):B.validate_cloud_gate(p)
    gate.write_text(json.dumps(value));source.write_text('{"changed":true}')
    with pytest.raises(ValueError):B.validate_cloud_gate(p)


def test_prepared_input_seal_cannot_adopt_physical_or_changed_work():
    good=dict(status='complete_PB_CPU_input_seal',plan_sha256='hash',query=96,CUDA_initialized=False,candidate_branches_executed=0,model_updates=0,new_policy_generations=0,branch_order=[str(i) for i in range(36)])
    assert B.require_prepared_CPU_seal(good,'hash')==good
    for key,value in [('plan_sha256','other'),('query',48),('CUDA_initialized',True),('candidate_branches_executed',1),('new_policy_generations',1),('model_updates',1),('branch_order',['same']*36)]:
        with pytest.raises(ValueError):B.require_prepared_CPU_seal(good|{key:value},'hash')
