import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fastwam.research.pb_pool_audit import array_sha, verify_metrics
from fastwam.research.candidate_pool import centered_effect_metrics


def sample():
    rows=[]
    for i in range(3):
        rows.append(dict(candidate_id=f'rank-{i}',valid=True,horizon=32,executed_steps=32,
                         endpoint_status='complete_horizon',observed_effect=[float(i),-float(i)],
                         predicted_effect=[float(i),-float(i)],historical_effect=[0.,0.]))
    rows.append(dict(candidate_id='rank-3',valid=True,horizon=32,executed_steps=10,
                     endpoint_status='invalid_branch',observed_effect=None,
                     predicted_effect=[99.,99.],historical_effect=[99.,99.]))
    return dict(candidates=rows)


def metrics(data):
    result=centered_effect_metrics(data)
    result['references']['constant_query']['absolute_mse']=None
    result['references']['constant_query']['absolute_mse_unavailable_reason']='Required query effect not recorded;only centered constant comparison is identified'
    return result


def test_invalid_endpoints_stay_missing_on_all_reference_supports():
    d=sample();r=verify_metrics(d,metrics(d))
    assert r['valid_candidates']==4 and r['complete_endpoints']==3
    assert r['references']['predicted']['centered_mse']==0
    assert r['references']['constant_query']['absolute_mse'] is None
    assert d['candidates'][-1]['observed_effect'] is None


def test_fabricated_zero_effect_for_invalid_branch_is_rejected():
    d=sample();d['candidates'][-1]['observed_effect']=[0.,0.]
    with pytest.raises(ValueError,match='imputed'):verify_metrics(d,{})


def test_changed_primary_reference_is_rejected():
    d=sample();m=metrics(d);m['references']['predicted']['centered_mse']=1.
    with pytest.raises(ValueError,match='Frozen metrics'):verify_metrics(d,m)


def test_array_identity_includes_dtype_shape_and_value():
    a=np.arange(6,dtype=np.float32)
    assert array_sha(a)!=array_sha(a.reshape(2,3))
    assert array_sha(a)!=array_sha(a.astype(np.float64))
    assert array_sha(a)!=array_sha(a+1)


def script():
    path=Path(__file__).resolve().parents[1]/'scripts/audit_rebuttal_PB_terminal_pools.py'
    spec=importlib.util.spec_from_file_location('audit_cli',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('key,value',[('GPUs',1),('queries',[8,96]),('max_cpu_workers',8),('scientific_jobs_read_only',False),('original_windows_unchanged',False),('new_policy_or_physical_actions',1)])
def test_auditor_cannot_expand_resources_replace_queries_or_touch_science(key,value):
    m=script();p=dict(schema='warm.PB-terminal-offline-audit.v1',GPUs=0,queries=[8,48,96],max_cpu_workers=3,scientific_jobs_read_only=True,original_windows_unchanged=True,new_policy_or_physical_actions=0,query_plans=[dict(query=q) for q in [8,48,96]])
    p[key]=value
    with pytest.raises(ValueError):m.validate_plan(p)


@pytest.mark.parametrize('exit_code,bounded',[(-15,False),(0,True)])
def test_interrupted_exit_cannot_release_even_a_complete_operator(tmp_path,monkeypatch,exit_code,bounded):
    m=script()
    monkeypatch.setattr(m,'verify_archive',lambda *a:dict(sha256='preserved'))
    monkeypatch.setattr(m,'read',lambda *a:dict(status='complete_original_PB_fixed_query_pool'))
    monkeypatch.setattr(m,'audit_pool',lambda *a,**k:pytest.fail('Interrupted query must not release measurements'))
    result=m.terminal_audit(dict(query=8,root=str(tmp_path/'raw')),dict(archive={},exit_code=exit_code,bounded_termination=bounded),tmp_path)
    assert result['status']=='incomplete_physical_query_retained' and result['primary_metrics'] is None
