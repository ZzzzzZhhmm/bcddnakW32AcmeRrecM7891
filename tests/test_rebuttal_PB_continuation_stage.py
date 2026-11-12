from pathlib import Path
import importlib.util
import json
import pytest

p=Path(__file__).resolve().parents[1]/'scripts/run_rebuttal_PB_continuation.py'
s=importlib.util.spec_from_file_location('PB_remaining_stage',p)
M=importlib.util.module_from_spec(s);s.loader.exec_module(M)


def plan():
    return dict(schema='warm.original-PB-remaining-resource-stage.v1',
        original_queries=[8,48,96],maximum_seconds=28800,GPUs=1,
        original_windows_unchanged=True,source_policy_forwards=0,
        interrupted_branches_never_retried=True,stage_root='/new',old_stage_root='/old',
        query_plans=[dict(query=q) for q in [8,48,96]])


@pytest.mark.parametrize('key,value', [('maximum_seconds',86400),('GPUs',2),
    ('original_windows_unchanged',False),('source_policy_forwards',True),
    ('interrupted_branches_never_retried',False),('original_queries',[4,8,48]),
    ('query_plans',[dict(query=48),dict(query=8)]),('query_plans',[]),
    ('query_plans',[dict(query=8),dict(query=8)]),('stage_root','/old')])
def test_cannot_reset_old_scope_expand_budget_or_select_reordered_queries(key,value):
    p=plan();p[key]=value
    with pytest.raises(ValueError):M.validate_plan(p)


def test_an_unqualified_original_query_may_stay_accounted_elsewhere_without_replacement():
    p=plan();p['query_plans']=[dict(query=8),dict(query=96)]
    with pytest.raises(ValueError,match='accounting'):M.validate_plan(p)
    p['omitted_query_accounting']={'48':dict(reason='original_AABA_not_complete')}
    assert [r['query'] for r in M.validate_plan(p)['query_plans']]==[8,96]


def owned_plan(tmp_path):
    q=tmp_path/'q8.json';stage=tmp_path/'separate_CCI';output=tmp_path/'q8_output'
    q.write_text(json.dumps(dict(query_replan=8,output=str(output),continuation_resource_stage=str(stage))))
    owner=tmp_path/'ownership.json';owner.write_text(json.dumps(dict(
        schema='warm.original-PB-fixed-query-ownership.v1',query=8,original_queries=[8,48,96],
        stage_root=str(stage),output=str(output),complete_outcome_claimed_here=False,
        query_plan=dict(path=str(q),sha256=M.H.sha(q)))))
    ref=dict(path=str(owner),sha256=M.H.sha(owner));p=plan();p['query_plans']=[dict(query=48),dict(query=96)]
    p['omitted_query_accounting']={'8':dict(reason='independently_owned_fixed_query',ownership=ref)}
    p['inputs']=[ref];return p,owner,q


def test_parallel_ownership_accounts_for_the_original_query_without_claiming_results(tmp_path):
    p,_,_=owned_plan(tmp_path)
    assert [r['query'] for r in M.validate_plan(p)['query_plans']]==[48,96]


@pytest.mark.parametrize('change',['unbound','changed_owner','changed_plan','claimed_complete','same_owner'])
def test_parallel_ownership_rejects_unsealed_changed_or_fake_completed_evidence(tmp_path,change):
    p,owner,q=owned_plan(tmp_path)
    if change=='unbound':p['inputs']=[]
    elif change=='changed_owner':owner.write_text(owner.read_text()+' ')
    elif change=='changed_plan':q.write_text(q.read_text()+' ')
    else:
        v=json.loads(owner.read_text())
        if change=='claimed_complete':v['complete_outcome_claimed_here']=True
        else:v['stage_root']=p['stage_root']
        owner.write_text(json.dumps(v));ref=dict(path=str(owner),sha256=M.H.sha(owner))
        p['inputs']=[ref];p['omitted_query_accounting']['8']['ownership']=ref
    with pytest.raises(ValueError):M.validate_plan(p)
