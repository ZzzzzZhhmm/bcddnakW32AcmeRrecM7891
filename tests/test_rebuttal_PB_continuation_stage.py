from pathlib import Path
import importlib.util
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
