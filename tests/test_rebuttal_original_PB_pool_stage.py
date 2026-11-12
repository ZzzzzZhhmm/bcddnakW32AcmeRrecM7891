from pathlib import Path
import datetime as dt
import importlib.util
import pytest

p = Path(__file__).resolve().parents[1]/'scripts/run_rebuttal_original_PB_pool_stage.py'
spec = importlib.util.spec_from_file_location('PB_physical_stage',p)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def plan():
    return dict(schema='warm.original-PB-physical-pool-stage.v1', queries=[8,48,96],
        phases=['qualify','pool'], maximum_seconds=5400, original_windows_unchanged=True,
        source_policy_forwards=0, qualification_failure_does_not_replace_fixed_query=True)


@pytest.mark.parametrize('key,value', [('queries',[8,49,96]),('phases',['pool']),
    ('maximum_seconds',7200),('original_windows_unchanged',False),
    ('source_policy_forwards',True),('source_policy_forwards',1),
    ('qualification_failure_does_not_replace_fixed_query',False)])
def test_fixed_study_and_no_clock_reset_are_required(key,value):
    p = plan()
    p[key] = value
    with pytest.raises(ValueError):
        M.validate_plan(p)


def test_deadline_is_clamped_without_extending_a_previous_window():
    first = dt.datetime(2026,11,12,11,45,tzinfo=dt.timezone.utc)
    latest = first+dt.timedelta(minutes=80)
    assert M.fixed_deadline(first,latest,5400) == latest
    assert M.fixed_deadline(first,first+dt.timedelta(hours=3),5400) == first+dt.timedelta(minutes=90)
    with pytest.raises(ValueError):
        M.fixed_deadline(first,first+dt.timedelta(seconds=599),5400)
    with pytest.raises(ValueError):
        M.fixed_deadline(first.replace(tzinfo=None),latest,5400)
