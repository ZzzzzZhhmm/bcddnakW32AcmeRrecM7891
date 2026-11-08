"""Finite-sample statistical edge cases; fixtures are not experiment results."""
import math
import pytest
from fastwam.research.paired_binary import binomial_tail_bounds,paired_difference_bounds,fixed_task_paired_report


def test_exact_reference_and_zero_event_upper_bound():
    lo,hi=binomial_tail_bounds(7,50,.025)
    # Published SciPy exact-binomial example; independent reference values.
    assert lo==pytest.approx(.05819170033997342,abs=1e-11)
    assert hi==pytest.approx(.26739600249700846,abs=1e-11)
    lo,hi=binomial_tail_bounds(0,100,.0125)
    assert lo==0 and hi==pytest.approx(1-math.pow(.0125,1/100))


def test_zero_discordance_is_uncertain_and_250_pairs_can_test_two_pp_margin():
    short=paired_difference_bounds(0,0,100,simultaneous_tasks=2)
    long=paired_difference_bounds(0,0,250,simultaneous_tasks=2)
    assert short['ci95_conservative'][0]<0<short['ci95_conservative'][1]
    assert short['one_sided95_lower']<-.02<long['one_sided95_lower']<0
    assert paired_difference_bounds(4,1,100)['difference']==.03


def test_incomplete_or_invalid_pairs_never_establish_noninferiority():
    rows=[dict(task='a',episode_id=i,phase='formal',status='complete',
               roles={'full':{'success':1},'candidate_null':{'success':1}}) for i in range(250)]
    complete=fixed_task_paired_report(rows,tasks=['a'],planned_pairs_per_task=250)
    assert complete['cohort_complete'] and complete['noninferiority_established']
    missing=fixed_task_paired_report(rows[:-1],tasks=['a'],planned_pairs_per_task=250)
    assert not missing['noninferiority_established'] and missing['tasks']['a']['not_observed']==1
    rows[-1]['status']='invalid_runtime'
    invalid=fixed_task_paired_report(rows,tasks=['a'],planned_pairs_per_task=250)
    assert not invalid['noninferiority_established'] and invalid['tasks']['a']['unknown']==1
    rows[0]['phase']='qualification'
    with pytest.raises(ValueError,match='pilots'):fixed_task_paired_report(rows,tasks=['a'],planned_pairs_per_task=250)
