import pytest

from scripts.run_rebuttal_resumable_campaign import invocation_stop_status


def test_bounded_invocation_stops_only_after_new_verified_roles():
    pending=(3,'candidate_null',340800003)
    assert invocation_stop_status(pending,0,1) is None
    assert invocation_stop_status(pending,1,1)=='partial_verified_invocation_limit'
    assert invocation_stop_status(pending,3,None) is None


def test_complete_owned_cohort_is_distinct_from_partial_invocation():
    assert invocation_stop_status(None,0,1)=='complete_verified_owned_cases'
    assert invocation_stop_status(None,1,1)=='complete_verified_owned_cases'


@pytest.mark.parametrize('limit',[0,-1,True,1.5])
def test_invalid_limits_are_rejected_before_work(limit):
    with pytest.raises(ValueError):invocation_stop_status(None,0,limit)


def test_bad_new_role_count_does_not_close_a_campaign():
    with pytest.raises(ValueError):invocation_stop_status((3,'full',340800003),-1,1)
