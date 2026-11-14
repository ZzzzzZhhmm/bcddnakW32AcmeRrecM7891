from datetime import datetime, timedelta, timezone

import pytest

from fastwam.research.cci_runtime_budget import assess_cci_invocation


NOW = datetime(2026, 11, 14, 2, tzinfo=timezone.utc)
EVIDENCE = {
    "source_kind": "authenticated_platform_allocation_metadata",
    "observation_sha256": "a" * 64,
}


def test_unknown_lifetime_allows_only_the_bounded_invocation_without_a_new_clock():
    plan = assess_cci_invocation(5400, 180, 900, now=NOW)
    assert plan["admit_bounded_invocation"]
    assert plan["total_bound_seconds"] == 6480
    assert plan["verified_platform_deadline"] is None
    assert not plan["remaining_platform_lifetime_verified"]
    assert not plan["new_resource_window_assumed"]
    assert not plan["ssh_helper_start_is_allocation_clock"]


def test_verified_short_deadline_blocks_a_previously_bounded_invocation():
    plan = assess_cci_invocation(5400, 180, 900, now=NOW,
        verified_platform_deadline=NOW + timedelta(hours=1), deadline_evidence=EVIDENCE)
    assert not plan["admit_bounded_invocation"]
    assert plan["reason"] == "verified_platform_deadline_insufficient"


def test_ssh_bootstrap_observation_is_not_an_allocation_deadline():
    with pytest.raises(ValueError, match="SSH process"):
        assess_cci_invocation(5400, 180, 900, now=NOW,
            verified_platform_deadline=NOW + timedelta(hours=4),
            deadline_evidence={**EVIDENCE, "source_kind": "ssh_helper_process_start"})


def test_owner_limit_counts_shutdown_and_verification_not_just_training():
    plan = assess_cci_invocation(14000, 180, 900, now=NOW)
    assert not plan["admit_bounded_invocation"]
    assert plan["reason"] == "invocation_exceeds_owner_limit"


@pytest.mark.parametrize("deadline,evidence", [
    (NOW + timedelta(hours=4), None),
    (None, EVIDENCE),
])
def test_deadline_and_evidence_are_required_together(deadline, evidence):
    with pytest.raises(ValueError, match="explicit evidence"):
        assess_cci_invocation(5400, 180, 900, now=NOW,
            verified_platform_deadline=deadline, deadline_evidence=evidence)


@pytest.mark.parametrize("arguments", [(True, 180, 900), (5400, -1, 900), (5400, 180, 0.5)])
def test_runtime_units_are_explicit_nonnegative_integer_seconds(arguments):
    with pytest.raises(ValueError):
        assess_cci_invocation(*arguments, now=NOW)


def test_verified_deadline_requires_timezone_and_a_source_digest():
    with pytest.raises(ValueError, match="timezone"):
        assess_cci_invocation(5400, 180, 900, now=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="SHA256"):
        assess_cci_invocation(5400, 180, 900, now=NOW,
            verified_platform_deadline=NOW + timedelta(hours=4),
            deadline_evidence={**EVIDENCE, "observation_sha256": "unknown"})
