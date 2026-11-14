"""Bound one CCI invocation without inventing a platform allocation clock.

An SSH bootstrap process may be restarted independently of the allocated
workspace. Its PID, start time and sleep argument are not allocation metadata.
An unknown external restart deadline remains unknown; it does not prevent a
bounded, checkpointed invocation within the owner's per-invocation limit.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from typing import Any, Mapping


def assess_cci_invocation(
    training_seconds: int,
    shutdown_seconds: int,
    verification_seconds: int,
    *,
    maximum_invocation_seconds: int = 4 * 60 * 60,
    verified_platform_deadline: datetime | None = None,
    deadline_evidence: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Check owned runtime bounds and, when available, a verified deadline.

    With no verified deadline, this permits only the bounded invocation and
    explicitly makes no claim about the workspace's remaining lifetime.
    Callers must still enforce capacity, source, campaign and uniqueness gates.
    """
    for name, value, minimum in (
        ("training_seconds", training_seconds, 1),
        ("shutdown_seconds", shutdown_seconds, 0),
        ("verification_seconds", verification_seconds, 0),
        ("maximum_invocation_seconds", maximum_invocation_seconds, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    observed = now or datetime.now(timezone.utc)
    if observed.tzinfo is None or observed.utcoffset() is None:
        raise ValueError("now must include a timezone")
    if (verified_platform_deadline is None) != (deadline_evidence is None):
        raise ValueError("a platform deadline requires its explicit evidence")
    if verified_platform_deadline is not None:
        if verified_platform_deadline.tzinfo is None or verified_platform_deadline.utcoffset() is None:
            raise ValueError("platform deadline must include a timezone")
        if deadline_evidence.get("source_kind") != "authenticated_platform_allocation_metadata":
            raise ValueError("SSH process metadata cannot establish a platform allocation deadline")
        if not re.fullmatch(r"[0-9a-f]{64}", deadline_evidence.get("observation_sha256", "")):
            raise ValueError("platform deadline evidence requires an observation SHA256")
    total = training_seconds + shutdown_seconds + verification_seconds
    within_invocation_limit = total <= maximum_invocation_seconds
    within_observed_deadline = (
        verified_platform_deadline is None
        or observed + timedelta(seconds=total) <= verified_platform_deadline
    )
    return {
        "admit_bounded_invocation": within_invocation_limit and within_observed_deadline,
        "total_bound_seconds": total,
        "maximum_invocation_seconds": maximum_invocation_seconds,
        "remaining_platform_lifetime_verified": verified_platform_deadline is not None,
        "verified_platform_deadline": (
            verified_platform_deadline.isoformat() if verified_platform_deadline else None
        ),
        "new_resource_window_assumed": False,
        "ssh_helper_start_is_allocation_clock": False,
        "reason": (
            "invocation_exceeds_owner_limit" if not within_invocation_limit else
            "verified_platform_deadline_insufficient" if not within_observed_deadline else
            "bounded_invocation_with_verified_platform_deadline" if verified_platform_deadline else
            "bounded_invocation_external_restart_deadline_unverified"
        ),
    }
