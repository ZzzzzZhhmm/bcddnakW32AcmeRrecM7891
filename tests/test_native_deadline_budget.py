"""Clock/admission fixtures do not execute a policy or simulator."""
import pytest
from fastwam.research import matched_native as M


def test_admission_is_checked_once_and_every_command_checks_same_absolute_clock(monkeypatch):
    calls=[];current=[10.]
    monkeypatch.setattr(M,'require_deadline',lambda plan:calls.append(plan) or 20.)
    plan={'fixed':'identity'};budget=M.make_deadline_budget(plan,clock=lambda:current[0])
    for _ in range(1000):budget()
    assert calls==[plan]
    current[0]=20.
    with pytest.raises(TimeoutError):budget()
    current[0]=21.
    with pytest.raises(TimeoutError):budget()
    assert len(calls)==1


def test_invalid_admission_never_yields_a_budget(monkeypatch):
    def deny(_):raise ValueError('bad source or proof')
    monkeypatch.setattr(M,'require_deadline',deny)
    with pytest.raises(ValueError):M.make_deadline_budget({})
