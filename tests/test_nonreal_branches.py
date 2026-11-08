import copy
import numpy as np
import pytest

from fastwam.research.branches import STATE_COMPONENTS, InvalidBranchExecution, StepResult, execute_branches


class Toy:
    def __init__(self, stop=None, fail=None, bad_restore=False):
        self.n = 0
        self.stop, self.fail, self.bad_restore = stop, fail, bad_restore

    def snapshot(self):
        return self.n

    def restore(self, state):
        self.n = state if not self.bad_restore else 99

    def fingerprint(self):
        return {key: str(self.n) for key in STATE_COMPONENTS}

    def effect_tokens(self):
        return np.array([self.n, 0.])

    def step(self, command):
        if self.n == self.fail:
            raise RuntimeError("injected execution failure")
        self.n += 1
        return StepResult(True, command.copy(), "success" if self.n == self.stop else None)


def run(backend, rows):
    return execute_branches(backend, query_id="q1", candidates={"a": np.ones((32, 2)), "b": np.zeros((32, 2))},
                            proposal_sha256="a"*64,
                            project_effect=lambda before, delta: delta, emit=rows.append)


def test_branch_horizon_and_parent_restored_between_candidates():
    backend, rows = Toy(), []
    result = run(backend, rows)
    assert backend.n == 0
    assert len(rows) == 4
    assert all(r["executed_steps"] == 32 for r in result)
    assert all(r["observed_effect"] == [32., 0.] for r in result)


def test_early_success_is_recorded_but_not_imputed():
    result = run(Toy(stop=7), [])
    assert all(r["executed_steps"] == 7 and r["observed_effect"] is None for r in result)
    assert all(r["termination"] == "success" for r in result)


@pytest.mark.parametrize("backend", [Toy(fail=2), Toy(bad_restore=True)])
def test_failures_are_durable_and_stop_contaminating_next_candidate(backend):
    rows = []
    with pytest.raises(RuntimeError, match="recorded"):
        run(backend, rows)
    assert len(rows) == 2
    assert rows[-1]["kind"] == "branch_result"
    assert rows[-1]["error"] is not None


def test_attempt_records_are_immutable_after_execution():
    rows = []
    run(Toy(), rows)
    assert rows[0]["actual_commands"] == []
    assert rows[0]["clipped_steps"] == []
    assert len(rows[1]["actual_commands"]) == 32


def test_mutating_initial_feature_read_restores_parent_before_failure():
    class MutatingRead(Toy):
        def effect_tokens(self):
            self.n += 1
            return super().effect_tokens()
    backend = MutatingRead()
    with pytest.raises(RuntimeError, match="mutated parent"):
        run(backend, [])
    assert backend.n == 0


def test_independent_outcome_read_before_parent_restoration():
    backend, rows = Toy(stop=7), []
    result = execute_branches(backend, query_id="q1", candidates={"a": np.zeros((32, 2))},
        proposal_sha256="b"*64, project_effect=lambda pre, delta: delta, emit=rows.append,
        observe_outcome=lambda: {"task_progress": backend.n})
    assert result[0]["independent_outcome"] == {"task_progress": 7}
    assert result[0]["observed_effect"] is None
    assert backend.n == 0


def test_known_executor_rejection_can_continue_only_after_exact_restoration():
    class RejectOne(Toy):
        def step(self, command):
            if command[0] == 1 and self.n == 2:
                raise InvalidBranchExecution('known planner rejection')
            return super().step(command)
    backend, rows = RejectOne(), []
    result=execute_branches(backend,query_id='q',candidates={'bad':np.ones((32,2)),'good':np.zeros((32,2))},
        proposal_sha256='a'*64,project_effect=lambda before,delta:delta,emit=rows.append,
        continue_invalid_execution=True)
    assert backend.n==0 and len(rows)==4
    assert result[0]['endpoint_status']=='invalid_branch' and result[0]['executed_steps']==2
    assert result[0]['observed_effect'] is None and result[0]['parent_restored'] is True
    assert 'independent_outcome' not in result[0]
    assert result[1]['endpoint_status']=='complete_horizon' and result[1]['observed_effect']==[32.,0.]


@pytest.mark.parametrize('backend',[Toy(fail=2),Toy(bad_restore=True)])
def test_opt_in_continuation_does_not_hide_unrelated_errors_or_restore_failures(backend):
    rows=[]
    with pytest.raises(RuntimeError,match='recorded'):
        execute_branches(backend,query_id='q',candidates={'a':np.ones((32,2)),'b':np.zeros((32,2))},
            proposal_sha256='a'*64,project_effect=lambda before,delta:delta,emit=rows.append,
            continue_invalid_execution=True)
    assert len(rows)==2


def test_executor_rejection_followed_by_bad_restore_still_stops():
    class BadAfterRejection(Toy):
        def step(self,command):
            self.bad_restore=True
            raise InvalidBranchExecution('known rejection')
    rows=[]
    with pytest.raises(RuntimeError,match='recorded'):
        execute_branches(BadAfterRejection(),query_id='q',candidates={'a':np.zeros((32,2)),'b':np.ones((32,2))},
            proposal_sha256='a'*64,project_effect=lambda before,delta:delta,emit=rows.append,
            continue_invalid_execution=True)
    assert len(rows)==2 and rows[-1]['parent_restored'] is False
