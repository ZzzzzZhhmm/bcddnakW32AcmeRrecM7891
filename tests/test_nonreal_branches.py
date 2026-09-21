import copy
import numpy as np
import pytest

from fastwam.research.branches import STATE_COMPONENTS, StepResult, execute_branches


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
