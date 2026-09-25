"""Candidate branch execution with an explicit simulator restoration contract.

Backends must capture physics, robot/controller state, task counters, RNG,
observations, and policy memory/cursor. This engine never calls check_success:
RMBench's task check mutates state, so the backend reads the executor's existing
termination flags. A backend must be qualified on the actual simulator before
scientific use; the protocol and toy tests alone are not such qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Callable, Mapping, Protocol

import numpy as np


STATE_COMPONENTS = frozenset({"physics", "controller", "task", "rng", "observation", "policy"})


@dataclass(frozen=True)
class StepResult:
    executed: bool
    command: np.ndarray | None
    termination: str | None = None
    clipping: bool = False


class BranchBackend(Protocol):
    def snapshot(self) -> object: ...
    def restore(self, snapshot: object) -> None: ...
    def fingerprint(self) -> Mapping[str, str]: ...
    def step(self, command: np.ndarray) -> StepResult: ...
    def effect_tokens(self) -> np.ndarray: ...


def _fingerprint(backend):
    values = dict(backend.fingerprint())
    if set(values) != STATE_COMPONENTS or any(not isinstance(v, str) or not v for v in values.values()):
        raise ValueError("backend fingerprint must cover all six state components")
    return values


def execute_branches(backend: BranchBackend, *, query_id: str,
                     candidates: Mapping[str, np.ndarray],
                     proposal_sha256: str,
                     project_effect: Callable[[np.ndarray, np.ndarray], np.ndarray],
                     emit: Callable[[dict], None], horizon: int = 32,
                     observe_outcome: Callable[[], dict] | None = None) -> list[dict]:
    """Record every attempt, run H commands without replanning, always restore.

    `candidates` are denormalized adapted means, not stored actions or new policy
    rollouts. Predictions/labels must be sealed elsewhere before entering here.
    A restore mismatch stops the job; it can never silently poison later queries.
    """
    if type(horizon) is not int or horizon <= 0 or not query_id:
        raise ValueError("positive integer horizon and query_id required")
    if len(proposal_sha256) != 64 or any(c not in "0123456789abcdef" for c in proposal_sha256):
        raise ValueError("sealed proposal metadata SHA256 required")
    validated = {}
    for candidate_id, actions in candidates.items():
        array = np.asarray(actions, dtype=np.float32)
        if not candidate_id or array.ndim != 2 or array.shape[0] != horizon or not array.shape[1] or not np.isfinite(array).all():
            raise ValueError("candidate must contain exactly H finite commands")
        validated[candidate_id] = array.copy()
    snapshot = backend.snapshot()
    parent = _fingerprint(backend)
    def restore():
        backend.restore(snapshot)
        if _fingerprint(backend) != parent:
            raise RuntimeError("branch restoration fingerprint mismatch")

    try:
        before = np.array(backend.effect_tokens(), copy=True)
        if not np.isfinite(before).all():
            raise ValueError("nonfinite initial effect tokens")
        # Observation/feature reads must not advance simulator/task state.
        if _fingerprint(backend) != parent:
            raise RuntimeError("effect token read mutated parent state")
    except Exception:
        restore()
        raise

    results = []
    for candidate_id, actions in validated.items():
        row = {"query_id": query_id, "candidate_id": candidate_id, "horizon": horizon,
               "proposal_sha256": proposal_sha256, "parent_fingerprint": parent,
               "executed_steps": 0, "endpoint_status": "attempted", "termination": None,
               "observed_effect": None, "planned_commands": actions.tolist(),
               "actual_commands": [], "clipped_steps": [], "error": None}
        emit(deepcopy({**row, "kind": "branch_attempt"}))
        failure = None
        try:
            restore()
            for t, command in enumerate(actions):
                outcome = backend.step(command.copy())
                if not isinstance(outcome, StepResult):
                    raise TypeError("step must return StepResult")
                if outcome.executed:
                    actual = np.asarray(outcome.command, dtype=np.float32)
                    if actual.shape != command.shape or not np.isfinite(actual).all():
                        raise ValueError("invalid actual command from executor")
                    row["actual_commands"].append(actual.tolist())
                    row["executed_steps"] += 1
                    if outcome.clipping:
                        row["clipped_steps"].append(t)
                row["termination"] = outcome.termination
                if outcome.termination is not None or not outcome.executed:
                    break
            if row["executed_steps"] == horizon:
                endpoint = np.asarray(backend.effect_tokens())
                effect = np.asarray(project_effect(before, endpoint - before), dtype=np.float32)
                if effect.ndim != 1 or not effect.size or not np.isfinite(effect).all():
                    raise ValueError("invalid observed effect projection")
                row["observed_effect"] = effect.tolist()
                row["endpoint_status"] = "complete_horizon"
            else:
                row["endpoint_status"] = "incomplete_horizon"
            if observe_outcome is not None:
                # Independent task outcome, read before restoring the parent.
                # This callback never receives predictions or ranking scores.
                row["independent_outcome"] = deepcopy(observe_outcome())
        except Exception as e:
            row["endpoint_status"] = "invalid_branch"
            row["error"] = f"{type(e).__name__}: {e}"
            failure = e
        finally:
            try:
                restore()
                row["parent_restored"] = True
            except Exception as e:
                row["parent_restored"] = False
                row["restoration_error"] = f"{type(e).__name__}: {e}"
                failure = e
            emit(deepcopy({**row, "kind": "branch_result"}))
        results.append(row)
        if failure is not None:
            raise RuntimeError("branch failed; attempt/result were recorded") from failure
    return results
