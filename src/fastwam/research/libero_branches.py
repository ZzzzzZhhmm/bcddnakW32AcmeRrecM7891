"""LIBERO/robosuite 1.4 snapshot backend for fixed-state action branches.

Model inference and working-memory updates are deliberately outside this class.
Only cached observations are encoded inside a branch. Qualification on the live
environment is mandatory; a MuJoCo qpos/qvel snapshot alone is insufficient.
"""
from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import hashlib
import pickle
import random
import time

import numpy as np

from .branches import StepResult


def digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def _plain(value):
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic, np.ndarray)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_plain(v) for v in value)
    if isinstance(value, dict):
        return all(_plain(k) and _plain(v) for k, v in value.items())
    return False


def plain_fields(obj):
    # The historical evaluator's runtime is a slots dataclass (no __dict__).
    values = ({f.name: getattr(obj, f.name) for f in fields(obj)}
              if is_dataclass(obj) else vars(obj))
    return {k: copy.deepcopy(v) for k, v in values.items() if _plain(v)}


class LiberoBranchBackend:
    def __init__(self, env, observation, *, encode, policy_fingerprint):
        import mujoco
        self.env, self.obs = env, copy.deepcopy(observation)
        self.encode, self.policy_fingerprint = encode, policy_fingerprint
        self.mj = mujoco
        self.model, self.data = env.sim.model._model, env.sim.data._data
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.last_outcome = None

    def _physics(self):
        state = np.empty(self.mj.mj_stateSize(self.model, self.spec), dtype=np.float64)
        self.mj.mj_getState(self.model, self.data, state, self.spec)
        visual = {key: np.array(getattr(self.model, key), copy=True)
                  for key in ("geom_rgba", "site_rgba", "mat_rgba", "body_pos", "body_quat")}
        return state, visual

    def _controllers(self):
        result = []
        for robot in self.env.robots:
            controller = robot.controller
            unsupported = {k: type(v).__name__ for k, v in vars(controller).items()
                           if k != "sim" and not _plain(v)}
            if unsupported:
                raise ValueError(f"unsupported controller state: {unsupported}")
            buffers = {k: copy.deepcopy(vars(v)) for k, v in vars(robot).items()
                       if type(v).__module__ == "robosuite.utils.buffers"}
            result.append(dict(controller=plain_fields(controller), robot=plain_fields(robot),
                               buffers=buffers, gripper_action=copy.deepcopy(robot.gripper.current_action)))
        return result

    def _task(self):
        domain = self.env.env
        return dict(env=plain_fields(domain),
                    object_states={k: plain_fields(v) for k, v in domain.object_states_dict.items()})

    def _observation(self):
        # Sensor/corruption/delay callables are immutable code and retain their
        # original closure. Their mutable observable clocks/caches are captured.
        return dict(obs=copy.deepcopy(self.obs), observables={
            k: plain_fields(v) for k, v in self.env.env._observables.items()})

    def _rng(self):
        import torch
        return dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state().numpy().copy(),
                    cuda=[s.cpu().numpy().copy() for s in torch.cuda.get_rng_state_all()])

    def snapshot(self):
        return dict(physics=self._physics(), controller=self._controllers(), task=self._task(),
                    observation=self._observation(), rng=self._rng(), policy=self.policy_fingerprint())

    def fingerprint(self):
        return {key: digest(value) for key, value in self.snapshot().items()}

    def restore(self, state):
        import torch
        vector, visual = state["physics"]
        for key, value in visual.items():
            getattr(self.model, key)[:] = value
        self.mj.mj_setState(self.model, self.data, vector, self.spec)
        self.mj.mj_forward(self.model, self.data)
        # mj_forward can update solver warm-start state included in INTEGRATION.
        self.mj.mj_setState(self.model, self.data, vector, self.spec)
        self.env.env.__dict__.update(copy.deepcopy(state["task"]["env"]))
        for key, values in state["task"]["object_states"].items():
            self.env.env.object_states_dict[key].__dict__.update(copy.deepcopy(values))
        for robot, saved in zip(self.env.robots, state["controller"], strict=True):
            robot.__dict__.update(copy.deepcopy(saved["robot"]))
            robot.controller.__dict__.update(copy.deepcopy(saved["controller"]))
            robot.gripper.current_action = copy.deepcopy(saved["gripper_action"])
            for key, values in saved["buffers"].items():
                getattr(robot, key).__dict__.update(copy.deepcopy(values))
        for key, values in state["observation"]["observables"].items():
            self.env.env._observables[key].__dict__.update(copy.deepcopy(values))
        self.obs = copy.deepcopy(state["observation"]["obs"])
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(torch.from_numpy(rng["torch"].copy()))
        torch.cuda.set_rng_state_all([torch.from_numpy(v.copy()) for v in rng["cuda"]])
        if self.policy_fingerprint() != state["policy"]:
            raise RuntimeError("branch mutated policy memory; refusing to continue")

    def effect_tokens(self):
        return self.encode(self.obs)

    def step(self, command):
        low, high = self.env.env.action_spec
        actual = np.clip(np.asarray(command, dtype=np.float32), low, high)
        self.obs, _, done, info = self.env.step(actual)
        self.last_outcome = dict(done=bool(done), info=info)
        return StepResult(True, actual, "success" if done else None,
                          not np.array_equal(actual, command))


def qualify(backend, *, horizon=32):
    """A-A-B-A checks complete integration/controller/observation restoration.

    No tolerance is fitted to outcomes: MuJoCo endpoint integration uses fixed
    1e-10 absolute tolerance and rendered uint8 endpoints must match exactly.
    """
    state, parent = backend.snapshot(), backend.fingerprint()
    a = np.zeros((horizon, 7), dtype=np.float32)
    a[:, -1] = -1
    b = a.copy()
    b[:, 0] = .1
    trials = []
    started = time.perf_counter()
    try:
        for commands in (a, a, b, a):
            backend.restore(state)
            if backend.fingerprint() != parent:
                raise RuntimeError("parent fingerprint failed restoration")
            before = time.perf_counter()
            executed = 0
            for command in commands:
                result = backend.step(command)
                executed += 1
                if result.termination:
                    break
            trials.append(dict(physics=backend._physics()[0], obs=copy.deepcopy(backend.obs),
                               seconds=time.perf_counter()-before, steps=executed))
        for j in (1, 3):
            if not np.allclose(trials[0]["physics"], trials[j]["physics"], atol=1e-10, rtol=0):
                raise RuntimeError("A-A/B-A physical endpoint differs")
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                if not np.array_equal(trials[0]["obs"][key], trials[j]["obs"][key]):
                    raise RuntimeError("A-A/B-A rendered endpoint differs")
        return dict(status="passed", horizon=horizon, order="A-A-B-A",
                    physics_atol=1e-10, rendered_exact=True,
                    branch_seconds=[t["seconds"] for t in trials],
                    executed_steps=[t["steps"] for t in trials], elapsed_seconds=time.perf_counter()-started)
    finally:
        backend.restore(state)
        if backend.fingerprint() != parent:
            raise RuntimeError("qualification failed final parent restoration")
