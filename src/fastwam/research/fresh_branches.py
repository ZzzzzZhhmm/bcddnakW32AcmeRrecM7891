"""Fresh-scene branch checks that never unpack or advance a factual policy."""
from __future__ import annotations

import numpy as np

from .branches import InvalidBranchExecution
from .rmbench_branches import _IdleChannel, exact_hold_path


class _CheckedChannel(_IdleChannel):
    def __init__(self, channel, errors):
        super().__init__(channel)
        self.errors = errors

    def send(self, _message):
        self.errors.append("forbidden planner RPC send")
        return super().send(_message)

    def recv(self):
        self.errors.append("forbidden planner RPC receive")
        return super().recv()


def require_sealed_actions(raw, refined, trace, frame, *, prefix=4):
    """Bind a recorded proposal and generation to its original executed prefix."""
    actions = {}
    for role, value in (("raw", raw), ("refined", refined)):
        value = np.asarray(value)
        if value.shape != (32, 14) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("sealed actions require finite native float32 H32x14")
        actions[role] = value.copy()
    reference = np.asarray(trace["commands"])[frame:frame + prefix]
    if reference.shape != (prefix, 14) or not np.array_equal(actions["refined"][:prefix], reference):
        raise ValueError("sealed refined prefix differs from original Full execution")
    if np.array_equal(actions["raw"], actions["refined"]):
        raise ValueError("raw and refined must exercise distinct recorded actions")
    return actions


class OfficialQposGuard:
    """Detect swallowed TOPP failures and forbid cuRobo RPC during qpos steps.

    Wrappers are removed before state snapshots. Nothing is restored via
    physical unpack; the caller constructs a new independent scene per branch.
    """
    def __init__(self, env):
        self.env = env
        self.channels = {}
        self.methods = {}
        self.errors = []

    def __enter__(self):
        robot = self.env.robot
        try:
            if robot.communication_flag:
                for name in ("left_conn", "right_conn"):
                    self.channels[name] = getattr(robot, name)
                    setattr(robot, name, _CheckedChannel(self.channels[name], self.errors))
            for name in ("left_mplib_planner", "right_mplib_planner"):
                planner = getattr(robot, name)
                original = planner.TOPP
                self.methods[name] = (original, "TOPP" in vars(planner))

                def checked(*args, _original=original, _name=name, **kwargs):
                    try:
                        result = _original(*args, **kwargs)
                        if result[1].shape[0] == 0 and not exact_hold_path(args[0]):
                            self.errors.append(_name + ": empty non-hold trajectory")
                        return result
                    except Exception as exc:
                        self.errors.append(_name + ": " + repr(exc))
                        raise

                planner.TOPP = checked
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def step(self, command):
        command = np.asarray(command)
        if command.dtype != np.float32 or command.shape != (14,) or not np.isfinite(command).all():
            raise ValueError("finite original float32 absolute qpos required")
        self.errors.clear()
        before = self.env.take_action_cnt
        self.env.take_action(command.copy(), action_type="qpos")
        if self.errors:
            raise InvalidBranchExecution("official TOPP rejection: " + str(self.errors))
        return self.env.take_action_cnt == before + 1

    def __exit__(self, *_):
        for name, channel in self.channels.items():
            setattr(self.env.robot, name, channel)
        self.channels.clear()
        for name, (method, was_field) in self.methods.items():
            planner = getattr(self.env.robot, name)
            if was_field:
                planner.TOPP = method
            else:
                del planner.TOPP
        self.methods.clear()


def require_repeated_endpoints(rows):
    """Require complete A-A-B-A and exact repeated physical/RGB endpoints."""
    if [r["role"] for r in rows] != ["refined", "refined", "raw", "refined"]:
        raise ValueError("fixed A-A-B-A order required")
    a = rows[0]
    if any(not row["valid"] or row["commands"] != 32 for row in rows):
        raise ValueError("full H32 support required for branch qualification")
    if any(row["parent_hashes"] != a["parent_hashes"] for row in rows):
        raise ValueError("fresh branch parent differs")
    if any(rows[i]["endpoint_hashes"] != a["endpoint_hashes"] for i in (1, 3)):
        raise ValueError("repeated A physical/controller/task/RGB endpoint differs")
    if rows[2]["endpoint_hashes"]["physics"] == a["endpoint_hashes"]["physics"]:
        raise ValueError("B did not exercise a distinct physical endpoint")
