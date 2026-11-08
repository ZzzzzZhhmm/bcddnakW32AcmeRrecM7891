"""Experimental RMBench qpos restoration adapter; live qualification required.

Only Rearrange Blocks and Battery Try with the pinned CPU-PhysX, no-video,
absolute-qpos executor are admitted. Idle cuRobo channels are guarded: any RPC
inside a branch fails instead of silently omitting a planner process's state.
Passing a reset probe does not qualify contact-stage queries or WARM memory.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import pickle
import random

import numpy as np

from .branches import InvalidBranchExecution, StepResult


@dataclass
class _Pose:
    p: np.ndarray
    q: np.ndarray


def _capture(value):
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic, np.ndarray)):
        return deepcopy(value)
    if type(value).__name__ == "Pose" and type(value).__module__.startswith("sapien"):
        return _Pose(np.array(value.p, copy=True), np.array(value.q, copy=True))
    if isinstance(value, (list, tuple)):
        return type(value)(_capture(v) for v in value)
    if isinstance(value, dict):
        return {_capture(k): _capture(v) for k, v in value.items()}
    raise TypeError(type(value).__name__)


def _decode(value):
    if isinstance(value, _Pose):
        import sapien
        return sapien.Pose(value.p, value.q)
    if isinstance(value, (list, tuple)):
        return type(value)(_decode(v) for v in value)
    if isinstance(value, dict):
        return {_decode(k): _decode(v) for k, v in value.items()}
    return deepcopy(value)


def python_state(obj):
    """Capture value fields; expose opaque handles explicitly in topology."""
    values, opaque = {}, {}
    for name, value in vars(obj).items():
        try:
            values[name] = _capture(value)
        except TypeError:
            opaque[name] = (type(value).__module__, type(value).__name__, id(value))
    return {"values": values, "opaque": opaque}


def restore_python(obj, state):
    current = python_state(obj)
    if current["opaque"] != state["opaque"]:
        raise RuntimeError("opaque object topology changed; restoration unsupported")
    for name in set(current["values"]) - set(state["values"]):
        delattr(obj, name)
    for name, value in state["values"].items():
        setattr(obj, name, _decode(value))


def digest(value):
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def exact_hold_path(path):
    path = np.asarray(path)
    return path.ndim == 2 and len(path) >= 2 and np.array_equal(path, np.repeat(path[:1], len(path), axis=0))


def rng_state():
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().numpy().copy(),
            "cuda": [s.cpu().numpy().copy() for s in torch.cuda.get_rng_state_all()]}


def articulation_state(env):
    return [{"name": art.name, "qf": np.array(art.get_qf(), copy=True),
             "joints": [(np.array(j.get_drive_target(), copy=True),
                         np.array(j.get_drive_velocity_target(), copy=True))
                        for j in art.get_active_joints()]}
            for art in env.scene.get_all_articulations()]


def canonicalize_query(env):
    """Round-trip the native state BEFORE observation/action sealing.

    Recompute derived scene/camera transforms without a physics step. Native
    CPU bytes, drive/force targets, task values (except refreshed now_obs),
    robot values and all exposed RNG states must remain bitwise unchanged.
    This is an explicit evaluation protocol step, not a hidden branch repair.
    """
    before = env.scene.physx_system.pack()
    poses = env.scene.pack_poses()
    task = python_state(env); task["values"].pop("now_obs", None)
    controls = (python_state(env.robot), articulation_state(env))
    rng = rng_state()
    env.scene.physx_system.unpack(before)
    env._update_render()
    observation = env.get_obs()
    after = env.scene.physx_system.pack()
    task_after = python_state(env); task_after["values"].pop("now_obs", None)
    if (before != after or digest(task) != digest(task_after) or
            digest(controls) != digest((python_state(env.robot), articulation_state(env))) or
            digest(rng) != digest(rng_state())):
        raise RuntimeError("query canonicalization changed native state, controller, task or RNG")
    return observation, {"status": "cpu_bitwise_unchanged", "before_action_sealing": True,
        "cpu_before_sha256": hashlib.sha256(before).hexdigest(),
        "cpu_after_sha256": hashlib.sha256(after).hexdigest(),
        "poses_before_sha256": hashlib.sha256(poses).hexdigest(),
        "poses_after_sha256": hashlib.sha256(env.scene.pack_poses()).hexdigest(),
        "physics_steps": 0, "task_controller_and_RNG_unchanged": True}


def prepare_native_query(env, *, max_rounds=5, emit=lambda record: None):
    """Construct an idempotent native checkpoint before sealing any proposal.

    Native root-pose re-encoding is allowed and explicitly recorded. This does
    NOT claim bitwise preservation of the original live pose encoding. No
    physics step, task/controller/RNG change or outcome-based selection is
    allowed. Failure to reach a fixed point within five rounds stops the probe.
    """
    if type(max_rounds) is not int or not 1 <= max_rounds <= 5:
        raise ValueError("native query preparation permits 1..5 rounds only")
    env.get_obs()  # Initialize read caches before checking invariants.
    original = env.scene.physx_system.pack()
    for index in range(max_rounds):
        before = env.scene.physx_system.pack(); poses = env.scene.pack_poses()
        task = python_state(env); task["values"].pop("now_obs", None)
        controls = (python_state(env.robot), articulation_state(env)); rng = rng_state()
        env.scene.physx_system.unpack(before); env._update_render()
        observation = env.get_obs()
        after = env.scene.physx_system.pack(); poses_after = env.scene.pack_poses()
        task_after = python_state(env); task_after["values"].pop("now_obs", None)
        invariant = (digest(task)==digest(task_after) and
                     digest(controls)==digest((python_state(env.robot), articulation_state(env))) and
                     digest(rng)==digest(rng_state()))
        emit({"round":index, "cpu_before":before, "cpu_after":after,
              "poses_before":poses, "poses_after":poses_after,
              "task_controller_rng_unchanged":invariant})
        if not invariant:
            raise RuntimeError("native query preparation changed task, controller or RNG")
        if before==after and poses==poses_after:
            return observation, {"status":"native_checkpoint_fixed_point", "before_action_sealing":True,
                "original_cpu_sha256":hashlib.sha256(original).hexdigest(),
                "original_cpu_encoding_changed":original!=after, "rounds":index+1,
                "cpu_before_sha256":hashlib.sha256(before).hexdigest(),
                "cpu_after_sha256":hashlib.sha256(after).hexdigest(),
                "poses_after_sha256":hashlib.sha256(poses_after).hexdigest(),
                "physics_steps":0, "task_controller_and_RNG_unchanged":True}
    raise RuntimeError("native checkpoint did not reach an exact fixed point; collection remains unqualified")


class _IdleChannel:
    def __init__(self, channel):
        self.channel = channel

    def send(self, _message):
        raise RuntimeError("planner RPC in a qpos branch: subprocess state is not captured")

    def recv(self):
        raise RuntimeError("planner RPC in a qpos branch: subprocess state is not captured")

    def __getattr__(self, name):
        return getattr(self.channel, name)


class RMBenchQposBackend:
    def __init__(self, env, observation, *, encode, policy_fingerprint, canonicalization):
        if env.task_name not in ("rearrange_blocks", "battery_try"):
            raise ValueError("only the two inspected task implementations are admitted")
        if type(env.scene.physx_system).__name__ != "PhysxCpuSystem":
            raise ValueError("only CPU PhysX pack/unpack is admitted")
        if env.eval_video_path is not None or env.save_data or env.render_freq:
            raise ValueError("disable video, data writes and interactive rendering for branches")
        if (canonicalization.get("status") not in ("cpu_bitwise_unchanged", "native_checkpoint_fixed_point") or
                not canonicalization.get("before_action_sealing") or
                canonicalization["cpu_after_sha256"] != hashlib.sha256(env.scene.physx_system.pack()).hexdigest() or
                canonicalization["poses_after_sha256"] != hashlib.sha256(env.scene.pack_poses()).hexdigest()):
            raise ValueError("canonicalize and seal the current query before constructing branches")
        self.env, self.obs = env, deepcopy(observation)
        self.encode, self.policy_fingerprint = encode, policy_fingerprint
        self.channels = {}
        self.topp_methods, self.topp_errors = {}, []
        if env.robot.communication_flag:
            for name in ("left_conn", "right_conn"):
                self.channels[name] = getattr(env.robot, name)
                setattr(env.robot, name, _IdleChannel(self.channels[name]))
        for name in ("left_mplib_planner", "right_mplib_planner"):
            planner = getattr(env.robot, name)
            original = planner.TOPP
            self.topp_methods[name] = (original, "TOPP" in vars(planner))
            def checked_topp(*args, _original=original, _name=name, **kwargs):
                try:
                    result = _original(*args, **kwargs)
                    if result[1].shape[0] == 0 and not exact_hold_path(args[0]):
                        self.topp_errors.append(_name + ": empty trajectory")
                    return result
                except Exception as exc:
                    self.topp_errors.append(f"{_name}: {type(exc).__name__}: {exc}")
                    raise
            planner.TOPP = checked_topp
        self.last_outcome = None

    def close(self):
        for name, channel in self.channels.items():
            setattr(self.env.robot, name, channel)
        self.channels = {}
        for name, (method, was_instance_field) in self.topp_methods.items():
            planner = getattr(self.env.robot, name)
            if was_instance_field:
                planner.TOPP = method
            else:
                del planner.TOPP
        self.topp_methods = {}

    def _physics(self):
        return {"cpu": self.env.scene.physx_system.pack(),
                "poses": deepcopy(self.env.scene.pack_poses())}

    def _controller(self):
        articulations = articulation_state(self.env)
        planners = {name: python_state(getattr(self.env.robot, name))
                    for name in ("left_mplib_planner", "right_mplib_planner")}
        return {"robot": python_state(self.env.robot), "articulations": articulations,
                "topp": planners, "curobo": "RPC forbidden; idle subprocess state is never consumed"}

    def _rng(self):
        return rng_state()

    def snapshot(self):
        return {"physics": self._physics(), "controller": self._controller(),
                "task": python_state(self.env), "rng": self._rng(),
                "observation": {"cached": deepcopy(self.obs), "cameras": python_state(self.env.cameras)},
                "policy": self.policy_fingerprint()}

    def fingerprint(self):
        return {name: digest(value) for name, value in self.snapshot().items()}

    def restore(self, state):
        import torch
        restore_python(self.env, state["task"])
        restore_python(self.env.robot, state["controller"]["robot"])
        for name, values in state["controller"]["topp"].items():
            restore_python(getattr(self.env.robot, name), values)
        arts = self.env.scene.get_all_articulations()
        saved = state["controller"]["articulations"]
        if len(arts) != len(saved):
            raise RuntimeError("articulation topology changed")
        for art, values in zip(arts, saved, strict=True):
            joints = art.get_active_joints()
            if art.name != values["name"] or len(joints) != len(values["joints"]):
                raise RuntimeError("joint topology changed")
            art.set_qf(values["qf"])
            for joint, (target, velocity) in zip(joints, values["joints"], strict=True):
                joint.set_drive_target(target)
                joint.set_drive_velocity_target(velocity)
        self.env.scene.physx_system.unpack(state["physics"]["cpu"])
        restore_python(self.env.cameras, state["observation"]["cameras"])
        # Recompute the canonical pose cache. Replaying ALL entity poses can
        # renormalize free articulation roots and change native physical bytes.
        self.env._update_render()
        self.obs = deepcopy(state["observation"]["cached"])
        rng = state["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(torch.from_numpy(rng["torch"].copy()))
        torch.cuda.set_rng_state_all([torch.from_numpy(v.copy()) for v in rng["cuda"]])
        self.last_outcome = None
        if self.policy_fingerprint() != state["policy"]:
            raise RuntimeError("policy history/cursor changed inside branch")

    def effect_tokens(self):
        return self.encode(self.obs)

    def step(self, command):
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (14,) or not np.isfinite(command).all():
            raise ValueError("finite absolute 14-dimensional qpos command required")
        before = self.env.take_action_cnt
        self.topp_errors = []
        self.env.take_action(command.copy(), action_type="qpos")
        if self.topp_errors:
            raise InvalidBranchExecution("official TOPP failed inside branch: " + str(self.topp_errors))
        executed = self.env.take_action_cnt == before + 1
        self.obs = self.env.get_obs()
        self.last_outcome = {"success": bool(self.env.eval_success), "executed": executed}
        end = "success" if self.env.eval_success else "step_limit" if not executed else None
        # This is the command passed to the official high-level executor. Its
        # TOPP/gripper controller still determines the low-level trajectory.
        return StepResult(executed, command.copy(), end, False)


def qualify_reset_probe(backend, *, horizon=32, perturbation=.01, context="reset", emit=lambda record: None):
    """Prespecified exact A-A-B-A check at one inspected state, never paper-ready.

    Exact physical/controller/task endpoints and uint8 RGB are required. A
    failure is retained; this function does not fit tolerances or weaken tests.
    The caller must independently establish the context. A pass leaves other
    task stages, transitions and actual policy-memory integration open.
    """
    if context not in ("reset", "expert_grasp"):
        raise ValueError("unsupported engineering qualification context")
    state, parent = backend.snapshot(), backend.fingerprint()
    initial = np.asarray(backend.obs["joint_action"]["vector"], dtype=np.float32)
    a = np.repeat(initial[None, :], horizon, axis=0)
    # Exercise actual TOPP trajectories on both arms. Exact holds legitimately
    # have zero trajectory duration in the official executor.
    a[:, 0] += np.where(np.arange(horizon) % 2 == 0, .005, -.005)
    a[:, 7] -= np.where(np.arange(horizon) % 2 == 0, .005, -.005)
    b = a.copy(); b[:, 0] += perturbation
    trials = []
    try:
        emit({"kind": "initial_parent", "expected": parent, "state": state})
        for commands in (a, a, b, a):
            backend.restore(state)
            observed = backend.fingerprint()
            emit({"kind": "parent_restore", "expected": parent, "observed": observed,
                  "state": backend.snapshot()})
            if observed != parent:
                raise RuntimeError("exact parent restoration failed")
            steps = 0
            for command in commands:
                outcome = backend.step(command)
                steps += int(outcome.executed)
                if outcome.termination:
                    break
            trials.append({"state": backend.snapshot(), "obs": deepcopy(backend.obs)})
            emit({"kind": "endpoint", "trial": len(trials)-1, "steps": steps, **trials[-1]})
            if steps != horizon:
                raise RuntimeError("reset qualification did not execute the complete horizon")
        if digest(trials[0]["state"]["physics"]) == digest(trials[2]["state"]["physics"]):
            raise RuntimeError("B did not exercise a distinct physical endpoint")
        for i in (1, 3):
            for key in ("physics", "controller", "task"):
                if digest(trials[0]["state"][key]) != digest(trials[i]["state"][key]):
                    raise RuntimeError(f"exact A-A/B-A {key} endpoint differs")
            for camera, image in trials[0]["obs"]["observation"].items():
                if "rgb" in image and not np.array_equal(image["rgb"], trials[i]["obs"]["observation"][camera]["rgb"]):
                    raise RuntimeError(f"exact A-A/B-A RGB differs: {camera}")
        return {"status": f"passed_{context}_probe_only", "order": "A-A-B-A", "horizon": horizon,
                "physical_and_RGB_exact": True, "full_scientific_qualification": False}
    finally:
        backend.restore(state)
        observed = backend.fingerprint()
        emit({"kind": "final_restore", "expected": parent, "observed": observed,
              "state": backend.snapshot()})
        if observed != parent:
            raise RuntimeError("final exact parent restoration failed")
