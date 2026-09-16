"""Teacher-forced offline inference harness. No controller or robot dependency.

A trusted backend is supplied by the model team after the real profile exists.
Importing that plugin executes Python: inspect it, and run on a host/container
without CAN access. This harness is not a security sandbox or a live executor.
"""
from __future__ import annotations

import importlib
import copy
import math
import time
from pathlib import Path

from .episodes import load_episode, read_json, require, vector, write_json


class MockBackend:
    """Format-only stand-in: zeros are not a safe movement command."""
    def reset(self, context):
        pass

    def observe_executed(self, command, successor):
        pass

    def predict(self, observation):
        return [[0.0] * 7 for _ in range(32)]

    def synchronize(self):
        pass


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered))-1)]


def policy_observation(obs):
    """Keep evaluator annotations in raw files out of the model API."""
    result = {k: obs[k] for k in ("seq", "timestamp_ns")}
    result["cameras"] = {view: {k: frame[k] for k in
                           ("path", "timestamp_ns", "width", "height", "color_space")}
                         for view, frame in obs["cameras"].items()}
    result["robot"] = {k: obs["robot"][k] for k in
                       ("timestamp_ns", "joint_position_rad", "tcp_position_m",
                        "tcp_quaternion_xyzw", "gripper_width_m")}
    return copy.deepcopy(result)


def policy_command(command):
    return copy.deepcopy({k: command[k] for k in
                         ("seq", "observation_seq", "timestamp_ns", "accepted",
                          "tcp_position_m", "tcp_quaternion_xyzw", "gripper_width_m")})


def replay(episode, output, *, mock=False, backend_spec=None, release=None, prefix=4):
    require(type(prefix) is int and 1 <= prefix <= 32, "prefix must be 1..32")
    require(not Path(output).exists(), "output exists; use a new file")
    meta, observations, commands, _ = load_episode(episode, allow_synthetic=mock)
    if mock:
        require(not backend_spec and not release, "mock cannot be combined with a real backend/release")
        backend = MockBackend()
        release_id = "MOCK_NOT_A_MODEL"
    else:
        require(bool(backend_spec) and bool(release), "real replay requires a trusted backend and release JSON")
        descriptor = read_json(release)
        require(descriptor["schema"] == "warm.real.shadow-release.v1", "unsupported release schema")
        require(descriptor["action_shape"] == [32, 7], "release must specify [32,7] actions")
        require(descriptor["action_space"] == "real_profile_normalized", "unknown model output space")
        require(descriptor["execution_prefix"] == prefix, "prefix differs from model release")
        require(isinstance(descriptor["release_id"], str) and descriptor["release_id"], "missing release id")
        module, sep, factory = backend_spec.partition(":")
        require(bool(sep) and bool(module) and bool(factory), "backend must be module:factory")
        backend = getattr(importlib.import_module(module), factory)(Path(release).resolve())
        release_id = descriptor["release_id"]
    # No outcome, split, hidden target or future observation is supplied as model
    # context. The trusted plugin must only use image_root to resolve image files.
    context = {k: meta[k] for k in ("task_id", "instruction", "camera_order", "control_frame", "tcp_frame")}
    context["image_root"] = str(Path(episode).resolve())
    backend.reset(context)
    records, latencies = [], []
    for i, obs in enumerate(observations):
        if i:
            backend.observe_executed(policy_command(commands[i-1]), policy_observation(obs))
        if i == len(commands) or i % prefix:
            continue
        backend.synchronize()  # Must synchronize CUDA for measured execution.
        start = time.perf_counter_ns()
        actions = backend.predict(policy_observation(obs))
        backend.synchronize()
        ms = (time.perf_counter_ns()-start)/1e6
        require(isinstance(actions, list) and len(actions) == 32, "prediction must contain 32 actions")
        for action in actions:
            vector(action, 7, "prediction")
        latencies.append(ms)
        records.append({"observation_seq": i, "latency_ms": ms, "actions": actions})
    result = {"schema": "warm.real.shadow-result.v1", "mock": mock, "release_id": release_id,
              "qualification": "format_only" if mock else "recorded_observation_inference_only",
              "teacher_forced": True, "robot_commands_sent_by_harness": False,
              "calls": len(records), "cold_call_ms": latencies[0],
              "latency_including_cold_ms": {"p50": percentile(latencies, .5),
                                           "p95": percentile(latencies, .95),
                                           "p99": percentile(latencies, .99)}, "predictions": records}
    write_json(output, result)
    return result
