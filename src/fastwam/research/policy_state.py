"""Read-only fingerprints of the actual deployed WARM history and cursors.

These fingerprints qualify that sealed simulator branches do not write policy
state. They are not a general policy-restoration API or a simulator certificate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json

import numpy as np


RETRIEVER_FIELDS = (
    "_active_episode_index", "_last_frame_index", "_seen_episode_indices",
    "_issued", "_inflight", "_consumed", "_delivered_step_digests",
    "_model_consumed", "_continuation_rows",
)
THREAD_FIELDS = ("_warm_thread_event", "_warm_thread_query_frame",
                 "_warm_thread_episode_index", "_warm_thread_null_steps")


def _plain(value):
    if isinstance(value, np.ndarray):
        return {"dtype":str(value.dtype), "shape":list(value.shape),
                "sha256":hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        pairs = [[_plain(k), _plain(v)] for k,v in value.items()]
        return {"mapping":sorted(pairs, key=lambda p:json.dumps(p[0],sort_keys=True))}
    if isinstance(value, (set, frozenset)):
        return {"set":sorted((_plain(v) for v in value),key=lambda v:json.dumps(v,sort_keys=True))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or type(value) in (bool, str, int, float):
        return value
    raise TypeError(f"unsupported policy-state field: {type(value).__name__}")


def digest_state(value):
    return hashlib.sha256(json.dumps(_plain(value),sort_keys=True,separators=(",",":"),
                                    allow_nan=False).encode()).hexdigest()


def deployed_policy_state(policy):
    controller = policy.controller
    memory = controller.retrospective_episode_memory
    history = None if memory is None or memory._capability is None else memory._memory.snapshot().sha256
    return {
        "history_snapshot_sha256":history,
        "controller":{name:deepcopy(getattr(controller,name)) for name in
            ("_active_episode_index", "_last_frame_index", "_seen_episode_indices",
             "_executed_actions_since_replan", "_executed_environment_actions_since_replan")},
        "retriever":{name:deepcopy(getattr(policy.retriever,name)) for name in RETRIEVER_FIELDS},
        "model_thread":{name:deepcopy(getattr(policy.model,name)) for name in THREAD_FIELDS},
        "queue":deepcopy(list(policy.queue._items)),
        "policy":{name:deepcopy(getattr(policy,name)) for name in
            ("_executed_policy_actions", "_next_episode_index", "_instruction", "_closed")},
    }


def deployed_policy_fingerprint(policy):
    return digest_state(deployed_policy_state(policy))
