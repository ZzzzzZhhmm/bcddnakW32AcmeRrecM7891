"""Fingerprint tests use bookkeeping fixtures, never simulated success labels."""
from dataclasses import dataclass
from types import SimpleNamespace as NS
import numpy as np
import pytest

from fastwam.research.policy_state import digest_state, deployed_policy_fingerprint, RETRIEVER_FIELDS, THREAD_FIELDS


def policy_fixture():
    controller=NS(retrospective_episode_memory=NS(_capability=None),_active_episode_index=0,
        _last_frame_index=4,_seen_episode_indices={0},_executed_actions_since_replan=[np.zeros(14)],
        _executed_environment_actions_since_replan=[np.ones(14)])
    retriever=NS(**{name:None for name in RETRIEVER_FIELDS})
    retriever._model_consumed={"query0"};retriever._continuation_rows=(2,)
    model=NS(**{name:None for name in THREAD_FIELDS})
    return NS(controller=controller,retriever=retriever,model=model,queue=NS(_items=[]),
              _executed_policy_actions=4,_next_episode_index=1,_instruction="task",_closed=False)


@pytest.mark.parametrize("field",["history","consumption","continuation","thread","queue","execution"])
def test_real_policy_state_components_cannot_change_silently(field):
    policy=policy_fixture();before=deployed_policy_fingerprint(policy)
    if field=="history":policy.controller._executed_actions_since_replan[0][0]=1
    if field=="consumption":policy.retriever._model_consumed.add("query1")
    if field=="continuation":policy.retriever._continuation_rows=(3,)
    if field=="thread":policy.model._warm_thread_null_steps=1
    if field=="queue":policy.queue._items.append(np.arange(14))
    if field=="execution":policy._executed_policy_actions+=1
    assert deployed_policy_fingerprint(policy)!=before


def test_fingerprint_is_order_independent_and_preserves_arrays():
    x=np.arange(4,dtype=np.float32)
    assert digest_state({"x":x,"seen":{2,1}})==digest_state({"seen":{1,2},"x":x.copy()})
    assert np.array_equal(x,np.arange(4,dtype=np.float32))
    with pytest.raises(TypeError,match="unsupported"):
        digest_state(object())
