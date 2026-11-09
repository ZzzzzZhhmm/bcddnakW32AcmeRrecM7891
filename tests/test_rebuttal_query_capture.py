"""Read-only capture fixtures; actual model/branch qualification is separate."""
from dataclasses import dataclass
from types import SimpleNamespace as NS
import numpy as np
import pytest

from fastwam.research.query_capture import capture_online_replan,clone_inference_context,require_original_generation
from fastwam.memory.manifest import sha256_array


@dataclass
class Context:
    candidate_actions:object
    target_effect:object=None


def test_context_is_an_independent_copy_and_never_contains_teacher_targets():
    original=Context(np.arange(4))
    copied=clone_inference_context(original);copied.candidate_actions[0]=99
    assert original.candidate_actions[0]==0
    with pytest.raises(ValueError,match='teacher'):
        clone_inference_context(Context(np.arange(4),target_effect=np.zeros(2)))


def test_capture_delegates_once_and_restores_the_original_methods():
    class Model:
        def _resolve_action_source(self,**kwargs):return 'source'
        def infer_action(self,**kwargs):
            self._resolve_action_source(action_source_context=Context(np.arange(3)))
            return {'action':np.zeros((32,14)),'warm_research_probe':{'arrays':{}}}
    model=Model();policy=NS(model=model)
    with capture_online_replan(policy) as record:model.infer_action(online_step='owned-step',num_inference_steps=10)
    assert record['source_calls']==record['infer_calls']==1
    assert record['online_step']=='owned-step' and record['sampler_options']=={'num_inference_steps':10}
    assert 'infer_action' not in vars(model) and '_resolve_action_source' not in vars(model)
    with pytest.raises(RuntimeError,match='exactly one'):
        with capture_online_replan(policy):pass
    assert 'infer_action' not in vars(model)


def test_capture_removes_wrappers_when_inference_raises():
    class Model:
        def infer_action(self,**kwargs):raise RuntimeError('executor failure')
        def _resolve_action_source(self,**kwargs):return None
    model=Model()
    with pytest.raises(RuntimeError,match='executor failure'):
        with capture_online_replan(NS(model=model)):model.infer_action(online_step='step')
    assert 'infer_action' not in vars(model) and '_resolve_action_source' not in vars(model)


def sealed_query():
    actions=np.zeros((32,14),dtype=np.float32)
    step=NS(step_sha256='bound',candidate_payload_sha256='retrieval',model_input_sha256='observation',derived_seed=17)
    generation=dict(before_any_execution_of_this_chunk=True,bound_step_sha256='bound',
        candidate_payload_sha256='retrieval',model_input_sha256='observation',derived_seed=17,
        shape=[32,14],dtype='float32',array_sha256=sha256_array(actions))
    return {'online_step':step,'action':actions.copy()},generation,actions


def test_original_binding_checks_the_entire_unexecuted_H32_and_its_seal():
    capture,generation,actions=sealed_query()
    require_original_generation(capture,generation,actions)
    capture['action'][-1,-1]=np.nextafter(np.float32(0),np.float32(1))
    with pytest.raises(ValueError,match='H32 differs'):
        require_original_generation(capture,generation,actions)


@pytest.mark.parametrize('key,value',[
    ('bound_step_sha256','other'),('candidate_payload_sha256','other'),
    ('model_input_sha256','other'),('derived_seed',18),
    ('before_any_execution_of_this_chunk',False),('array_sha256','other'),
    ('shape',[4,14]),('dtype','float64')])
def test_original_binding_rejects_context_or_artifact_substitution(key,value):
    capture,generation,actions=sealed_query();generation[key]=value
    with pytest.raises(ValueError):require_original_generation(capture,generation,actions)


@pytest.mark.parametrize('replacement',[np.zeros((4,14),dtype=np.float32),np.zeros((32,14),dtype=np.float64)])
def test_original_binding_rejects_short_prefix_or_different_precision(replacement):
    capture,generation,actions=sealed_query();capture['action']=replacement
    with pytest.raises(ValueError,match='H32 differs'):require_original_generation(capture,generation,actions)
