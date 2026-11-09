"""Read-only capture fixtures; actual model/branch qualification is separate."""
from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace as NS
import numpy as np
import pytest

from fastwam.research.query_capture import capture_online_replan,clone_inference_context,require_original_generation
from fastwam.research.action_witness import seal_generated_action


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


def sealed_query(tmp_path):
    actions=np.zeros((32,14),dtype=np.float32)
    step=NS(step_sha256='bound',candidate_payload_sha256='retrieval',model_input_sha256='observation',derived_seed=17)
    model=NS(infer_action=lambda **kwargs:{'action':actions})
    with seal_generated_action(model,tmp_path/'original',{}):model.infer_action(online_step=step)
    generation=json.loads((tmp_path/'original/generation.json').read_text())
    file_sha=hashlib.sha256((tmp_path/'original/model_actions.npy').read_bytes()).hexdigest()
    return {'online_step':step,'action':actions.copy()},generation,actions,file_sha


def test_original_binding_checks_the_entire_unexecuted_H32_and_its_seal(tmp_path):
    capture,generation,actions,file_sha=sealed_query(tmp_path)
    require_original_generation(capture,generation,actions,original_actions_sha256=file_sha)
    capture['action'][-1,-1]=np.nextafter(np.float32(0),np.float32(1))
    with pytest.raises(ValueError,match='H32 differs'):
        require_original_generation(capture,generation,actions,original_actions_sha256=file_sha)


@pytest.mark.parametrize('key,value',[
    ('bound_step_sha256','other'),('candidate_payload_sha256','other'),
    ('model_input_sha256','other'),('derived_seed',18),
    ('before_any_execution_of_this_chunk',False),('array_sha256','other'),
    ('shape',[4,14]),('dtype','float64')])
def test_original_binding_rejects_context_or_artifact_substitution(tmp_path,key,value):
    capture,generation,actions,file_sha=sealed_query(tmp_path);generation[key]=value
    with pytest.raises(ValueError):require_original_generation(capture,generation,actions,original_actions_sha256=file_sha)


@pytest.mark.parametrize('replacement',[np.zeros((4,14),dtype=np.float32),np.zeros((32,14),dtype=np.float64)])
def test_original_binding_rejects_short_prefix_or_different_precision(tmp_path,replacement):
    capture,generation,actions,file_sha=sealed_query(tmp_path);capture['action']=replacement
    with pytest.raises(ValueError,match='H32 differs'):require_original_generation(capture,generation,actions,original_actions_sha256=file_sha)


def test_generation_contract_hashes_the_npy_file_including_its_header(tmp_path):
    from fastwam.memory.manifest import sha256_array
    capture,generation,actions,file_sha=sealed_query(tmp_path)
    assert generation['array_sha256']==file_sha
    assert file_sha!=sha256_array(actions)
    with pytest.raises(ValueError,match='sealed metadata'):
        require_original_generation(capture,generation,actions,original_actions_sha256=sha256_array(actions))
