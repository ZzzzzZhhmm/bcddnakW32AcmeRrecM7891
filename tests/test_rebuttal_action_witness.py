import json
from types import SimpleNamespace
import numpy as np
import pytest
from fastwam.research.action_witness import seal_generated_action


def test_seals_exact_output_before_consumer_and_restores_method(tmp_path):
    action=np.arange(6,dtype=np.float32).reshape(3,2)
    class Model:
        def infer_action(self,**kwargs):return {'action':action}
    model=Model();step=SimpleNamespace(step_sha256='a'*64,model_input_sha256='b'*64,candidate_payload_sha256='c'*64,derived_seed=7)
    with seal_generated_action(model,tmp_path/'witness',{'role':'full'}) as record:
        output=model.infer_action(online_step=step)
        assert output['action'] is action and record['output_sealed']
        assert np.array_equal(np.load(tmp_path/'witness/model_actions.npy'),action)
    assert 'infer_action' not in vars(model)
    assert json.loads((tmp_path/'witness/generation.json').read_text())['before_consumer_validation']


def test_preserves_consumer_failure_and_out_of_range_values_without_clipping(tmp_path):
    class Model:
        def infer_action(self,**kwargs):return {'action':np.array([[5.25,-6.]],dtype=np.float32)}
    model=Model()
    with pytest.raises(ValueError,match='normalizer guard'):
        with seal_generated_action(model,tmp_path/'witness',{}):
            model.infer_action();raise ValueError('normalizer guard')
    assert np.array_equal(np.load(tmp_path/'witness/model_actions.npy'),[[5.25,-6.]])
    assert json.loads((tmp_path/'witness/consumer_status.json').read_text())['consumer_completed'] is False
    assert 'infer_action' not in vars(model)


def test_producer_errors_and_nonfinite_outputs_remain_observable(tmp_path):
    class Model:
        def infer_action(self,**kwargs):raise RuntimeError('producer failed')
    model=Model()
    with pytest.raises(RuntimeError,match='producer failed'):
        with seal_generated_action(model,tmp_path/'error',{}):model.infer_action()
    assert json.loads((tmp_path/'error/consumer_status.json').read_text())['output_sealed'] is False
    assert 'infer_action' not in vars(model)


def test_missing_call_is_rejected(tmp_path):
    model=SimpleNamespace(infer_action=lambda:None)
    with pytest.raises(RuntimeError,match='missing generated'):
        with seal_generated_action(model,tmp_path/'missing',{}):pass


def test_nonfinite_output_is_saved_and_never_changed(tmp_path):
    class Model:
        def infer_action(self,**kwargs):return {'action':np.array([[np.nan,np.inf]],dtype=np.float32)}
    model=Model()
    with seal_generated_action(model,tmp_path/'nonfinite',{}):model.infer_action()
    metadata=json.loads((tmp_path/'nonfinite/generation.json').read_text())
    assert not metadata['all_finite'] and metadata['minimum'] is None
    assert np.isnan(np.load(tmp_path/'nonfinite/model_actions.npy')[0,0])


def test_torch_output_preserves_tensor_identity_values_and_gradient_state(tmp_path):
    torch=pytest.importorskip('torch')
    value=torch.arange(6,dtype=torch.float32).reshape(3,2).requires_grad_()
    model=SimpleNamespace(infer_action=lambda **kwargs:{'action':value})
    version=value._version
    with seal_generated_action(model,tmp_path/'torch',{}):output=model.infer_action()
    assert output['action'] is value and value.grad is None and value._version==version
    assert np.array_equal(np.load(tmp_path/'torch/model_actions.npy'),value.detach().numpy())
