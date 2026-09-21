import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('training_update',Path(__file__).resolve().parents[1]/'scripts/diagnose_nonreal_training_update.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_small_update_is_retained_in_fp32_without_mutating_original():
    torch=pytest.importorskip('torch')
    state={'bias':torch.tensor([-2.],dtype=torch.bfloat16)}
    gradient={'bias':torch.tensor([.1],dtype=torch.bfloat16)}
    result=module.compare_updates(state,gradient,learning_rate=1e-5,weight_decay=.01)
    assert result['bf16_direct']['bias']['changed_elements']==0
    assert result['fp32_master']['bias']['changed_elements']==1
    assert result['fp32_master']['bias']['changed_after_bf16_cast']==0
    assert state['bias'].item()==-2


def test_saved_master_slice_crosses_rank_boundary_without_full_concat():
    torch=pytest.importorskip('torch')
    other_spec=importlib.util.spec_from_file_location('zero_gate',Path(__file__).resolve().parents[1]/'scripts/inspect_nonreal_zero_gate.py')
    other=importlib.util.module_from_spec(other_spec)
    other_spec.loader.exec_module(other)
    partitions=[torch.arange(4),torch.arange(4,8)]
    assert torch.equal(other.slice_partitions(partitions,3,3),torch.tensor([3,4,5]))
    with pytest.raises(ValueError,match='exceeds'):
        other.slice_partitions(partitions,7,2)


def test_training_probe_disables_native_autocast_like_deepspeed():
    torch=pytest.importorskip('torch')

    class Model:
        device='cpu'

        def training_loss(self, sample):
            assert not torch.is_autocast_enabled('cpu')
            return sample

    with torch.autocast('cpu',dtype=torch.bfloat16):
        assert torch.is_autocast_enabled('cpu')
        assert module.native_bf16_training_loss(Model(),42)==42
        assert torch.is_autocast_enabled('cpu')
