from copy import deepcopy
import pytest
from fastwam.research.training_environment import compare_static_training_environment


def example():
    return dict(python_version='3.10.20', platform='Linux-kernel-a-with-glibc2.35',
                accelerate_version='1.12.0', deepspeed_version='0.18.5',
                torch_version='2.7.1+cu128', torch_cuda_version='12.8', cudnn_version=90701,
                cuda_available=True, gpu_count=1,
                gpu_devices=[dict(index=0, name='NVIDIA H100 80GB HBM3', capability=[9, 0])])


def test_match_is_not_full_resume_admission():
    actual=example();parent=dict(actual,deepspeed_config={'zero_optimization':{'stage':1}})
    result=compare_static_training_environment(parent,actual)
    assert not result['differences']
    assert not result['full_numerical_runtime_verified']
    assert not result['formal_resume_validation_bypassed']
    assert result['optimizer_updates']==0 and not result['model_loaded']


def test_platform_difference_is_explicit_and_not_silently_allowed():
    parent=example();actual=deepcopy(parent);actual['platform']='Linux-kernel-b-with-glibc2.35'
    result=compare_static_training_environment(parent,actual)
    assert result['status']=='static_environment_differs_from_parent'
    assert result['differences']==[dict(field='platform',parent=parent['platform'],observed=actual['platform'])]


def test_gpu_and_software_changes_are_not_hidden_by_platform_difference():
    parent=example();actual=deepcopy(parent)
    actual['platform']='another-kernel';actual['gpu_devices'][0]['name']='another-GPU'
    actual['torch_version']='another-version'
    result=compare_static_training_environment(parent,actual)
    assert {r['field'] for r in result['differences']}=={'platform','gpu_devices','torch_version'}


@pytest.mark.parametrize('mutation',[lambda x:x.pop('platform'),lambda x:x.update(ignored_override=True),
    lambda x:x.update(gpu_count=2),lambda x:x.update(cuda_available=1),
    lambda x:x['gpu_devices'][0].update(capability=[9])])
def test_malformed_observed_runtime_is_rejected(mutation):
    parent=example();actual=deepcopy(parent);mutation(actual)
    with pytest.raises(ValueError):compare_static_training_environment(parent,actual)
