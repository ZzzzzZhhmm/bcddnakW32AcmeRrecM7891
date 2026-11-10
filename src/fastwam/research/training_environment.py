"""Read-only, pre-model environment diagnostics; never a resume admission."""
from __future__ import annotations

import importlib.metadata
import platform


STATIC_FIELDS = (
    'python_version', 'platform', 'accelerate_version', 'deepspeed_version',
    'torch_version', 'torch_cuda_version', 'cudnn_version', 'cuda_available',
    'gpu_count', 'gpu_devices',
)


def capture_static_training_environment():
    """Capture actual software/GPU descriptors without constructing a model."""
    import torch

    available = bool(torch.cuda.is_available())
    devices = []
    for index in range(torch.cuda.device_count() if available else 0):
        devices.append(dict(index=index, name=str(torch.cuda.get_device_name(index)),
                            capability=list(torch.cuda.get_device_capability(index))))
    return dict(
        python_version=platform.python_version(), platform=platform.platform(),
        accelerate_version=importlib.metadata.version('accelerate'),
        deepspeed_version=importlib.metadata.version('deepspeed'),
        torch_version=str(torch.__version__), torch_cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(), cuda_available=available,
        gpu_count=len(devices), gpu_devices=devices,
    )


def compare_static_training_environment(parent_runtime, observed):
    """Report every field difference without allowing any migration override."""
    if not isinstance(parent_runtime, dict) or not isinstance(observed, dict):
        raise TypeError('runtime descriptors must be dictionaries')
    if set(observed) != set(STATIC_FIELDS) or not set(STATIC_FIELDS) <= set(parent_runtime):
        raise ValueError('missing or unexpected static runtime field')
    if type(observed['cuda_available']) is not bool or type(observed['gpu_count']) is not int:
        raise ValueError('invalid CUDA availability/count types')
    devices = observed['gpu_devices']
    if not isinstance(devices, list) or len(devices) != observed['gpu_count']:
        raise ValueError('GPU inventory and count disagree')
    if observed['cuda_available'] != bool(devices):
        raise ValueError('CUDA availability and GPU inventory disagree')
    for index, device in enumerate(devices):
        if (not isinstance(device, dict) or set(device) != {'index', 'name', 'capability'}
                or type(device['index']) is not int or device['index'] != index
                or not isinstance(device['name'], str)
                or not isinstance(device['capability'], list) or len(device['capability']) != 2
                or any(type(x) is not int for x in device['capability'])):
            raise ValueError('invalid actual GPU descriptor')
    differences = [dict(field=key, parent=parent_runtime[key], observed=observed[key])
                   for key in STATIC_FIELDS if parent_runtime[key] != observed[key]]
    return dict(
        status=('static_environment_differs_from_parent' if differences else
                'static_environment_matches_parent_not_full_resume_admission'),
        differences=differences, checked_fields=list(STATIC_FIELDS),
        phase='pre_model_environment_only', model_loaded=False, optimizer_updates=0,
        formal_resume_validation_bypassed=False, full_numerical_runtime_verified=False,
        scope='Static software/GPU descriptors only. The live post-prepare numerical runtime and full optimizer/scheduler/RNG lineage must still pass the original strict validator.',
    )
