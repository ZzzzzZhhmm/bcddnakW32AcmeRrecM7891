import os
from types import SimpleNamespace

import pytest

from fastwam.research.libero_egl import install_software_egl


@pytest.mark.parametrize('cuda', ['0','1','2','3','GPU-example'])
def test_software_egl_never_rewrites_cuda_visibility(monkeypatch, cuda):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', cuda)
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID', '0')
    monkeypatch.setenv('LIBGL_ALWAYS_SOFTWARE', '1')
    monkeypatch.setenv('MUJOCO_GL', 'egl')
    context = SimpleNamespace()
    def display():
        assert os.environ['MUJOCO_EGL_DEVICE_ID'] == '0'
        assert os.environ['CUDA_VISIBLE_DEVICES'] == cuda
        return 'software-display'
    def importer(name):
        assert 'MUJOCO_EGL_DEVICE_ID' not in os.environ
        assert os.environ['CUDA_VISIBLE_DEVICES'] == cuda
        return context if name.startswith('robosuite') else SimpleNamespace(create_initialized_egl_device_display=display)
    report = install_software_egl(importer=importer)
    assert report['egl_device'] == '0'
    assert context.create_initialized_egl_device_display(device_id=99) == 'software-display'


def test_failed_import_restores_environment(monkeypatch):
    monkeypatch.setenv('MUJOCO_EGL_DEVICE_ID','3')
    monkeypatch.setenv('LIBGL_ALWAYS_SOFTWARE','1')
    monkeypatch.setenv('MUJOCO_GL','egl')
    def fail(name):
        raise ImportError('missing dependency')
    with pytest.raises(ImportError):
        install_software_egl(importer=fail)
    assert os.environ['MUJOCO_EGL_DEVICE_ID'] == '3'
