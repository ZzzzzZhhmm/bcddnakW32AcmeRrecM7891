"""Process-local software-EGL selection independent of CUDA visibility.

robosuite 1.4 assumes that an EGL device index is a physical CUDA index.
Mesa's software EGL device is a separate namespace. Keep CUDA_VISIBLE_DEVICES
unchanged and use MuJoCo's EGL-only selector for this software-rendering path.
No installed dependency file or trained model is changed.
"""
from __future__ import annotations

import importlib
import os


def install_software_egl(*, importer=importlib.import_module):
    if os.environ.get('LIBGL_ALWAYS_SOFTWARE') != '1' or os.environ.get('MUJOCO_GL') != 'egl':
        raise ValueError('LIBERO Table4 requires explicit Mesa software EGL')
    device = os.environ.get('WARM_TABLE4_EGL_DEVICE_ID', '0')
    if not device.isdigit():
        raise ValueError('WARM_TABLE4_EGL_DEVICE_ID must be a nonnegative EGL index')
    previous = os.environ.pop('MUJOCO_EGL_DEVICE_ID', None)
    try:
        # Avoid robosuite's import-time CUDA-membership assertion. CUDA's
        # visibility itself must never be removed or changed, even temporarily.
        context = importer('robosuite.renderers.context.egl_context')
        mujoco_egl = importer('mujoco.egl')
    except BaseException:
        if previous is not None:
            os.environ['MUJOCO_EGL_DEVICE_ID'] = previous
        raise
    os.environ['MUJOCO_EGL_DEVICE_ID'] = device

    def create_display(device_id=0):
        del device_id  # robosuite's CUDA-associated index is irrelevant to Mesa.
        return mujoco_egl.create_initialized_egl_device_display()

    context.create_initialized_egl_device_display = create_display
    return dict(backend='Mesa software EGL', egl_device=device,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                selector='mujoco.egl (independent of CUDA indices)')
