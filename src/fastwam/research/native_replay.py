"""Strict observations and value-state identities across independently reset scenes.

Opaque Python handles are compared by declared type rather than process address;
this is not a certificate that their hidden state has been captured. Native CPU
physics, drive targets, task values, camera values and RNG remain exact checks.
"""
from __future__ import annotations
import numpy as np


def observation_arrays(observation):
    result={name:np.array(camera['rgb'],copy=True) for name,camera in observation['observation'].items() if 'rgb' in camera}
    result['native_qpos']=np.array(observation['joint_action']['vector'],copy=True)
    if len(result)<4:raise ValueError('head and two wrist cameras plus native qpos required')
    return result


def require_observation(reference,observed):
    if set(reference)!=set(observed):raise ValueError('observation channels differ')
    for key in reference:
        if reference[key].dtype!=observed[key].dtype or not np.array_equal(reference[key],observed[key]):
            raise ValueError('native replay observation differs: '+key)


def portable_python_state(obj):
    from .rmbench_branches import python_state
    state=python_state(obj)
    return dict(values=state['values'],opaque_types={key:value[:2] for key,value in state['opaque'].items()})


def native_parent(env,observation):
    from .rmbench_branches import articulation_state,rng_state
    return dict(physics=env.scene.physx_system.pack(),poses=env.scene.pack_poses(),
        task=portable_python_state(env),robot=portable_python_state(env.robot),
        cameras=portable_python_state(env.cameras),articulations=articulation_state(env),
        topp={name:portable_python_state(getattr(env.robot,name)) for name in ('left_mplib_planner','right_mplib_planner')},
        observations=observation_arrays(observation),rng=rng_state(),
        opaque_scope='Opaque handles are type-checked only. Planner RPC is not used by native qpos replay. No hidden-state completeness claim.')


def state_hashes(state):
    from .rmbench_branches import digest
    return {key:digest(value) for key,value in state.items()}


def require_trace_step(trace,index,before,after):
    if not np.array_equal(before,trace['before'][index]) or not np.array_equal(after,trace['after'][index]):
        raise ValueError('native command replay differs from the complete source at command '+str(index))
