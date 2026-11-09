from types import SimpleNamespace
import numpy as np
import pytest
from fastwam.research.native_replay import require_observation,portable_python_state,require_trace_step,physical_joint_state


def test_observation_rejects_changed_pixel_missing_camera_or_coerced_dtype():
    source={'camera':np.zeros((2,2,3),dtype=np.uint8),'native_qpos':np.zeros(14,dtype=np.float32)}
    require_observation(source,{k:v.copy() for k,v in source.items()})
    changed={k:v.copy() for k,v in source.items()};changed['camera'][0,0,0]=1
    with pytest.raises(ValueError,match='camera'):require_observation(source,changed)
    with pytest.raises(ValueError,match='channels'):require_observation(source,{'native_qpos':source['native_qpos']})
    with pytest.raises(ValueError,match='native_qpos'):require_observation(source,{**source,'native_qpos':source['native_qpos'].astype('float64')})


def test_cross_scene_handles_remain_explicitly_typed_and_value_state_is_owned():
    obj=SimpleNamespace(controller=object(),counter=3,qpos=np.array([1.,2.]))
    state=portable_python_state(obj);obj.qpos[0]=8;obj.counter=9
    assert state['values']['counter']==3 and state['values']['qpos'][0]==1
    assert state['opaque_types']=={'controller':('builtins','object')}


def test_replay_rejects_a_single_native_state_ulp():
    trace={'before':np.ones((1,14),dtype=np.float32),'after':np.ones((1,14),dtype=np.float32)}
    require_trace_step(trace,0,trace['before'][0],trace['after'][0])
    changed=trace['after'][0].copy();changed[3]=np.nextafter(changed[3],np.float32(2))
    with pytest.raises(ValueError,match='command 0'):require_trace_step(trace,0,trace['before'][0],changed)


def test_physical_reader_uses_articulation_coordinates_and_owns_arrays():
    measured=np.array([.125,.25]);velocity=np.array([.01,.02])
    art=SimpleNamespace(name='robot',get_qpos=lambda:measured,get_qvel=lambda:velocity)
    env=SimpleNamespace(scene=SimpleNamespace(get_all_articulations=lambda:[art]))
    state=physical_joint_state(env);measured[0]=9;velocity[0]=8
    assert state[0]['qpos'][0]==.125 and state[0]['qvel'][0]==.01
