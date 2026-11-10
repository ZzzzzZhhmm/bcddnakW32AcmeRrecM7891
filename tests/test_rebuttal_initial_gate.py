import numpy as np
import pytest

from fastwam.memory.manifest import sha256_array
from fastwam.research.initial_gate import initial_observation,require_empty_initial_history,gate_measurement


def test_saved_initial_observation_preserves_original_arrays_and_rejects_future_fields(tmp_path):
    values={k:np.zeros((2,3,3),np.uint8) for k in ('head_camera','left_camera','right_camera')}
    values['native_qpos']=np.arange(14,dtype=np.float64)
    path=tmp_path/'obs.npz';np.savez(path,**values)
    hashes={k:sha256_array(v) for k,v in values.items()}
    result=initial_observation(path,hashes)
    assert result['joint_action']['vector'].dtype==np.float64
    assert np.array_equal(result['joint_action']['vector'],values['native_qpos'])
    np.savez(path,**values,target_action=np.zeros((32,14)))
    with pytest.raises(ValueError,match='current'):initial_observation(path,hashes)


def test_changed_initial_observation_is_rejected(tmp_path):
    values={k:np.zeros((2,3,3),np.uint8) for k in ('head_camera','left_camera','right_camera')}
    values['native_qpos']=np.zeros(14,np.float32)
    path=tmp_path/'obs.npz';np.savez(path,**values)
    with pytest.raises(ValueError,match='hashes'):initial_observation(path,{})


@pytest.mark.parametrize('kwargs,evidence',[({'history':[]},None),({},{}),({}, {'prior':1})])
def test_history_is_native_empty_not_replaced_with_zeros(kwargs,evidence):
    with pytest.raises(ValueError,match='empty'):require_empty_initial_history(kwargs,evidence)


def probe():
    return dict(force_null=False,comparison=True,source_mode='full',gate_threshold=.05,
        arrays=dict(valid=np.ones((1,4),bool),alpha=np.array([.2]),g=np.array([.2]),
            source_gate=np.array([.2]),selected_index=np.array([0]),v_det=np.array([True])))


def measure(value,rows=None):
    return gate_measurement(value,np.array([0,1,2,3]) if rows is None else rows,
        np.arange(4),np.array([0,0,1,1,1]),['a','b'],slots=4)


def test_positive_gate_does_not_fabricate_false_activation_or_skill_coverage():
    result=measure(probe())
    assert result['positive_effective_gate'] and result['independently_measured_relevant_coverage'] is None
    assert result['task_success'] is None and 'false_activation' not in result
    assert result['candidate_task_counts']=={'a':2,'b':2}


@pytest.mark.parametrize('change',[dict(force_null=True),dict(comparison=False),dict(source_mode='gaussian')])
def test_forced_or_ablated_controls_are_not_natural_gate_measurements(change):
    value=probe();value.update(change)
    with pytest.raises(ValueError):measure(value)


def test_search_universe_escape_is_rejected_even_for_invalid_slot():
    value=probe();value['arrays']['valid'][0,3]=False
    with pytest.raises(ValueError,match='escaped'):measure(value,np.array([0,1,2,4]))


@pytest.mark.parametrize('value',[np.nan,-.1,1.1])
def test_nonfinite_or_out_of_range_gates_are_rejected(value):
    record=probe();record['arrays']['g']=np.array([value])
    with pytest.raises(ValueError,match='scalar gate'):measure(record)


def test_hard_rejection_is_preserved_without_a_claim_of_relevance():
    value=probe();value['arrays'].update(v_det=np.array([False]),g=np.array([0.]),source_gate=np.array([0.]))
    result=measure(value)
    assert not result['positive_effective_gate'] and not result['deterministic_eligibility']
    value['arrays']['g']=np.array([.1])
    with pytest.raises(ValueError,match='disagree'):measure(value)
