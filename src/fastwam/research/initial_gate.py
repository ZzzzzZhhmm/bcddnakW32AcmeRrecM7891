"""Recorded initial-state availability probes; no skill or success labels."""
from pathlib import Path
import numpy as np


def initial_observation(path, expected_hashes):
    from fastwam.memory.manifest import sha256_array
    with np.load(Path(path), allow_pickle=False) as data:
        values={name:data[name].copy() for name in data.files}
    permitted={'head_camera','left_camera','right_camera','front_camera','native_qpos'}
    required={'head_camera','left_camera','right_camera','native_qpos'}
    if not required<=set(values)<=permitted:
        raise ValueError('only the recorded current camera/joint observation is permitted')
    if {k:sha256_array(v) for k,v in values.items()}!=expected_hashes:
        raise ValueError('recorded initial observation hashes differ')
    joints=values['native_qpos']
    if joints.shape!=(14,) or joints.dtype.kind!='f' or not np.isfinite(joints).all():
        raise ValueError('finite original14D joint API input required')
    for name,value in values.items():
        if name!='native_qpos' and (value.dtype!=np.uint8 or value.ndim!=3 or value.shape[-1]!=3):
            raise ValueError('original uint8 RGB camera required')
    return dict(observation={k:{'rgb':v} for k,v in values.items() if k!='native_qpos'},
                joint_action={'vector':joints})


def require_empty_initial_history(kwargs, evidence):
    if kwargs or evidence is not None:
        raise ValueError('query0 must use native empty history, not a fabricated replacement')


def gate_measurement(probe, bank_rows, eligible_rows, task_indices, vocabulary, *, slots=32):
    if probe.get('force_null') is not False or probe.get('comparison') is not True:
        raise ValueError('natural full-comparison probe required')
    if probe.get('source_mode')!='full':
        raise ValueError('unmodified full source mode required')
    def array(value):
        if hasattr(value,'detach'):
            value=value.detach().cpu()
            if value.is_floating_point():value=value.float()
            value=value.numpy()
        return np.asarray(value)
    arrays={k:array(v) for k,v in probe['arrays'].items()}
    valid=arrays['valid'];rows=np.asarray(bank_rows);eligible=np.asarray(eligible_rows)
    if valid.dtype!=np.bool_ or valid.shape!=(1,slots) or rows.shape!=(slots,) or rows.dtype.kind not in 'iu':
        raise ValueError('exact deployed candidate slots and boolean mask required')
    if np.any(rows[valid[0]]<0) or not np.isin(rows[rows>=0],eligible).all():
        raise ValueError('retrieved row escaped the frozen eligible universe')
    task_indices=np.asarray(task_indices)
    if np.any(rows[rows>=0]>=len(task_indices)):
        raise ValueError('bank row outside original task provenance')
    result={}
    for name in ('alpha','g','source_gate'):
        value=arrays[name]
        if value.shape!=(1,) or not np.isfinite(value).all() or not 0<=value[0]<=1:
            raise ValueError('finite scalar gate in[0,1] required')
        result[name]=float(value[0])
    selection=arrays['selected_index'];vdet=arrays['v_det']
    if selection.shape!=(1,) or selection.dtype.kind not in 'iu' or not 0<=selection[0]<slots:
        raise ValueError('original selected rank required')
    if vdet.shape!=(1,) or vdet.dtype!=np.bool_ or (not vdet[0] and result['g']!=0):
        raise ValueError('deterministic rejection and effective gate disagree')
    selected=int(selection[0]);selected_row=int(rows[selected])
    result.update(valid_candidates=int(valid.sum()),candidate_slots=slots,selected_rank=selected,
        selected_bank_row=selected_row,selected_task=(str(vocabulary[int(task_indices[selected_row])]) if selected_row>=0 else None),
        candidate_task_counts={str(name):int(np.sum(task_indices[rows[valid[0]]]==i)) for i,name in enumerate(vocabulary)},
        deterministic_eligibility=bool(vdet[0]),positive_effective_gate=result['g']>0,
        configured_gate_threshold=float(probe['gate_threshold']),
        independently_measured_relevant_coverage=None,task_success=None,
        scope='Availability/gate response only. Positive gate is not a false-activation label without independent relevance truth.')
    return result
