"""Prevent unpaired or mismatched workers from creating apparent paired gains."""
from copy import deepcopy
import pytest
from scripts.merge_rebuttal_role_trials import merge_records


def records():
    result=[]
    for role,success in [('full',1),('candidate_null',0)]:
        identity=dict(core_source_sha256='c',collector_source_sha256='d',checkpoint_sha256='w',
            plan_sha256='p',search_condition='full_bank',task='a',phase='formal')
        row=dict(task='a',episode_id=0,seed=1,phase='formal',checkpoint_sha256='w',plan_sha256='p',
            search_condition='full_bank',roles={role:{'status':'complete','success':success}})
        result.append(dict(row=row,identity=identity,initial_identity={'camera':'same'},worker_evidence=role))
    return result


def test_pair_requires_both_roles_and_matching_initial_truth():
    values=records();pair,=merge_records(values)
    assert pair['status']=='complete' and pair['full_minus_null']==1
    partial,=merge_records(values[:1])
    assert partial['status']=='incomplete_pair' and partial['full_minus_null'] is None
    values[1]['initial_identity']['camera']='different'
    with pytest.raises(ValueError,match='camera'):merge_records(values)


def test_model_drift_and_duplicate_attempts_are_never_silently_pooled():
    values=records();values[1]['identity']['core_source_sha256']='other'
    with pytest.raises(ValueError,match='mismatch'):merge_records(values)
    values=records()
    with pytest.raises(ValueError,match='duplicate'):merge_records(values+[deepcopy(values[0])])
