"""Bank-intervention bookkeeping tests do not establish absent skill coverage."""
import numpy as np
import pytest
from fastwam.research.repertoire import matched_search_rows,restricted_cosine_search


def test_size_matched_holdout_is_nonempty_and_has_no_target_family():
    vocabulary=['rearrange_blocks','put_back_block','press_button','battery_try']
    indices=np.repeat(np.arange(4),[9,7,8,6])
    rows,report=matched_search_rows(indices,vocabulary,'rearrange_blocks')
    clean,held=rows.values()
    assert len(clean)==len(held)==14 and len(np.unique(clean))==14
    assert not np.isin(indices[held],[0,1]).any()
    assert np.sum(indices[clean]==0)==9
    assert not clean.flags.writeable and not held.flags.writeable
    assert 'independent execution labels' in report['actual_absent_skill_claim']
    repeat,_=matched_search_rows(indices,vocabulary,'rearrange_blocks')
    assert np.array_equal(repeat['clean_size_matched'],clean)


def test_restricted_search_cannot_leak_heldout_continuation_and_has_stable_ties():
    keys=np.array([[1.,0.],[1.,0.],[0.,1.],[-1.,0.]])
    rows,scores=restricted_cosine_search(keys,[1.,0.],np.array([3,2,1]),top_k=2,continuation_rows=(0,3))
    assert rows.tolist()==[1,3] and np.allclose(scores,[1,-1])
    rows,_=restricted_cosine_search(keys,[1.,0.],np.array([1,0,2,3]),top_k=2)
    assert rows.tolist()==[0,1]


@pytest.mark.parametrize('rows,k',[(np.array([0,0]),2),(np.array([-1,1]),2),(np.array([0]),2)])
def test_invalid_or_undersized_search_universe_is_rejected(rows,k):
    with pytest.raises(ValueError):restricted_cosine_search(np.eye(3),[1.,0.,0.],rows,top_k=k)
