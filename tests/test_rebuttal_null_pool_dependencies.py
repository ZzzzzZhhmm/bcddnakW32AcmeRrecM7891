import hashlib
import json

import pytest

from fastwam.research.fresh_pool import seal_branch
from fastwam.research.null_pool_dependencies import load_pool_dependencies


def create(pool, name, *, parent='same', endpoint='same', status='complete_horizon', attempt=1):
    path = pool / name / f'attempt-{attempt:03d}'
    path.mkdir(parents=True)
    record = dict(query_id='fixed-fresh-rearrange_blocks-e0000-q0048', branch_id=name,
        parent_recreated=True, parent_hashes={'native':parent}, endpoint_hashes={'native':endpoint},
        endpoint_status=status, executed_steps=32 if status=='complete_horizon' else 3,
        observed_effect=[0.1,0.2] if status=='complete_horizon' else None,
        independent_label={'applicable':0 if status=='complete_horizon' else None},
        sealed_query_sha256=hashlib.sha256((pool/'sealed_query.json').read_bytes()).hexdigest())
    seal_branch(path, record)
    return path


@pytest.fixture
def pool(tmp_path):
    (tmp_path/'sealed_query.json').write_text(json.dumps({'query_id':'fixed-fresh-rearrange_blocks-e0000-q0048'}))
    create(tmp_path,'anchor-a');create(tmp_path,'anchor-aa')
    return tmp_path


def load(pool, **kwargs):
    return load_pool_dependencies(pool,query_id='fixed-fresh-rearrange_blocks-e0000-q0048',selected_rank=29,**kwargs)


def test_pending_records_keep_unknown_raw_and_cannot_complete_query(pool):
    result=load(pool,allow_pending=True)
    assert result['raw'] is None and not result['complete']
    assert result['pending']==['rank-29','anchor-final']
    with pytest.raises(ValueError,match='missing'):load(pool)


def test_resolution_is_monotonic_and_strict_default_accepts_only_complete_dependencies(pool):
    create(pool,'rank-29')
    assert load(pool,allow_pending=True)['pending']==['anchor-final']
    create(pool,'anchor-final')
    assert load(pool)['complete'] and load(pool)['raw']['independent_label']['applicable']==0


def test_pending_mode_never_selects_among_duplicate_terminal_attempts(pool):
    create(pool,'rank-29');create(pool,'rank-29',attempt=2)
    with pytest.raises(ValueError,match='duplicate'):load(pool,allow_pending=True)


@pytest.mark.parametrize('name,fields',[
    ('rank-29',{'parent':'changed'}),('anchor-final',{'endpoint':'changed'}),
    ('anchor-final',{'status':'invalid_branch'}),
])
def test_existing_bad_records_never_bypass_verification_in_pending_mode(pool,name,fields):
    create(pool,name,**fields)
    with pytest.raises(ValueError,match='differs'):load(pool,allow_pending=True)


def test_terminal_receipt_corruption_stops_before_new_physics(pool):
    path=create(pool,'rank-29');(path/'result.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):load(pool,allow_pending=True)


def test_first_two_initial_anchors_remain_mandatory(tmp_path):
    (tmp_path/'sealed_query.json').write_text('{}');create(tmp_path,'anchor-a')
    with pytest.raises(ValueError,match='anchor-aa'):load(tmp_path,allow_pending=True)


def test_original_query_identity_is_checked_even_when_other_records_are_pending(pool):
    with pytest.raises(ValueError,match='identity'):
        load_pool_dependencies(pool,query_id='wrong',selected_rank=29,allow_pending=True)
