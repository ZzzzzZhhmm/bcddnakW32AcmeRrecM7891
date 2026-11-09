import hashlib
import json

import pytest

from fastwam.research.factual_effect_reader import bind_reader_files
from fastwam.research.fresh_pool import next_branch, read_verified_branch, seal_branch


def test_effect_reader_rejects_changed_original_artifacts_before_heavy_loading(tmp_path):
    runtime={};contract={}
    for field,other in [('warm_encoder_contract_path','encoder_contract_sha256'),('warm_camera_contract_path','camera_contract_sha256'),
            ('warm_m1_data_config_path','m1_data_config_sha256'),('dataset_stats_path','normalization_stats_sha256')]:
        path=tmp_path/field;path.write_text('{}');runtime[field]=str(path);contract[other]=hashlib.sha256(b'{}').hexdigest()
    assert len(bind_reader_files(runtime,contract))==4
    (tmp_path/'warm_m1_data_config_path').write_text('{"changed":1}')
    with pytest.raises(ValueError,match='m1_data'):bind_reader_files(runtime,contract)


def test_complete_record_protects_real_raw_endpoint_and_interrupted_attempt_has_no_receipt(tmp_path):
    endpoint=tmp_path/'endpoint.pickle';endpoint.write_bytes(b'native endpoint')
    result=dict(query_id='q8',branch_id='rank-0',parent_recreated=True,endpoint_status='complete_horizon',
        executed_steps=32,observed_effect=[.1,.2],independent_label={'applicable':0})
    seal_branch(tmp_path,result)
    assert read_verified_branch(tmp_path,query_id='q8',branch_id='rank-0')==result
    endpoint.write_bytes(b'altered endpoint')
    with pytest.raises(ValueError,match='changed'):read_verified_branch(tmp_path,query_id='q8',branch_id='rank-0')
    with pytest.raises(ValueError,match='identity'):read_verified_branch(tmp_path,query_id='q48',branch_id='rank-0')


def test_invalid_effects_and_labels_are_not_fabricated(tmp_path):
    result=dict(query_id='q8',branch_id='rank-0',parent_recreated=True,endpoint_status='invalid_branch',
        executed_steps=3,observed_effect=None,independent_label={'applicable':None})
    seal_branch(tmp_path,result)
    for fields in ({'observed_effect':[0.]},{'independent_label':{'applicable':0}},{'parent_recreated':False}):
        with pytest.raises(ValueError):seal_branch(tmp_path,{**result,**fields})


def test_fixed_sequence_does_not_skip_rejection_or_pick_better_rank():
    sequence=['anchor-a','anchor-aa','rank-0','rank-1','anchor-final']
    assert next_branch(sequence,{'anchor-a','anchor-aa','rank-0'})=='rank-1'
    with pytest.raises(ValueError):next_branch(sequence,{'rank-32'})
