from copy import deepcopy
import numpy as np
import pytest
from fastwam.memory.robotwin_artifacts import RobotwinQposZScore
from fastwam.research.candidate_pool import decode_branch_actions,prepare_candidate_pool,centered_effect_metrics


def test_generated_branch_decode_preserves_unclipped_backward_commands():
    normalizer=RobotwinQposZScore(np.zeros(14,dtype=np.float32),np.ones(14,dtype=np.float32))
    actions=np.zeros((32,14),dtype=np.float32);actions[25,4]=5.25
    original=actions.copy();native=decode_branch_actions(normalizer,actions)
    assert np.array_equal(native,normalizer.denormalize(original)) and np.array_equal(actions,original)
    assert native[25,4]==5.25


def test_generated_decode_matches_the_original_processor_backward_map():
    torch=pytest.importorskip('torch')
    from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
    mean=np.linspace(-.3,.4,14,dtype=np.float32);std=np.linspace(.1,.9,14,dtype=np.float32)
    normalizer=RobotwinQposZScore(mean,std)
    original=SingleFieldLinearNormalizer({'mean':torch.from_numpy(mean),'std':torch.from_numpy(std)},mode='z-score')
    actions=np.linspace(-7,8,32*14,dtype=np.float32).reshape(32,14)
    native=decode_branch_actions(normalizer,actions)
    assert np.allclose(native,original.backward(torch.from_numpy(actions)).numpy(),rtol=1e-6,atol=1e-6)


def test_zero_std_decode_does_not_impose_float32_invertibility():
    normalizer=RobotwinQposZScore(np.ones(14,dtype=np.float32),np.zeros(14,dtype=np.float32))
    actions=np.full((32,14),5.25,dtype=np.float32)
    assert np.array_equal(decode_branch_actions(normalizer,actions),normalizer.denormalize(actions))


def test_pool_keeps_all32_ranks_and_exact_adapted_means():
    normalizer=RobotwinQposZScore(np.zeros(14,dtype=np.float32),np.ones(14,dtype=np.float32))
    actions=np.arange(32*32*14,dtype=np.float32).reshape(1,32,32,14)/1000
    valid=np.ones((1,32),dtype=bool);valid[0,7]=False
    commands,native,evidence=prepare_candidate_pool({'arrays':{'valid':valid,'adapted_actions':actions}},normalizer)
    assert len(commands)==31 and native.shape==(32,32,14) and 'rank-7' not in commands
    assert np.array_equal(commands['rank-31'],actions[0,31])
    assert evidence['candidate_slots']==32 and not evidence['native_commands_clipped']


def row(predicted,observed,historical=None):
    return {'candidates':[dict(candidate_id=f'rank-{i}',valid=True,endpoint_status='complete_horizon',executed_steps=32,horizon=32,
        predicted_effect=p,historical_effect=[0,0] if historical is None else historical[i],observed_effect=o)
        for i,(p,o) in enumerate(zip(predicted,observed,strict=True))]}


def test_centering_removes_query_offsets_and_exposes_constant_predictions():
    value=row([[10,11],[12,13]],[[0,1],[2,3]])
    report=centered_effect_metrics(value)
    assert report['references']['predicted']['centered_mse']==0
    assert report['references']['predicted']['centered_explained_variance']==1
    assert report['references']['predicted']['absolute_mse']==100
    assert report['references']['constant_query']['centered_explained_variance']==0
    value['candidates'][1]['predicted_effect']=[10,11]
    assert centered_effect_metrics(value)['references']['predicted']['centered_explained_variance']==0


def test_degenerate_truth_and_partial_endpoints_do_not_manufacture_information():
    value=row([[0,0],[1,1]],[[0,0],[0,0]])
    report=centered_effect_metrics(value)
    assert report['references']['predicted']['centered_explained_variance'] is None
    value['candidates'][1].update(endpoint_status='incomplete_horizon',executed_steps=16,observed_effect=None)
    report=centered_effect_metrics(value)
    assert report['complete_endpoints']==1 and report['references']=={}


def test_bad_shape_duplicate_or_imputed_horizon_is_rejected():
    value=row([[0,0],[1,1]],[[0,0],[1,1]])
    value['candidates'][1]['executed_steps']=16
    with pytest.raises(ValueError,match='H32'):centered_effect_metrics(value)
    value['candidates'][1]['executed_steps']=32;value['candidates'][1]['candidate_id']='rank-0'
    with pytest.raises(ValueError,match='duplicate'):centered_effect_metrics(value)
