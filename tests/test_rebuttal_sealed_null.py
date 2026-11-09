"""Input-contract fixtures only; native execution is a separate qualification."""
from copy import deepcopy
import hashlib,json
from types import SimpleNamespace
import numpy as np
import pytest
from fastwam.research.action_witness import seal_generated_action
from fastwam.research.sealed_null import require_null_binding,require_native_null_actions


def fixture(tmp_path):
    model=SimpleNamespace(infer_action=lambda **kw:{'action':np.zeros((32,14),dtype=np.float32)})
    step=SimpleNamespace(step_sha256='old',candidate_payload_sha256='payload',model_input_sha256='input',derived_seed=17)
    directory=tmp_path/'original'
    with seal_generated_action(model,directory,{}):model.infer_action(online_step=step)
    generation=json.loads((directory/'generation.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    args=dict(checkpoint_sha256='c',plan_sha256='p',core_source_sha256='core',runtime_source_sha256='runtime',
        generation_file_sha256=digest(directory/'generation.json'),model_actions_file_sha256=digest(directory/'model_actions.npy'),
        native_actions_file_sha256='native',context_file_sha256='context',telemetry_file_sha256='telemetry',task='t',seed=7,episode=0,query=8)
    seal=dict(checkpoint_sha256='c',plan_sha256='p',core_source_sha256='core',task='t',seed=7,episode_id=0,query_ordinal=8,frame_index=32,
        phase='formal',horizon=32,execution_prefix=4,original_generation_sha256=args['generation_file_sha256'],original_model_actions_sha256=args['model_actions_file_sha256'],
        native_actions_sha256='native',factual_context_sha256='context',original_telemetry_sha256='telemetry',simulator_source={'sha256':'runtime'},
        exact_original_full_H32=True,exact_original_query_RGB_and_drive_target=True,same_context_and_Gaussian=True,factual_rng_isolated=True,
        before_any_generated_null_execution=True,counterfactual_branches_executed=0,bound_step_sha256='actual',
        candidate_payload_sha256='payload',model_input_sha256='input',derived_seed=17,
        binding_comparison=dict(original_bound_step_sha256='old',comparison_bound_step_sha256='old',actual_bound_step_sha256='actual',actual_capability_unmodified=True))
    return seal,generation,args


def test_actual_generation_file_contract_can_bind_a_later_null_seal(tmp_path):
    seal,generation,args=fixture(tmp_path);require_null_binding(seal,generation,**args)


@pytest.mark.parametrize('field,value',[
    ('query_ordinal',24),('seed',8),('checkpoint_sha256','other'),('phase','qualification'),
    ('original_model_actions_sha256','other'),('native_actions_sha256','other'),('factual_context_sha256','other'),
    ('same_context_and_Gaussian',False),('counterfactual_branches_executed',1),('candidate_payload_sha256','other')])
def test_other_query_model_context_or_post_truth_seal_is_rejected(tmp_path,field,value):
    seal,generation,args=fixture(tmp_path);seal[field]=value
    with pytest.raises(ValueError):require_null_binding(seal,generation,**args)


def test_timing_view_must_still_reconstruct_original_full_binding(tmp_path):
    seal,generation,args=fixture(tmp_path);seal['binding_comparison']['comparison_bound_step_sha256']='different'
    with pytest.raises(ValueError,match='proof'):require_null_binding(seal,generation,**args)


def test_identical_null_and_refined_is_kept_as_a_result_and_returned_arrays_are_owned():
    actions={key:np.zeros((32,14),dtype=np.float32) for key in ('refined','candidate_null')}
    saved=require_native_null_actions(actions,actions['refined']);saved['refined'][0,0]=1
    assert not actions['refined'].any() and not saved['candidate_null'].any()


def test_a_change_in_the_unexecuted_horizon_or_a_precision_change_is_rejected():
    a=np.zeros((32,14),dtype=np.float32);b=a.copy();b[-1,-1]=np.nextafter(np.float32(0),np.float32(1))
    with pytest.raises(ValueError,match='Full H32'):require_native_null_actions({'refined':b,'candidate_null':a},a)
    with pytest.raises(ValueError,match='float32'):require_native_null_actions({'refined':a,'candidate_null':a.astype(np.float64)},a)
