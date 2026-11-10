from copy import deepcopy
from datetime import datetime,timezone
from types import SimpleNamespace
from dataclasses import dataclass
import json
import pytest
import torch
from fastwam.research.matched_native import (
    DESIGN_SHA256,INFERENCE,ROLE_TYPES,validate_plan,validate_training_attestation,
    require_deadline,pair_receipts,next_request,capture_replan,PolicyOutputError,invalid_action_proof,file_sha,
)

def plan(phase='recorded_dev'):
    p=dict(schema='warm.rebuttal.matched-native.v1',phase=phase,design_sha256=DESIGN_SHA256,inference=INFERENCE.copy(),root_seed=3407,
        task='press_button',query_frames=list(range(0,40,4)),recorded_episode=435,episode_id=0,
        first_admitted_utc='2026-11-11T00:00:00+00:00',absolute_deadline_utc='2026-11-11T01:00:00+00:00',roles={})
    for role,(cfg,cls,schema) in ROLE_TYPES.items():
        p['roles'][role]=dict(config_name=cfg,model_class=cls,checkpoint_schema=schema,checkpoint_sha256=role,
            checkpoint_update=3000 if phase=='formal' else 200 if role=='full' else 100)
    if phase=='formal':p.update(task='rearrange_blocks',official_step_limit=700,case_ids=list(range(10)),cohort_case_ids=list(range(50)),
        shard_max_seconds=43200,worker_max_seconds=7200,native_concurrency_qualified=True)
    return p

def test_roles_have_distinct_checkpoint_types_and_final_admission():
    assert validate_plan(plan())['phase']=='recorded_dev'
    p=plan('formal');assert validate_plan(p)['roles']['full']['checkpoint_update']==3000
    p['roles']['full']['checkpoint_update']=200
    with pytest.raises(ValueError,match='checkpoint update'):validate_plan(p)

@pytest.mark.parametrize('key,value',[('config_name','sim_rmbench.yaml'),('model_class','WarmRetrospectionFastWAM'),('checkpoint_schema','warm.retrospection-checkpoint')])
def test_pooled_checkpoint_never_passes_through_Full_factory(key,value):
    p=plan();p['roles']['pooled_memory_prior'][key]=value
    with pytest.raises(ValueError,match='factory/schema'):validate_plan(p)

def test_original_single_checkpoint_and_false_native_horizon_are_rejected():
    p=plan();p['roles']['pooled_memory_prior']['checkpoint_sha256']='full'
    with pytest.raises(ValueError,match='independent'):validate_plan(p)
    p=plan('formal');p['official_step_limit']=600
    with pytest.raises(ValueError,match='native'):validate_plan(p)
    p=plan('formal');p['native_concurrency_qualified']=False
    with pytest.raises(ValueError,match='qualification'):validate_plan(p)

def test_attestation_must_match_exact_update_and_independent_weight():
    b=plan()['roles']['pooled_memory_prior'];a=dict(schema='warm.training-attestation',checkpoint_step=100,actual_global_step=100,checkpoint_sha256='pooled_memory_prior',root_seed=3407)
    validate_training_attestation(b,a)
    for key,value in [('checkpoint_sha256','full'),('actual_global_step',3000),('root_seed',3408)]:
        bad=dict(a,**{key:value})
        with pytest.raises(ValueError,match='lineage'):validate_training_attestation(b,bad)

def test_same_absolute_phase_window_survives_restart():
    p=plan();now=datetime(2026,11,11,0,59,tzinfo=timezone.utc)
    expected=datetime(2026,11,11,1,tzinfo=timezone.utc).timestamp()
    assert require_deadline(p,now)==expected
    with pytest.raises(TimeoutError):require_deadline(p,datetime(2026,11,11,1,tzinfo=timezone.utc))
    p['absolute_deadline_utc']='2026-11-11T03:00:00+00:00'
    with pytest.raises(ValueError,match='window'):require_deadline(p,now)

def paired():
    q=dict(frame=0,derived_seed=10,base_gaussian=dict(shape=[1,32,14],sha256='actual'),model_input_sha256='rgb',proprio_sha256='state',bank_rows=list(range(32)))
    left=dict(role='full',plan_sha256='plan',task='battery_try',episode_id=2,phase='formal',initial_identity={'instruction':'same','rgb':'same'},
        common_contract={'bank':'same'},queries=[q],status='complete_native',success=1)
    right=deepcopy(left);right.update(role='pooled_memory_prior',success=0)
    return left,right

def test_pair_requires_actual_initial_noise_and_factual_input_identity():
    a,b=paired();assert pair_receipts(a,b)['full_minus_pooled']==1
    for key in ('base_gaussian','derived_seed','model_input_sha256','bank_rows'):
        bad=deepcopy(b);bad['queries'][0][key]='different'
        with pytest.raises(ValueError,match='Gaussian'):pair_receipts(a,bad)
    bad=deepcopy(b);bad['initial_identity']['instruction']='different'
    with pytest.raises(ValueError,match='initial input'):pair_receipts(a,bad)

def test_later_diverged_policy_observations_are_not_forced_equal():
    a,b=paired();a['queries'].append(dict(a['queries'][0],frame=4,model_input_sha256='left'))
    b['queries'].append(dict(b['queries'][0],frame=4,model_input_sha256='right'))
    assert pair_receipts(a,b)['actual_noise_pairs_verified']==1
    with pytest.raises(ValueError):pair_receipts(a,b,recorded_dev=True)

def test_unknowns_never_get_success_zero_and_DEV_has_no_task_difference():
    a,b=paired();b.update(status='unknown_runtime_or_interruption',success=None)
    assert pair_receipts(a,b)['full_minus_pooled'] is None
    assert pair_receipts(a,b,recorded_dev=True)['full_minus_pooled'] is None

def test_counterbalanced_order_and_duplicate_known_outcome_rejection():
    assert next_request([], [0,1])==(0,'full')
    assert next_request([], [1])==(1,'pooled_memory_prior')
    row=dict(episode_id=0,role='full',status='complete_native')
    assert next_request([row],[0])==(0,'pooled_memory_prior')
    with pytest.raises(ValueError,match='Duplicate'):next_request([row,row],[0])

def test_unknown_exit_cannot_spawn_duplicate_but_explicit_terminal_retry_is_possible():
    row=dict(episode_id=0,role='full',status='unknown_runtime_or_interruption',attempt_id='kept')
    with pytest.raises(RuntimeError,match='Unknown or active'):next_request([row],[0])
    assert next_request([row],[0],known_terminal_attempts=['kept'])==(0,'full')
    complete=dict(episode_id=0,role='full',status='complete_native',attempt_id='later')
    assert next_request([row,complete],[0],known_terminal_attempts=['kept'])==(0,'pooled_memory_prior')

@dataclass
class Context:
    candidate_actions:torch.Tensor
    target_action:object=None
    target_effect:object=None
    current_semantic_teacher:object=None
    target_action_valid_mask:object=None
    future_valid_mask:object=None

class Model:
    mode='unchanged'
    def __init__(self,action):self.action=action
    def _resolve_action_source(self,**kwargs):
        self.received=kwargs;self.resolved=SimpleNamespace(source=kwargs['base_gaussian']+1,conditioning_tokens=torch.zeros(1,12,4));return self.resolved
    def infer_action(self,**kwargs):
        self.source_value=self._resolve_action_source(base_gaussian=kwargs['gaussian'],action_source_context=kwargs['context'])
        self.output={'action':self.action};return self.output

@pytest.mark.parametrize('shape',[(32,14),(1,32,14)])
def test_actual_capture_preserves_both_valid_native_output_layouts_and_mode(tmp_path,shape):
    model=Model(torch.zeros(shape));policy=SimpleNamespace(model=model);gaussian=torch.randn(1,32,14);context=Context(torch.ones(1,32,32,14))
    with capture_replan(policy,tmp_path/'query') as record:
        result=model.infer_action(online_step='fixed',gaussian=gaussian,context=context)
    assert result is model.output and model.source_value is model.resolved and model.received['base_gaussian'] is gaussian
    assert model.mode=='unchanged' and not hasattr(model,'_warm_research_probe')
    assert 'infer_action' not in vars(model) and '_resolve_action_source' not in vars(model)
    assert torch.equal(record['base_gaussian'],gaussian)
    assert json.loads((tmp_path/'query/generation.json').read_text())['shape']==list(shape)

def test_invalid_returned_model_action_is_sealed_before_known_failure(tmp_path):
    model=Model(torch.full((32,14),float('nan')));policy=SimpleNamespace(model=model)
    with pytest.raises(PolicyOutputError):
        with capture_replan(policy,tmp_path/'bad'):
            model.infer_action(online_step='fixed',gaussian=torch.zeros(1,32,14),context=Context(torch.ones(1,32,32,14)))
    assert (tmp_path/'bad/model_actions.npy').exists()
    assert json.loads((tmp_path/'bad/generation.json').read_text())['all_finite'] is False
    assert json.loads((tmp_path/'bad/consumer_status.json').read_text())['consumer_completed'] is False
    assert 'infer_action' not in vars(model)

def test_future_teacher_is_rejected_before_source_consumption(tmp_path):
    model=Model(torch.zeros(32,14));policy=SimpleNamespace(model=model)
    with pytest.raises(ValueError,match='future/teacher'):
        with capture_replan(policy,tmp_path/'future'):
            model.infer_action(online_step='fixed',gaussian=torch.zeros(1,32,14),context=Context(torch.ones(1,32,32,14),target_action=torch.ones(1)))
    assert not hasattr(model,'received') and 'infer_action' not in vars(model)

def test_TOPP_observer_preserves_original_exception_and_restores_methods():
    from scripts.run_rebuttal_matched_native import TOPPObserver
    class Planner:
        def TOPP(self,value):raise RuntimeError('native failure')
    robot=SimpleNamespace(left_mplib_planner=Planner(),right_mplib_planner=Planner());env=SimpleNamespace(robot=robot)
    with TOPPObserver(env) as observed:
        with pytest.raises(RuntimeError,match='native failure'):robot.left_mplib_planner.TOPP('original')
        assert observed.errors==[{'planner':'left_mplib_planner','error':"RuntimeError('native failure')"}]
    assert 'TOPP' not in vars(robot.left_mplib_planner)

def test_generic_exit_or_flag_cannot_manufacture_an_identified_policy_failure(tmp_path):
    import numpy as np
    assert invalid_action_proof(tmp_path) is False
    query=tmp_path/'generations/q0';query.mkdir(parents=True)
    np.save(query/'model_actions.npy',np.zeros((32,14),dtype=np.float32),allow_pickle=False)
    metadata=dict(before_consumer_validation=True,shape=[32,14],all_finite=False,action_file_sha256=file_sha(query/'model_actions.npy'))
    (query/'generation.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='failure metadata'):invalid_action_proof(tmp_path)
    metadata['all_finite']=True;(query/'generation.json').write_text(json.dumps(metadata))
    assert invalid_action_proof(tmp_path) is False

def test_verified_result_rejects_foreign_weights_and_unknown_zero_label(tmp_path):
    from scripts.run_rebuttal_matched_campaign import verified_result
    p=plan();p.update(evaluation_source_sha256='core',controller_source_sha256='helper')
    row=dict(schema='warm.rebuttal.matched-native-role.v1',role='full',task=p['task'],phase=p['phase'],episode_id=0,
        plan_sha256='fixed',checkpoint_sha256='full',evaluation_source_sha256='core',controller_source_sha256='helper',
        status='unknown_runtime_or_interruption',success=0)
    (tmp_path/'result.json').write_text(json.dumps(row))
    with pytest.raises(ValueError,match='must remain unknown'):verified_result(tmp_path,p,'fixed',0,'full')
    row['success']=None;(tmp_path/'result.json').write_text(json.dumps(row))
    assert verified_result(tmp_path,p,'fixed',0,'full')['success'] is None
    row['checkpoint_sha256']='pooled_memory_prior';(tmp_path/'result.json').write_text(json.dumps(row))
    with pytest.raises(ValueError,match='typed role'):verified_result(tmp_path,p,'fixed',0,'full')
