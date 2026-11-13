from pathlib import Path
import dataclasses
import datetime
import importlib.util
import json

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('test_PB_null_bind',ROOT/'src/fastwam/research/pb_namespace_null.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


@pytest.fixture
def plan(tmp_path,monkeypatch):
    root=tmp_path/'PB_namespace2_null_20261113_v1'
    p=dict(schema=M.SCHEMA,case_ids=[0,1],query=8,frame=32,horizon=32,execution_prefix=4,
        checkpoint_step=12000,original_source_sha256=M.CORE,checkpoint_sha256=M.WEIGHT,
        simulator_source_sha256=M.SIM,automatic_restarts=0,maximum_seconds=12600,
        maximum_case_seconds=6000,additional_reserved_GPU_hours=3.5,
        prior_reserved_GPU_hours=M.PRIOR_RESERVED,cumulative_reserved_GPU_hours=143.8054115725,
        lifetime_cap_GPU_hours=144,max_concurrent_GPUs=6,original_windows_unchanged=True,
        policy_outcome_based_selection=False,effect_descriptor='original_uniform_mean_DINO_delta_v1',
        next_interaction_rubric='warm.rmbench.PB-counted-interaction.v1',
        minimum_admission_free_bytes=4*1024**3,runtime_free_space_guard_bytes=512*1024**2,
        latest_end_utc='2026-11-13T11:00:00+00:00',root=str(root),cases=[])
    for case in [0,1]:
        source=tmp_path/f'original-case{case}';source.mkdir()
        f=source/'factual.plan.json';f.write_text(json.dumps(dict(case_id=case,seed=340800000+case,output=str(source/'factual'))))
        op=source/'operator.json';op.write_text(json.dumps(dict(status='complete_fixed_original_factual_episode',
            namespace_case_id=case,candidate_seed=340800000+case,source_sha256=M.CORE,source_sha256_after=M.CORE,
            checkpoint_sha256=M.WEIGHT,plan_sha256=M.sha(f),model_updates=0,
            initial_observation_exact_to_qualification=True,first_replan_exact_to_qualification=True,
            original_namespace_initial_identity_exact=True,native_commands=1500,success_flag=False,
            probe_status={'8':'sealed_before_execution'})))
        native=source/'raw_command.txt';native.write_text('unchanged original command')
        ref=lambda x:dict(path=str(x),sha256=M.sha(x))
        d=root/f'case{case:02d}'
        p['cases'].append(dict(case_id=case,physical_seed=340800000+case,source=str(source/'factual'),
            output=str(d/'results'),runtime=str(d/'runtime'),lease=str(d/'native.worker.lock'),
            factual_plan=ref(f),factual_operator=ref(op),inputs=[ref(native)]))
    class Namespace:
        @staticmethod
        def validate_execution_binding(x,kind):
            assert kind=='factual'
            return {},dict(case_id=x['case_id'],seed=x['seed'])
    monkeypatch.setattr(M,'namespace_module',lambda:Namespace)
    return p


def test_complete_fixed_two_sources_bind_with_no_policy_selection(plan):
    assert M.validate_plan(plan) is plan
    assert plan['cumulative_reserved_GPU_hours']<=plan['lifetime_cap_GPU_hours']


@pytest.mark.parametrize('key,value',[
    ('query',48),('case_ids',[1,0]),('checkpoint_step',3000),('maximum_seconds',25200),
    ('additional_reserved_GPU_hours',7.0),('prior_reserved_GPU_hours',0.0),
    ('policy_outcome_based_selection',True),('original_windows_unchanged',False),
    ('automatic_restarts',1),('effect_descriptor','fitted_rescaling'),
    ('latest_end_utc','2026-11-13T15:00:00+00:00'),('frame',True),
])
def test_no_changed_source_scope_budget_or_metric_is_admitted(plan,key,value):
    plan[key]=value
    with pytest.raises(ValueError):M.validate_plan(plan)


@pytest.mark.parametrize('field', ['runtime','output','lease'])
def test_independent_control_cannot_write_original_source_or_sibling(plan,field):
    plan['cases'][0][field]=plan['cases'][1][field]
    with pytest.raises(ValueError):M.validate_plan(plan)


def test_factual_failure_or_unreached_query_stays_unknown(plan):
    f=Path(plan['cases'][0]['factual_operator']['path']);v=json.loads(f.read_text());v['native_commands']=35
    f.write_text(json.dumps(v));plan['cases'][0]['factual_operator']['sha256']=M.sha(f)
    with pytest.raises(ValueError):M.validate_plan(plan)


def test_raw_input_modification_is_rejected(plan):
    Path(plan['cases'][1]['inputs'][0]['path']).write_text('changed original action')
    with pytest.raises(ValueError):M.validate_plan(plan)


def test_a_foreign_source_or_actual_seed_is_not_a_replacement(plan):
    plan['cases'][0]['physical_seed']=340800001
    with pytest.raises(ValueError):M.validate_plan(plan)


def test_deadline_never_restarts_or_crosses_original_cutoff():
    z=datetime.timezone.utc;first=datetime.datetime(2026,11,13,2,tzinfo=z);latest=datetime.datetime(2026,11,13,11,tzinfo=z)
    assert M.deadline(first,latest)==first+datetime.timedelta(hours=3.5)
    assert M.deadline(latest-datetime.timedelta(hours=1),latest)==latest
    with pytest.raises(ValueError):M.deadline(latest-datetime.timedelta(minutes=20),latest)
    with pytest.raises(ValueError):M.deadline(first.replace(tzinfo=None),latest)


def test_case_window_reserves_exit_and_archive_margin_inside_same_allocation():
    z=datetime.timezone.utc;first=datetime.datetime(2026,11,13,2,tzinfo=z)
    end=first+datetime.timedelta(hours=3.5)
    assert M.case_deadline(first,end)==first+datetime.timedelta(seconds=6000)
    later=end-datetime.timedelta(hours=1)
    assert M.case_deadline(later,end)==end-datetime.timedelta(seconds=180)
    with pytest.raises(ValueError):M.case_deadline(end-datetime.timedelta(seconds=1900),end)


def test_only_identical_existing_robot_asset_aliases_keep_original_config(tmp_path):
    asset=tmp_path/'robot-assets';asset.mkdir()
    original=dict(left_robot_file=str(asset),right_robot_file=str(asset),step_limit=1500)
    actual=dict(original,left_robot_file=str(asset/'..'/'robot-assets'))
    assert M.exact_native_config(actual,original)==original
    assert actual['left_robot_file']!=original['left_robot_file']
    other=tmp_path/'other-assets';other.mkdir();actual['left_robot_file']=str(other)
    with pytest.raises(ValueError):M.exact_native_config(actual,original)
    with pytest.raises(ValueError):M.exact_native_config(dict(original,step_limit=700),original)
    with pytest.raises(ValueError):M.exact_native_config(dict(original,added_flag=True),original)


def context_fixture():
    torch=pytest.importorskip('torch')
    names=['query_context','candidate_context','candidate_actions','candidate_start_proprio','candidate_effect_pre',
           'candidate_effect_delta','candidate_timing','candidate_support','candidate_valid_mask','episode_tokens',
           'episode_mask','episode_action_summaries','episode_action_mask']
    values={name:torch.ones(1) for name in names}
    values.update(candidate_actions=torch.zeros(1,32,32,14),episode_tokens=torch.zeros(1,16,768),
                  target_action=None,target_effect=None,current_semantic_teacher=None,target_action_valid_mask=None,
                  future_valid_mask=None,forced_candidate_indices=None)
    cls=dataclasses.make_dataclass('SyntheticEngineeringContext',[(k,object) for k in values])
    return cls(**values)


def test_context_capture_preserves_actual_tensors_and_omits_absent_future_fields():
    torch=pytest.importorskip('torch')
    c=context_fixture();p=M.inference_context_payload(c)
    assert 'target_effect' not in p and len(p)==13
    assert torch.equal(p['episode_tokens'],c.episode_tokens) and p['episode_tokens'] is not c.episode_tokens


@pytest.mark.parametrize('key', ['target_action','target_effect','current_semantic_teacher','target_action_valid_mask','future_valid_mask'])
def test_any_future_tensor_is_rejected_before_null_generation(key):
    torch=pytest.importorskip('torch')
    c=context_fixture();setattr(c,key,torch.zeros(1))
    with pytest.raises(ValueError):M.inference_context_payload(c)


def test_a_type_description_or_nonfinite_history_cannot_replace_recorded_history():
    torch=pytest.importorskip('torch')
    c=context_fixture();c.episode_tokens={'unserialized_type':'numpy.ndarray'}
    with pytest.raises(ValueError):M.inference_context_payload(c)
    c=context_fixture();c.episode_tokens[0,0,0]=float('nan')
    with pytest.raises(ValueError):M.inference_context_payload(c)
