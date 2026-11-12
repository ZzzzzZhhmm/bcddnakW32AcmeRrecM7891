from pathlib import Path
import importlib.util
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/qualify_rebuttal_original_PB.py'
spec=importlib.util.spec_from_file_location('PB_qualification',path)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def binding():
    return dict(M.PB_PROTOCOL,original_source_sha256=M.CORE_SHA,
        original_source_commit=M.COMMIT)


def test_qualification_has_engineering_scope_only():
    p=M.validate_qualification_plan(binding())
    assert (p['qualification_queries'],p['qualification_commands'])==(1,4)


@pytest.mark.parametrize('key,value',[
    ('task_name','battery_try'),('qualified_original_run','shared-v8'),
    ('checkpoint_step',3000),('root_seed',3408),('physical_seed',1),
    ('physical_seed',False),('native_limit',1000),('qualification_queries',32),
    ('qualification_queries',True),('qualification_commands',1500),
    ('checkpoint_sha256','another-model'),('original_source_sha256','another-source'),
    ('original_source_commit','another-commit')])
def test_scientific_binding_and_small_qualification_scope_cannot_drift(key,value):
    p=binding();p[key]=value
    with pytest.raises(ValueError):M.validate_qualification_plan(p)


def test_PB_audit_does_not_mutate_success_counters():
    class Native:
        card_id_1=2;card_id_2=3
        press_cnt_1=1;press_cnt_2=2;press_cnt_check_button=0
        press_flag_1=False;press_flag_2=True;press_flag_check_button=False
        def check_success(self):
            raise AssertionError('An audit must not call the mutating success check')
    env=Native();before=vars(Native).copy()
    a=M.press_button_task_state(env);b=M.press_button_task_state(env)
    assert a==b and a['press_cnt_1']==1 and vars(Native)==before


stage_path=path.parent/'run_rebuttal_original_PB_stage.py'
stage_spec=importlib.util.spec_from_file_location('PB_stage',stage_path)
S=importlib.util.module_from_spec(stage_spec);stage_spec.loader.exec_module(S)


def stage():
    return dict(schema='warm.original-PB-new-stage-plan.v1',task='press_button',
        maximum_stage_seconds=7200,qualification_maximum_seconds=2400,
        physical_seed=0,capture_replans=[8,48,96],predecessor_key='bt-40',
        original_windows_unchanged=True)


@pytest.mark.parametrize('key,value',[
    ('maximum_stage_seconds',14400),('qualification_maximum_seconds',7200),
    ('physical_seed',1),('physical_seed',False),('capture_replans',[0,1,2]),
    ('predecessor_key','rb-30'),('original_windows_unchanged',False)])
def test_stage_refuses_reset_limits_or_replaced_queries(key,value):
    p=stage();p[key]=value
    with pytest.raises(ValueError):S.validate_stage_plan(p)


def test_missing_output_cannot_be_archived_as_a_completed_run(tmp_path):
    with pytest.raises(AssertionError):S.archive(tmp_path/'missing',tmp_path/'archives','run')
