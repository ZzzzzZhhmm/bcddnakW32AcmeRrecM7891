from pathlib import Path
import copy
import importlib.util

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'PB_counted_interaction', ROOT / 'src/fastwam/research/pb_mechanism.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def plan():
    return dict(schema='warm.original-PB-first-state-fresh-pool.v1',
        task_name='press_button', root_seed=3407, physical_seed=0,
        checkpoint_step=12000, horizon=32, execution_prefix=4,
        checkpoint_sha256=M.WEIGHT_SHA256, original_source_sha256=M.CORE_SHA256,
        factual_source_plan_sha256=M.TRACE_PLAN_SHA256,
        next_interaction_rubric=M.RUBRIC,
        effect_descriptor='original_uniform_mean_DINO_delta_v1',
        query_replan=8, selected_rank=7)


@pytest.mark.parametrize('key,value', [
    ('physical_seed', False), ('physical_seed', 1), ('root_seed', 3408),
    ('checkpoint_step', 3000), ('horizon', 4), ('execution_prefix', 32),
    ('checkpoint_sha256', 'another'), ('original_source_sha256', 'shared'),
    ('factual_source_plan_sha256', 'different'), ('query_replan', 9),
    ('selected_rank', -1), ('selected_rank', 32), ('selected_rank', True),
    ('next_interaction_rubric', 'posthoc'), ('effect_descriptor', 'qpos_MSE')])
def test_pinned_scientific_inputs_cannot_change(key, value):
    p = plan()
    p[key] = value
    with pytest.raises(ValueError):
        M.validate_plan(p)


def test_all_three_queries_accept_the_original_model_selected_rank():
    for query in (8, 48, 96):
        p = plan()
        p['query_replan'] = query
        assert M.validate_plan(p)['selected_rank'] == 7


def source(n=1500):
    return dict(status='complete_fixed_original_factual_episode', candidate_seed=0,
        source_sha256=M.CORE_SHA256, source_sha256_after=M.CORE_SHA256,
        checkpoint_sha256=M.WEIGHT_SHA256,
        initial_observation_exact_to_qualification=True,
        first_replan_exact_to_qualification=True,
        hidden_PB_counter_state_audited_only_not_policy_input=True,
        model_updates=0, native_commands=n, success_flag=False)


def trace(n=1500):
    return [dict(index=i, take_action_cnt=i+1,
        task_state_before={}, task_state_after={}) for i in range(n)]


def test_full_or_genuine_native_success_only_are_complete():
    assert M.validate_source(source(), trace(), 96) == 1500
    s = source(100)
    with pytest.raises(ValueError):
        M.validate_source(s, trace(100), 8)
    s['success_flag'] = True
    assert M.validate_source(s, trace(100), 8) == 100
    with pytest.raises(ValueError):
        M.validate_source(s, trace(100), 48)


def test_counter_audit_and_source_exactness_are_required():
    s = source()
    s['hidden_PB_counter_state_audited_only_not_policy_input'] = False
    with pytest.raises(ValueError):
        M.validate_source(s, trace(), 8)
    t = trace()
    t[4]['take_action_cnt'] = 99
    with pytest.raises(ValueError):
        M.validate_source(source(), t, 8)


def progress(counts=(0, 0), targets=(2, 3), success=False):
    return dict(rubric=M.RUBRIC, targets=list(targets), counts=list(counts),
        terminal_success=success)


@pytest.mark.parametrize('counts,success,expected', [
    ((0, 0), False, 0), ((1, 0), False, 1), ((2, 3), True, 1),
    ((3, 0), False, 0), ((2, 4), False, 0), ((1, 0), True, None)])
def test_counter_labels_use_native_targets_and_completed_interactions(counts, success, expected):
    label = M.label(progress(), progress(counts, success=success), valid_branch=True)
    assert label['applicable'] == expected


def test_invalid_or_already_unrepairable_parent_is_unknown():
    assert M.label(progress(), progress((1, 0)), valid_branch=False)['applicable'] is None
    assert M.label(progress((3, 0)), progress((3, 1)), valid_branch=True)['applicable'] is None
    assert M.label(progress((2, 3), success=True), progress((2, 3), success=True), valid_branch=True)['applicable'] is None
    assert M.label(progress((1, 0)), progress(), valid_branch=True)['applicable'] is None


def test_observation_never_calls_mutating_native_success_check():
    class Native:
        card_id_1 = 2
        card_id_2 = 3
        press_cnt_1 = 1
        press_cnt_2 = 0
        press_cnt_check_button = 0
        press_flag_1 = True
        press_flag_2 = False
        press_flag_check_button = False
        eval_success = False

        def check_success(self):
            raise AssertionError('This call would change the task')

    env = Native()
    assert M.observe(env) == M.observe(env)
    assert M.observe(env)['counts'] == [1, 0]


def records():
    a = dict(parent_hashes={'physics':'parent'}, endpoint_status='early_success',
        executed_steps=12, endpoint_hashes={'physics':'full'}, observed_effect=None,
        independent_label={'applicable':1}, error=None,
        parent_progress=progress(), endpoint_progress=progress((2,3), success=True),
        secondary_prefix4=None)
    rows = {key:copy.deepcopy(a) for key in ['anchor-a','anchor-aa','anchor-qual-final','rank-7']}
    rows['rank-7']['endpoint_hashes']['physics'] = 'raw'
    rows['rank-7']['independent_label']['applicable'] = 0
    return rows


def test_repeat_success_is_censored_for_H32_effect_but_label_remains_known():
    rows = records()
    r = M.require_repeats(rows, 7)
    assert not r['full_H32_effect_available']
    assert r['raw_distinct_physical_endpoint']
    assert rows['anchor-a']['observed_effect'] is None
    assert M.repair(rows, 7)['repaired'] is True
    assert not M.repair(rows, 7)['population_success_rate_estimated']


def test_no_motion_or_invalid_raw_is_preserved_without_manufacturing_repair():
    rows = records()
    rows['rank-7']['endpoint_hashes'] = rows['anchor-a']['endpoint_hashes'].copy()
    rows['rank-7']['independent_label']['applicable'] = None
    assert not M.require_repeats(rows, 7)['raw_distinct_physical_endpoint']
    assert M.repair(rows, 7)['repaired'] is None


def test_repeat_endpoint_or_counter_drift_is_rejected():
    rows = records()
    rows['anchor-aa']['endpoint_progress']['counts'][0] = 1
    with pytest.raises(ValueError):
        M.require_repeats(rows, 7)


def test_sealed_evidence_is_not_overwritten_on_resume(tmp_path):
    path = tmp_path/'proof.json'
    M.immutable_json(path, {'native':1})
    before = path.read_bytes()
    M.immutable_json(path, {'native':1})
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        M.immutable_json(path, {'native':2})
    assert path.read_bytes() == before
