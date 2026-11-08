"""Rubric fixtures check scoring logic, not true simulator applicability."""
from copy import deepcopy
from fastwam.research.rmbench_outcomes import RUBRIC,label_next_interaction


def rb():return dict(rubric=RUBRIC,task='rearrange_blocks',terminal_success=False,stage_id=0,press_cnt=0,
    lifted=[False,False],first_placement=False,predicate_margins=[.01],object_fell_below_table=False)


def test_ordered_progress_and_regression_are_distinguished():
    before=rb();after=deepcopy(before);after['lifted'][0]=True
    assert label_next_interaction(before,after,valid_branch=True)['applicable']==1
    assert label_next_interaction(after,before,valid_branch=True)['applicable']==0
    after['press_cnt']=2
    assert label_next_interaction(before,after,valid_branch=True)['applicable']==0
    after['terminal_success']=True
    assert label_next_interaction(before,after,valid_branch=True)['applicable']==1
    assert label_next_interaction(before,after,valid_branch=False)['applicable'] is None


def test_ambiguous_geometric_boundary_stays_unknown():
    before=rb();after=deepcopy(before);after['predicate_margins']=[.001]
    assert label_next_interaction(before,after,valid_branch=True)['applicable'] is None


def test_new_battery_configuration_is_progress_without_using_hidden_correctness():
    before=dict(rubric=RUBRIC,task='battery_try',terminal_success=False,object_fell_below_table=False,
        in_slots=[True,True],lifted=[False,False],combination=[0,0],correct_combination=[1,1],predicate_margins=[.01,.01])
    after=deepcopy(before);after['combination']=[1,0]
    result=label_next_interaction(before,after,valid_branch=True,seen_combinations=[(0,0)])
    assert result['applicable']==1
    after['correct_combination']=[1,0]
    assert label_next_interaction(before,after,valid_branch=True,seen_combinations=[(0,0)])==result
    assert label_next_interaction(before,after,valid_branch=True,seen_combinations=[(0,0),(1,0)])['applicable']==0
