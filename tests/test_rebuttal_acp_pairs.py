import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('acp_pairs',Path(__file__).resolve().parents[1]/'scripts/run_rebuttal_acp_pairs.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_scheduler_visibility_is_not_remapped_outside_allocation():
    assert module.assigned_devices(2,'GPU-a,GPU-b')==['GPU-a','GPU-b']
    assert module.assigned_devices(2,None)==['0','1']


def test_single_gpu_layout_shares_only_the_one_assigned_device():
    assert module.assigned_devices(1,'GPU-assigned','shared_single_gpu')==['GPU-assigned','GPU-assigned']
    assert module.assigned_devices(1,None,'shared_single_gpu')==['0','0']
    for count,declared in ((2,'0,1'),(1,'0,1'),(1,'-1'),(1,'all')):
        with pytest.raises(ValueError):module.assigned_devices(count,declared,'shared_single_gpu')
    with pytest.raises(ValueError):module.assigned_devices(1,'0','unknown')


@pytest.mark.parametrize('count,declared',[(1,None),(2,'0,0'),(2,'all'),(2,'0,1,2'),(2,'0,-1')])
def test_invalid_or_unallocated_devices_are_rejected(count,declared):
    with pytest.raises(ValueError):module.assigned_devices(count,declared)


def test_role_worker_keeps_original_formal_cohort_and_full_length():
    plan=dict(core_code='/code',bundle_args='/args',runtime='/sim',run_root='/out',evidence_plan='/plan',roles=['full','candidate_null'])
    full=module.worker_argv(plan,'full');null=module.worker_argv(plan,'candidate_null')
    assert full[full.index('--cases')+1]=='10' and full[full.index('--phase')+1]=='formal'
    assert '--command-cap' not in full and '--command-cap' not in null
    assert full[full.index('--roles')+1]=='full' and null[null.index('--roles')+1]=='candidate_null'
    assert full[full.index('--seed-start')+1]==null[null.index('--seed-start')+1]=='340800000'
    with pytest.raises(ValueError):module.worker_argv(plan,'other')
