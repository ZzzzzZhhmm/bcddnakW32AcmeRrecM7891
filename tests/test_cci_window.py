import copy

import pytest

from fastwam.research.cci_window import admit_window, observe_bootstrap


def observation():
    return dict(bootstrap_pid=7,bootstrap_start_ticks=10000,bootstrap_age_seconds=300,
                observed_epoch_seconds=1000,cmdline_sha256='a'*64,
                clock_source='observed_workload_bootstrap_not_pid1')


def process(root,pid,command,parent,start):
    target=root/str(pid);target.mkdir()
    (target/'cmdline').write_bytes(command)
    fields=['S',str(parent)]+['0']*17+[str(start)]+['0']*6
    (target/'stat').write_text(str(pid)+' (bootstrap with spaces) '+' '.join(fields))


def test_observation_uses_workload_bootstrap_and_does_not_expose_command(tmp_path):
    (tmp_path/'uptime').write_text('9000 0')
    process(tmp_path,1,b'/pause\0',0,1)
    process(tmp_path,7,b'bash\0-c\0openssh-server authorized_keys stop-cci\0',0,870000)
    value=observe_bootstrap(tmp_path,clock=lambda:1000,ticks_per_second=100)
    assert value['bootstrap_pid']==7 and value['bootstrap_age_seconds']==300
    assert len(value['cmdline_sha256'])==64
    assert all('openssh' not in str(v) and 'authorized_keys' not in str(v) for v in value.values())


@pytest.mark.parametrize('count',[0,2])
def test_missing_or_ambiguous_workload_is_rejected(tmp_path,count):
    (tmp_path/'uptime').write_text('9000 0')
    for index in range(count):process(tmp_path,7+index,b'bash\0openssh-server authorized_keys stop-cci\0',0,870000)
    with pytest.raises(ValueError,match='Unique'):observe_bootstrap(tmp_path,clock=lambda:1000,ticks_per_second=100)


def test_task_and_receipt_margin_are_both_reserved():
    result=admit_window(observation(),now=1010,task_seconds=7200)
    assert result['remaining_seconds']==14090 and result['required_seconds']==7380
    assert result['new_GPU_or_physical_actions']==0


def test_task_near_previous_restart_is_rejected():
    value=observation();value['bootstrap_age_seconds']=14000
    with pytest.raises(ValueError,match='restart window'):admit_window(value,now=1000,task_seconds=3150)


@pytest.mark.parametrize('now',[999,1031])
def test_future_or_stale_observation_is_rejected(now):
    with pytest.raises(ValueError,match='Stale or future'):admit_window(observation(),now=now,task_seconds=100)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),True,-1,0])
def test_invalid_task_budget_is_rejected(value):
    with pytest.raises(ValueError):admit_window(observation(),now=1000,task_seconds=value)


@pytest.mark.parametrize('key,value',[('clock_source','pid1'),('bootstrap_pid',True),('bootstrap_start_ticks',0),('cmdline_sha256','bad')])
def test_unverified_identity_is_rejected(key,value):
    modified=copy.deepcopy(observation());modified[key]=value
    with pytest.raises(ValueError):admit_window(modified,now=1000,task_seconds=100)


def test_exact_margin_boundary_is_rejected():
    value=observation();value['bootstrap_age_seconds']=13900
    with pytest.raises(ValueError,match='restart window'):admit_window(value,now=1000,task_seconds=320)
