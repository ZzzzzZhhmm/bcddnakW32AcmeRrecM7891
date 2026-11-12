"""Read-only CCI admission checks against the observed container bootstrap.

The pause process can outlive restarted workload containers. Do not use PID 1,
GPU availability, or an unchanged kernel boot ID as a workload-lifetime clock.
Only command hashes are retained; bootstrap command contents are never returned.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import time


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid {label}")
    return float(value)


def observe_bootstrap(proc_root=Path('/proc'), *, clock=time.time, ticks_per_second=None):
    """Observe the unique known CCI bootstrap without returning its arguments."""
    proc_root=Path(proc_root)
    hz=os.sysconf('SC_CLK_TCK') if ticks_per_second is None else ticks_per_second
    hz=_finite_number(hz,'clock tick rate')
    if hz<=0:raise ValueError('Nonpositive clock tick rate')
    uptime=_finite_number(float((proc_root/'uptime').read_text().split()[0]),'uptime')
    observed=_finite_number(clock(),'observation time')
    matches=[]
    for directory in proc_root.iterdir():
        if not directory.name.isdecimal():continue
        try:
            command=(directory/'cmdline').read_bytes()
            fields=(directory/'stat').read_text().rpartition(') ')[2].split()
            parent=int(fields[1]);start_ticks=int(fields[19])
        except (OSError,ValueError,IndexError):continue
        executable=command.split(b'\0',1)[0].rsplit(b'/',1)[-1]
        if parent!=0 or executable!=b'bash':continue
        if not all(marker in command for marker in [b'openssh-server',b'authorized_keys',b'stop-cci']):continue
        age=uptime-start_ticks/hz
        if age<0:raise ValueError('Bootstrap starts after observed uptime')
        matches.append(dict(bootstrap_pid=int(directory.name),bootstrap_start_ticks=start_ticks,
                            bootstrap_age_seconds=age,observed_epoch_seconds=observed,
                            cmdline_sha256=hashlib.sha256(command).hexdigest(),
                            clock_ticks_per_second=hz,clock_source='observed_workload_bootstrap_not_pid1'))
    if len(matches)!=1:raise ValueError('Unique workload bootstrap could not be verified')
    return matches[0]


def admit_window(observation, *, now, task_seconds, receipt_margin_seconds=180,
                 restart_period_seconds=14400, maximum_observation_age_seconds=30):
    """Require the full task and receipt margin inside the conservative CCI window."""
    if observation.get('clock_source')!='observed_workload_bootstrap_not_pid1':
        raise ValueError('Unverified workload-lifetime clock')
    for key in ['bootstrap_pid','bootstrap_start_ticks']:
        if type(observation.get(key)) is not int or observation[key]<=0:
            raise ValueError('Invalid bootstrap identity')
    digest=observation.get('cmdline_sha256','')
    if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Missing bootstrap identity hash')
    values={label:_finite_number(value,label) for label,value in dict(
        now=now,task=task_seconds,margin=receipt_margin_seconds,period=restart_period_seconds,
        max_age=maximum_observation_age_seconds,observed=observation.get('observed_epoch_seconds'),
        age=observation.get('bootstrap_age_seconds')).items()}
    if values['task']<=0 or values['margin']<0 or values['period']<=0 or values['max_age']<0 or values['age']<0:
        raise ValueError('Invalid workload window budget')
    elapsed=values['now']-values['observed']
    if elapsed<0 or elapsed>values['max_age']:raise ValueError('Stale or future bootstrap observation')
    remaining=values['period']-values['age']-elapsed
    required=values['task']+values['margin']
    if required>=remaining:raise ValueError('Insufficient CCI restart window for task and receipts')
    return dict(admitted=True,bootstrap_pid=observation['bootstrap_pid'],
                bootstrap_start_ticks=observation['bootstrap_start_ticks'],
                remaining_seconds=remaining,required_seconds=required,
                restart_period_seconds=values['period'],new_GPU_or_physical_actions=0)
