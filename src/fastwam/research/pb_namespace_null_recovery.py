"""Admit only known pre-intervention collector-layout failures, in the old window."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, json

SCHEMA='warm.PB-namespace2-null-prebranch-recovery.v1'
ERROR="ValueError('Original v3 candidate/history tensor layout differs')"

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def reference(row):
    p=Path(row['path'])
    if sha(p)!=row['sha256']:raise ValueError('Changed recovery input: '+str(p))
    return p

def validate_failures(parent, runner, cases, now):
    if parent.get('status')!='terminal_fixed_q8_withdrawal_all_existing_output_archived' or runner.get('status')!='complete' or runner.get('exit_code')!=0:
        raise ValueError('Original supervisor must have a known terminal')
    first=datetime.datetime.fromisoformat(parent['first_admitted_utc']);end=datetime.datetime.fromisoformat(parent['absolute_deadline_utc'])
    hard=datetime.datetime(2026,11,13,11,tzinfo=datetime.timezone.utc)
    if any(d.tzinfo is None for d in [first,end,now]) or end!=min(first+datetime.timedelta(seconds=12600),hard):
        raise ValueError('Original null allocation clock differs')
    if not first<now<end-datetime.timedelta(seconds=1980):raise ValueError('Insufficient unchanged recovery window')
    if parent.get('case_ids')!=[0,1] or set(cases)!={0,1}:raise ValueError('Keep both original cases')
    for case,op in cases.items():
        expected=dict(status='failed_or_interrupted_q8_null_preserve_all',case_id=case,physical_seed=340800000+case,query=8,frame=32,live_reproduction_queries=9,reproduced_native_commands=32,extra_policy_queries=0,new_branch_commands=0,model_updates=0,branches={},error=ERROR)
        if any(op.get(k)!=v for k,v in expected.items()):raise ValueError('Only the known layout failure before intervention is eligible')
        old=parent['cases'][str(case)]
        if old.get('exit_code')!=1 or old.get('bounded_termination') is not False:raise ValueError('Unknown child outcomes cannot be retried')
    return first,end

def execution_plan(original,root):
    import copy
    root=PurePosixPath(root)
    if not root.is_absolute() or root.name not in {'PB_namespace2_null_recovery_20261113_v1','PB_namespace2_null_recovery_20261113_v2'}:raise ValueError('Independent recovery output required')
    out=copy.deepcopy(original);out['root']=str(root)
    for case,row in enumerate(out['cases']):
        if row['case_id']!=case:raise ValueError('Fixed case order differs')
        row['output']=str(root/f'case{case:02d}'/'results')
    return out


def validate_bootstrap_failure(parent,runner,cases,original_parent):
    if (parent.get('status')!='terminal_fixed_recovery_all_existing_output_archived'
            or runner.get('status')!='complete' or runner.get('exit_code')!=0
            or parent.get('first_admitted_utc')!=original_parent['first_admitted_utc']
            or parent.get('absolute_deadline_utc')!=original_parent['absolute_deadline_utc']
            or set(cases)!={0,1}):
        raise ValueError('Previous bootstrap owner must be known terminal in the same window')
    for case,op in cases.items():
        expected=dict(status='failed_or_interrupted_q8_null_preserve_all',case_id=case,
                      live_reproduction_queries=0,reproduced_native_commands=0,
                      extra_policy_queries=0,new_branch_commands=0,model_updates=0,branches={})
        if any(op.get(k)!=v for k,v in expected.items()) or not str(op.get('error','')).startswith('ImportError(') or '_mean_effect' not in str(op['error']):
            raise ValueError('Only the known zero-execution import error is eligible')
        old=parent['cases'][str(case)]
        if old.get('exit_code')!=1 or old.get('bounded_termination') is not False:
            raise ValueError('Unknown previous bootstrap outcome')
