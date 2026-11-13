"""Resource-only recovery of the already registered, unstarted PB q96 pools."""
from pathlib import Path, PurePosixPath
import datetime as dt
import hashlib
import json

SCHEMA='warm.PB-namespace2-unstarted-q96-stage.v1'
OLD_UIDS={0:'0f0e8eeeba3544b181a73cacc62d7eae',1:'d5aab172af5746c5b89923e4dc86d2c0'}


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_name(case):
    if type(case) is not int or case not in OLD_UIDS:raise ValueError('Only the two registered namespace cases may continue')
    return f'warm-pb-ns2-q96-case{case}-20261113-a'


def require_old_terminal(operator,runner):
    if (runner.get('status')!='failed' or type(runner.get('exit_code')) is not int
            or runner['exit_code']!=-15 or not runner.get('end_utc')):
        raise ValueError('The original scientific runner must have its actual SIGTERM exit receipt')
    first=dt.datetime.fromisoformat(operator['first_admitted_utc'])
    oldend=dt.datetime.fromisoformat(operator['absolute_deadline_utc'])
    stopped=dt.datetime.fromisoformat(runner['end_utc'])
    if any(v.tzinfo is None for v in [first,oldend,stopped]) or not first<stopped<=oldend:
        raise ValueError('Original admission and terminal clocks must remain bound')
    if (oldend-first).total_seconds()>36000 or '96' in operator.get('queries',{}):
        raise ValueError('Do not restart an owned q96 or extend the original source window')
    return first,oldend,stopped


def bounded_deadline(first,old_first,old_end,old_stopped):
    if any(v.tzinfo is None for v in [first,old_first,old_end,old_stopped]):raise ValueError('Timezone-aware clocks required')
    used=(old_stopped-old_first).total_seconds()
    if not 0<used<36000 or first<old_stopped:raise ValueError('Original elapsed allocation is not a refund, overlapping owner or a new clock')
    end=min(first+dt.timedelta(seconds=min(14400,36000-used)),old_end)
    if (end-first).total_seconds()<600:raise ValueError('Too little original-window time for a new q96 allocation')
    return end


def require_q96_plan(base,current):
    changed={k for k in set(base)|set(current) if base.get(k)!=current.get(k)}
    if (base.get('query_replan')!=48 or current.get('query_replan')!=96
            or not changed <= {'query_replan','selected_rank','output'}
            or current['output']!=str(PurePosixPath(base['output']).with_name('pool-q0096'))):
        raise ValueError('Only the original predeclared q96 query, its sealed rank and assigned output may differ')
    return changed


def validate_plan(p):
    expected=dict(schema=SCHEMA,query=96,maximum_seconds=14400,shutdown_margin_seconds=180,
        automatic_restarts=0,GPUs=1,CPUs=8,RAM_GiB=128,resource_type='idle',
        original_windows_unchanged=True,source_policy_forwards=0,model_updates=0,
        q48_interrupted_branches_never_retried=True,source_or_query_replacement=False,
        minimum_admission_free_bytes=4*1024**3,runtime_free_space_guard_bytes=512*1024**2,
        cumulative_reserved_GPU_hours=143.8054115725,lifetime_cap_GPU_hours=144)
    if any(type(p.get(k)) is not type(v) or p.get(k)!=v for k,v in expected.items()):
        raise ValueError('Recovery cannot change the registered science, budget, resource type or shutdown allowance')
    if p.get('job_name')!=expected_name(p.get('case_id')) or p.get('old_UID')!=OLD_UIDS[p['case_id']]:
        raise ValueError('Case/platform ownership differs')
    return p


def validate_cloud_gate(p):
    gate=read(p['cloud_terminal_gate'])
    expected=dict(schema='warm.PB-namespace2-old-platform-terminal.v1',case_id=p['case_id'],old_UID=p['old_UID'],
                  old_platform_terminal=True,old_scientific_exit=-15,new_job_name=p['job_name'])
    if any(type(gate.get(k)) is not type(v) or gate.get(k)!=v for k,v in expected.items()):
        raise ValueError('Fresh authenticated platform-terminal confirmation is required before recovery')
    ref=gate['authenticated_platform_receipt']
    if set(ref)!={'path','sha256'} or sha(ref['path'])!=ref['sha256']:
        raise ValueError('The cloud confirmation must retain its authenticated source receipt')
    return gate


def require_prepared_CPU_seal(value,expected_plan_sha256):
    expected=dict(status='complete_PB_CPU_input_seal',plan_sha256=expected_plan_sha256,
                  query=96,CUDA_initialized=False,candidate_branches_executed=0,model_updates=0,new_policy_generations=0)
    if any(type(value.get(k)) is not type(v) or value.get(k)!=v for k,v in expected.items()):
        raise ValueError('Only the bound no-CUDA/no-physical q96 input seal may be reused')
    order=value.get('branch_order',[])
    if len(order)!=36 or len(set(order))!=36:raise ValueError('The original32candidate/fourFull order must remain complete')
    return value
