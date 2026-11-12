"""Read-only zero-GPU audit of the separately registered two-source PB study."""
from pathlib import Path
import argparse,csv,datetime,fcntl,json,sys,time,traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fastwam.research.pb_pool_audit import audit_pool,read,sha,verify_archive


def now():return datetime.datetime.now(datetime.timezone.utc)
def write(path,value):
    temporary=Path(path).with_suffix('.atomic');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--expected-plan-sha256',required=True);ap.add_argument('--validate-only',action='store_true');cli=ap.parse_args()
    assert sha(cli.plan)==cli.expected_plan_sha256;p=read(cli.plan)
    assert p['schema']=='warm.PB-namespace2-offline-audit.v1' and p['GPUs']==0 and type(p['GPUs']) is int
    assert p['case_ids']==[0,1] and p['queries']==[8,48,96] and p['scientific_jobs_read_only'] is True
    for row in p['stage_plans']+p['inputs']:assert sha(row['path'])==row['sha256']
    stages=[read(row['path']) for row in p['stage_plans']];assert [s['case_id'] for s in stages]==[0,1]
    deadline=datetime.datetime.fromisoformat(p['absolute_deadline_utc']);out=Path(p['output'])
    for s in stages:assert not out.resolve().is_relative_to(Path(s['stage_root']).resolve())
    if cli.validate_only:
        print(json.dumps(dict(status='namespace_CPU_audit_plan_validated',GPUs=0,new_policy_or_physical_actions=0)));return
    out.mkdir(exist_ok=False);lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='waiting_for_fixed_namespace_source_terminals',started_utc=now().isoformat(),GPUs=0,new_policy_or_physical_actions=0,scientific_jobs_read_only=True,reports={})
    try:
        while True:
            terminal=[]
            for stage in stages:
                root=Path(stage['stage_root']);f=root/'operator.json';operator=read(f) if f.is_file() else None
                terminal.append(bool(operator and operator['status'].startswith(('terminal_','failed_or_interrupted_'))))
                for query in p['queries']:
                    key=f"case{stage['case_id']}-q{query}"
                    if key in state['reports']:continue
                    row=operator.get('queries',{}).get(str(query)) if operator else None
                    if row and row['status'] in ('terminal_pool_archived','incomplete_pool_archived_no_replacement','CPU_seal_failed_fixed_query_retained'):
                        planref=row['plan'];assert sha(planref['path'])==planref['sha256'];pool=read(planref['path'])
                        archive=verify_archive(pool['output'],row['archive'])
                        if row['status']=='terminal_pool_archived' and row['exit_code']==0:
                            report=audit_pool(pool['output'],query=query,plan_sha256=planref['sha256'],namespace_plan=pool)
                            report.update(status='verified_fixed_namespace_pool',archive=archive,namespace_case_id=stage['case_id'],physical_seed=stage['physical_seed'])
                        else:report=dict(status='incomplete_pool_retained',namespace_case_id=stage['case_id'],query=query,archive=archive,primary_metrics=None,raw=row,no_imputation=True)
                        target=out/(key+'.json');assert not target.exists();write(target,report);state['reports'][key]=dict(status=report['status'],report=str(target),sha256=sha(target))
                    elif row and row['status'] in ('not_reached_no_replacement','not_started_original_hard_window_insufficient'):
                        state['reports'][key]=dict(status=row['status'],primary_metrics=None)
                    elif terminal[-1]:state['reports'][key]=dict(status='source_or_query_incomplete_no_replacement',primary_metrics=None,source_status=operator['status'])
            state['updated_utc']=now().isoformat();write(out/'operator.json',state)
            if all(terminal) or now()>=deadline:break
            time.sleep(15)
        state.update(status='terminal_namespace_offline_audit',ended_utc=now().isoformat(),all_six_verified=len(state['reports'])==6 and all(r['status']=='verified_fixed_namespace_pool' for r in state['reports'].values()))
        with (out/'centered_effects.csv').open('x',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['case_id','physical_seed','query','complete_H32','valid_K','reference','centered_MSE'])
            for row in state['reports'].values():
                if row['status']!='verified_fixed_namespace_pool':continue
                report=read(row['report']);metrics=report['metrics']
                for name,value in metrics['references'].items():writer.writerow([report['namespace_case_id'],report['physical_seed'],report['query'],metrics['complete_endpoints'],metrics['valid_candidates'],name,value['centered_mse']])
        write(out/'operator.json',state);print(json.dumps(state))
    except BaseException as exc:
        state.update(status='failed_namespace_offline_audit_preserve_raw',error=repr(exc),traceback=traceback.format_exc(),ended_utc=now().isoformat());write(out/'operator.json',state);raise

if __name__=='__main__':main()
