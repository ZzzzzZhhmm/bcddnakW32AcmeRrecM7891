"""Derive descriptive logged wall times on a read-only zero-GPU ACP worker."""
from pathlib import Path
import argparse,csv,datetime,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fastwam.research.native_timing_audit import audit_plan,read,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--expected-plan-sha256',required=True);p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    if sha(a.plan)!=a.expected_plan_sha256:raise ValueError('Frozen plan identity differs')
    plan=read(a.plan)
    if a.validate_only:
        if plan['GPUs']!=0 or len(plan['roles'])!=plan['expected_complete_roles']:raise ValueError('Zero-GPU snapshot differs')
        print(json.dumps(dict(status='plan_validated_not_audited',GPUs=0)));return
    output=Path(plan['output']);output.mkdir(exist_ok=False)
    report,rows=audit_plan(plan)
    with (output/'logged_replans.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['% Descriptive collector wall times; includes raw evidence capture; no hardware-matched overhead claim.']
    for name,g in report['groups'].items():
        steady=g['steady']['pipeline_s'];fmt=lambda key:f"{1000*steady[key]:.1f}" if steady else 'NA'
        lines.append(name.replace('_',r'\_')+f" & {g['roles']} & {g['all']['pipeline_s']['n']} & {1000*g['first']['pipeline_s']['median_s']:.1f} & {fmt('median_s')} & {fmt('p95_s')}"+r' \\')
    (output/'timing_rows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report.update(audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=a.expected_plan_sha256)
    (output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['complete_roles','complete_pairs','logged_replans','GPUs','groups','release']}))


if __name__=='__main__':main()
