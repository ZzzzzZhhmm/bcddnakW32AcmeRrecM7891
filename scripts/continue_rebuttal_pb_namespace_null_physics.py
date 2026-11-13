"""Execute existing sealed q8 actions in a fresh physical-only process."""
from pathlib import Path
import argparse,datetime,importlib.util,json,copy
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--expected-plan-sha256',required=True);ap.add_argument('--case',type=int,choices=[0,1],required=True);ap.add_argument('--validate-only',action='store_true');a=ap.parse_args()
    guard=load('PB_sealed_physical_guard',ROOT/'src/fastwam/research/pb_null_sealed_physics.py');assert guard.sha(a.plan)==a.expected_plan_sha256;p=guard.read(a.plan)
    row=guard.validate(p,a.case,datetime.datetime.now(datetime.timezone.utc))
    if a.validate_only:print(json.dumps(dict(status='known_postseal_OOM_inputs_bound',policy_queries=0,physics=0)));return
    base=load('PB_sealed_physical_base',ROOT/'scripts/run_rebuttal_pb_namespace_null.py');old=guard.read(guard.reference(p['original_plan']));original=base.BIND.validate_plan(old)
    execution=copy.deepcopy(original);execution['root']=p['root'];execution['cases'][a.case]['output']=row['output']
    admission=dict(absolute_deadline_utc=p['active_end_utc'])
    base.run_case(execution,a.case,admission,sealed_recovery=row)
if __name__=='__main__':main()
