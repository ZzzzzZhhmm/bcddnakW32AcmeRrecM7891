"""Run a new fixed PB study after a known-terminal CCI predecessor.

This supervisor does not access Git, import a model, replace a seed, or restart
an old phase clock. Qualification and factual collection use frozen executables.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,signal,subprocess,tarfile,time,traceback

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.datetime.now(datetime.timezone.utc)
def write(p,value):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.atomic');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def validate_stage_plan(p):
 if p.get('schema')!='warm.original-PB-new-stage-plan.v1' or p.get('task')!='press_button':raise ValueError('Wrong independent study')
 if (p.get('maximum_stage_seconds'),p.get('qualification_maximum_seconds'),p.get('physical_seed'),p.get('capture_replans'))!=(7200,2400,0,[8,48,96]):raise ValueError('Fixed PB stage limits or observations changed')
 if type(p.get('physical_seed')) is not int:raise ValueError('Fixed physical seed must be an integer')
 if p.get('predecessor_key')!='bt-40' or not p.get('original_windows_unchanged'):raise ValueError('Original phase must remain unchanged')
 return p

def archive(root,directory,label):
 root=Path(root);assert root.is_dir();directory.mkdir(exist_ok=True);files=[f for f in sorted(root.rglob('*')) if f.is_file()];assert all(not f.is_symlink() for f in files)
 m={f.relative_to(root).as_posix():dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files};a=directory/(label+'.tar.gz');assert not a.exists();write(directory/(label+'.manifest.json'),m)
 with tarfile.open(a,'w:gz') as t:
  for n in m:t.add(root/n,arcname=n,recursive=False)
 with tarfile.open(a) as t:
  assert set(t.getnames())==set(m)
  for n,v in m.items():
   data=t.extractfile(n).read();assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256'] and sha(root/n)==v['sha256']
 return dict(archive=str(a),sha256=sha(a),files=len(m),all_raw_archived=True)

def main():
 import fcntl
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True,type=Path);ap.add_argument('--expected-plan-sha256',required=True);cli=ap.parse_args();assert sha(cli.plan)==cli.expected_plan_sha256;p=validate_stage_plan(read(cli.plan))
 root=Path(p['stage_root']);root.mkdir(exist_ok=True);lease=(root/'supervisor.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);op=root/'operator.json';assert not op.exists(),'Existing stage needs explicit adjudication,not a duplicate run'
 state=dict(status='waiting_for_known_terminal_predecessor',prepared_plan_sha256=cli.expected_plan_sha256,started_supervisor_utc=now().isoformat(),model_inference_queries=0,qualification_started=False,trace_started=False)
 def save():state['updated_utc']=now().isoformat();write(op,state)
 save();child=None
 try:
  for v in p['frozen_inputs']:assert sha(v['path'])==v['sha256']
  deadline_admit=datetime.datetime.fromisoformat(p['latest_admission_utc']);pred=Path(p['predecessor_segment'])
  while True:
   if now()>=deadline_admit:raise TimeoutError('New study admission window expired;old study clock remains unchanged')
   if (pred/'exit.json').exists():
    terminal=read(pred/'exit.json');assert type(terminal.get('exit_code')) is int
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True,capture_output=True)
    if gpu.returncode==0 and not gpu.stdout.strip():
     campaign=read(Path(p['predecessor_shard'])/'campaign/campaign_state.json');pid=campaign.get('active_pid');alive=bool(pid and Path('/proc',str(pid)).exists())
     if not alive:break
   save();time.sleep(15)
  import shutil
  assert shutil.disk_usage(root).free>4*1024**3
  oldplan=read(Path(p['predecessor_shard'])/'plan.json')
  for f in [Path(oldplan['runtime_isolation']['persistent_lease_path']),Path(p['predecessor_shard'])/'campaign/campaign.lock',root.parent/'original_specialist_native_execution.worker.lock']:
   with f.open('a') as handle:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # Preserve complete or incomplete preceding attempts without changing labels.
  previous=[]
  for a in sorted((Path(p['predecessor_shard'])/'campaign/attempts').iterdir()):
   if a.is_dir() and a.name.startswith(('0045-full-','0046-')):
    assert (a/'exit.json').exists();previous.append(archive(a,root/'predecessor_archives',a.name))
  first=now();end=min(first+datetime.timedelta(seconds=7200),datetime.datetime.fromisoformat(p['latest_end_utc']));assert (end-first).total_seconds()>=6000
  admission=dict(schema='warm.original-PB-new-stage-admission.v1',first_admitted_utc=first.isoformat(),absolute_deadline_utc=end.isoformat(),qualification_plan_sha256=p['qualification_plan_sha256'],original_windows_unchanged=True,old_predecessor_exit=terminal,preceding_raw_archives=previous,scope='New original-PB12000 fixed first-state mechanism fixture;not original accuracy recovery or a reset of any previous run.')
  af=root/'admission.json';assert not af.exists();write(af,admission);state.update(status='qualification_pending',first_admitted_utc=first.isoformat(),absolute_deadline_utc=end.isoformat());save()
  def run(label,argv,maximum):
   nonlocal child
   assert not (root/(label+'_exit.json')).exists()
   with (root/(label+'_console.log')).open('x') as log:
    child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'))
    state.update(status=label+'_running',actual_child_pid=child.pid,active_argv=argv);save();limit=min(time.monotonic()+maximum,time.monotonic()+(end-now()).total_seconds());bounded=False
    while child.poll() is None:
     if time.monotonic()>=limit:
      bounded=True;os.killpg(child.pid,signal.SIGINT)
      try:child.wait(timeout=120)
      except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
      break
     save();time.sleep(5)
    result=dict(exit_code=child.returncode,bounded_termination=bounded,actual_pid=child.pid,ended_utc=now().isoformat(),argv=argv);write(root/(label+'_exit.json'),result);child=None;return result
  state['qualification_started']=True;save();r=run('qualification',[p['python'],'-B',p['qualification_entry'],'--plan',p['qualification_plan'],'--expected-plan-sha256',p['qualification_plan_sha256'],'--admission',str(af),'--expected-admission-sha256',sha(af)],2400)
  if Path(p['qualification_output']).is_dir():state['qualification_archive']=archive(p['qualification_output'],root/'archives','qualification')
  else:state['qualification_output_created']=False
  save()
  if r['exit_code']!=0:raise RuntimeError('Qualification failed;preserve fixed seed and all raw output')
  q=read(Path(p['qualification_output'])/'operator.json');assert q['status']=='complete_one_original_replan_four_native_commands' and q['inference_queries']==1 and q['native_commands']==4 and q['action_output_finite']
  trace=read(p['trace_template']);assert trace['capture_replans']==[8,48,96] and trace['physical_seed']==0 and trace['max_native_commands']==1500
  trace.update(first_admitted_utc=admission['first_admitted_utc'],absolute_deadline_utc=admission['absolute_deadline_utc'],qualification_operator_sha256=sha(Path(p['qualification_output'])/'operator.json'))
  tf=root/'trace.plan.json';assert not tf.exists();write(tf,trace);state['trace_started']=True;save();r=run('trace',[p['python'],'-B',p['trace_entry'],'--plan',str(tf),'--expected-plan-sha256',sha(tf)],max(1,(end-now()).total_seconds()-120))
  if Path(trace['output']).is_dir():state['trace_archive']=archive(trace['output'],root/'archives','trace')
  else:state['trace_output_created']=False
  save()
  if r['exit_code']!=0:raise RuntimeError('Trace failed or timed out;no seed/query replacement')
  result=read(Path(trace['output'])/'operator.json');assert result['status']=='complete_fixed_original_factual_episode'
  state.update(status='complete_fixed_PB_fixture_raw_archived',actual_success_flag=result['success_flag'],native_commands=result['native_commands'],replans=result['inference_queries'],population_success_rate_available=False,candidate_branch_labels_available=False);save()
 except BaseException as exc:
  state.update(status='failed_or_blocked_preserve_all',error=repr(exc),traceback=traceback.format_exc(),ended_utc=now().isoformat());save();raise
 finally:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGINT)
   try:child.wait(timeout=120)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
if __name__=='__main__':main()
