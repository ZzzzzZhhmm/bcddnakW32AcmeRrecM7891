#!/usr/bin/env python3
"""Execute a sealed native DEV q8 pool in independently recreated scenes.

No policy generation, physical unpack, Git access or result-based sampling.
All32 original valid candidates and repeated Full anchors are fixed beforehand.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib,importlib.util,json,os,pickle,random,signal,socket,sys,time,traceback

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def plain(items):return [{k:v.tolist() if hasattr(v,'tolist') else v for k,v in row.items()} for row in items]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seal',type=Path,required=True);ap.add_argument('--expected-seal-sha256',required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--admission',type=Path);ap.add_argument('--expected-admission-sha256')
    ap.add_argument('--resume-audit',type=Path);ap.add_argument('--expected-resume-audit-sha256')
    ap.add_argument('--preflight-isolation',type=Path,help='CPU-only check of another already qualified private worker')
    cli=ap.parse_args();seal_path=cli.seal.resolve(strict=True);sealed=seal_path.parent;seal=read(seal_path);ROOT=Path(__file__).resolve().parents[1];out=cli.output.resolve()
    if sha(seal_path)!=cli.expected_seal_sha256 or seal['schema']!='warm.rebuttal.native-DEV-query-seal.v1':raise ValueError('Fixed query seal differs')
    if (seal['task'],seal['query'],seal['horizon'],seal['candidate_slots'],seal['valid_candidates'])!=('rearrange_blocks',8,32,32,32) or seal['formal_TEST'] is not False:raise ValueError('Only the fixed native DEV q8/allK study is supported')
    if seal['seed']!=[3407,3408][seal['case_id']] or seal['expected_branch_order']!=['anchor-a','anchor-aa',*[f'rank-{i}' for i in range(32)],'anchor-final']:raise ValueError('Registered seed or complete branch order differs')
    manifest=read(sealed/'manifest.json')
    if {f.name for f in sealed.iterdir() if f.is_file()}!=set(manifest)|{'manifest.json'}:raise ValueError('Sealed export acquired extra/missing files')
    for name,item in manifest.items():
        if Path(name).name!=name or (sealed/name).stat().st_size!=item['bytes'] or sha(sealed/name)!=item['sha256']:raise ValueError('Sealed raw bytes differ')
    for name,item in seal['inputs'].items():
        f=Path(name)
        if f.stat().st_size!=item['bytes'] or sha(f)!=item['sha256']:raise ValueError('Original source input changed: '+name)
    source_plan_path=next(Path(n) for n in seal['inputs'] if Path(n).name=='plan.json');source_plan=read(source_plan_path)
    if sha(source_plan_path)!=seal['source_plan_sha256']:raise ValueError('Source plan identity differs')
    if out.is_relative_to(ROOT) or out.is_relative_to(sealed):raise ValueError('External output outside source and sealed inputs required')
    if out.exists()!=(cli.resume_audit is not None) or cli.preflight_only and cli.resume_audit is not None:raise ValueError('Existing output requires explicit terminal/archive adjudication;preflight must be fresh')
    H=load('frozen_native_helper',Path(seal['native_helper_code'])/'scripts/run_rebuttal_matched_native.py');admission=None;resume=None;runtime_plan=source_plan.copy()
    if cli.preflight_only:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        if cli.preflight_isolation:
            isolation=read(cli.preflight_isolation)
            if Path(isolation['runtime_root'])/'MATCHED_WORKER_ISOLATION.json'!=cli.preflight_isolation.resolve():raise ValueError('Private worker marker path differs')
            runtime_plan.update(native_runtime=isolation['runtime_root'],runtime_isolation=isolation)
    else:
        if cli.preflight_isolation is not None:raise ValueError('GPU runtime isolation must be frozen in the admission')
        if cli.admission is None or cli.expected_admission_sha256 is None or sha(cli.admission)!=cli.expected_admission_sha256:raise ValueError('Explicit immutable resource admission required')
        admission=read(cli.admission);start=datetime.datetime.fromisoformat(admission['first_admitted_utc']);end=datetime.datetime.fromisoformat(admission['absolute_deadline_utc']);now=datetime.datetime.now(datetime.timezone.utc)
        if admission['schema']!='warm.native-DEV-q8-pool-admission.v1' or (end-start).total_seconds()!=10800 or not start<=now<end:raise ValueError('Original bounded three-hour admission required')
        if admission.get('minimum_free_bytes')!=512*1024**2 or admission.get('maximum_raw_bytes')!=256*1024**2:raise ValueError('Fixed raw/storage protection required')
        if admission['sealed_query_sha256']!=cli.expected_seal_sha256 or Path(admission['output']).resolve()!=out:raise ValueError('Admission changed query or output')
        if cli.resume_audit:
            if cli.expected_resume_audit_sha256 is None or sha(cli.resume_audit)!=cli.expected_resume_audit_sha256:raise ValueError('Resume audit hash differs')
            resume=read(cli.resume_audit);old=read(out/'admission_snapshot.json')
            if (resume.get('schema')!='warm.native-query.known-terminal-recovery.v1' or resume.get('actual_processes_absent') is not True or resume.get('all_existing_raw_archived') is not True):raise ValueError('Unknown processes or unarchived raw evidence cannot be retried')
            if type(resume.get('known_exit_code')) is not int:
                previous_operator=read(out/'operator.json')
                if not (resume.get('known_exit_code') is None and resume.get('external_termination_observed') is True and resume.get('execution_platform')=='CCI' and resume.get('new_sshd_started_after_last_heartbeat') is True and resume.get('previous_execution_host')==resume.get('observed_execution_host')==previous_operator.get('execution_host')):raise ValueError('Uncollected exit requires explicit same-host external-termination evidence;never invent an exit code')
            if resume['previous_operator_sha256']!=sha(out/'operator.json') or resume['output']!=str(out) or resume['sealed_query_sha256']!=cli.expected_seal_sha256:raise ValueError('Resume belongs to another native query/invocation')
            if sha(resume['archive']['path'])!=resume['archive']['sha256']:raise ValueError('Previous raw recovery archive differs')
            for field in ['first_admitted_utc','absolute_deadline_utc','sealed_query_sha256','minimum_free_bytes','maximum_raw_bytes','output']:
                if admission[field]!=old[field]:raise ValueError('Recovery must preserve original query/window/storage: '+field)
            if sha(cli.admission)!=sha(out/'admission_snapshot.json') and admission.get('previous_admission_sha256')!=sha(out/'admission_snapshot.json'):raise ValueError('Implementation recovery must bind the original admission')
        prerequisite=admission['previous_stage_terminal_archive_receipt'];proof=read(prerequisite['path'])
        if sha(prerequisite['path'])!=prerequisite['sha256'] or proof['actual_processes_absent'] is not True or proof['all_existing_raw_archived'] is not True:raise ValueError('Previous rb40 stage remains active/unknown or unarchived')
        marker=read(admission['external_worker_reservation_path'])
        if marker['status']!='reserved_native_query_followup_before_first_process' or marker['query_output']!=str(out) or marker['sealed_query_sha256']!=cli.expected_seal_sha256:raise ValueError('Cross-client private-worker reservation differs')
        isolation=admission['runtime_isolation'];isolation_file=Path(isolation['runtime_root'])/'MATCHED_WORKER_ISOLATION.json'
        if read(isolation_file)!=isolation or sha(isolation_file)!=admission['runtime_isolation_sha256']:raise ValueError('Qualified private worker marker differs')
        runtime_plan.update(native_runtime=isolation['runtime_root'],runtime_isolation=isolation)
        lease=H.M.activate_native_isolation(runtime_plan)
    core,identity=H.install_core(runtime_plan)
    if out.is_relative_to(core) or out.is_relative_to(Path(seal['native_helper_code'])):raise ValueError('Output overlaps frozen scientific source')
    if admission and identity(ROOT)['sha256']!=admission['controller_source_sha256']:raise ValueError('Branch controller source changed')
    import multiprocessing as mp,numpy as np,torch
    from scripts.qualify_rebuttal_rmbench_task import resolved_config,source_identity
    from fastwam.models.warm.retrospection_model import _spatial_effect_summary
    from fastwam.research.factual_effect_reader import FrozenFactualEffectReader
    from fastwam.research.fresh_branches import OfficialQposGuard
    from fastwam.research.fresh_pool import atomic_json,read_verified_branch,seal_branch
    from fastwam.research.native_replay import native_parent,observation_arrays,require_observation,require_trace_step,state_hashes,physical_joint_state
    from fastwam.research.branches import InvalidBranchExecution
    from fastwam.research.rmbench_outcomes import observe_task_progress,label_next_interaction
    from fastwam.research.candidate_pool import as_array,centered_effect_metrics
    actions_file=sealed/'sealed_native_actions.npz'
    with np.load(actions_file,allow_pickle=False) as z:actions={k:z[k].copy() for k in z.files}
    if list(actions)!=seal['expected_branch_order'] or any(x.shape!=(32,14) or x.dtype!=np.float32 or not np.isfinite(x).all() for x in actions.values()):raise ValueError('All35 sealed finite native H32 branches required')
    if not np.array_equal(actions['anchor-a'],actions['anchor-aa']) or not np.array_equal(actions['anchor-a'],actions['anchor-final']):raise ValueError('Repeated Full actions differ')
    runtime=Path(runtime_plan['native_runtime']);runtime_before=source_identity(runtime)['sha256']
    if runtime_before!=seal['runtime_source_sha256']:raise ValueError('Private native runtime source changed')
    if resume:
        previous=out/'previous_invocations'/f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')}-{os.getpid()}";previous.mkdir(parents=True,exist_ok=False)
        (previous/'operator.json').write_bytes((out/'operator.json').read_bytes());(previous/'admission.json').write_bytes(cli.admission.read_bytes());(previous/'recovery.json').write_bytes(cli.resume_audit.read_bytes())
    else:out.mkdir()
    if admission and not resume:(out/'admission_snapshot.json').write_bytes(cli.admission.read_bytes())
    op=out/'operator.json';state=dict(status='inputs_verified',pid=os.getpid(),execution_host=socket.gethostname(),sealed_query_sha256=cli.expected_seal_sha256,completed_branches=[],completed_candidate_branches=0,independent_candidate_labels=0,formal_TEST=False,policy_loaded=False,physical_unpack=False)
    def save():state['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic_json(op,state)
    save()
    if cli.preflight_only:
        state.update(status='actual_seal_and_runtime_inputs_ready_no_truth',physical_branches_executed=0,DINO_loaded=False,resource_window_admitted=False,controller_source_sha256=identity(ROOT)['sha256'],qualified_runtime_marker=runtime_plan['runtime_isolation']);save()
        if torch.cuda.is_initialized():raise RuntimeError('Preflight unexpectedly initialized CUDA')
        print(json.dumps(state));return
    interrupted=False
    def stop(*_):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    last_storage_check=0.
    def budget():
        nonlocal last_storage_check
        import shutil
        if interrupted or datetime.datetime.now(datetime.timezone.utc)>=end:raise TimeoutError('Original query window ended;never reset it')
        if time.monotonic()-last_storage_check>=15:
            last_storage_check=time.monotonic()
            if shutil.disk_usage(out).free<admission['minimum_free_bytes']:raise OSError('Preserve raw evidence;native-query storage guard')
            if sum(f.stat().st_size for f in out.rglob('*') if f.is_file())>admission['maximum_raw_bytes']:raise OSError('Preserve raw evidence;native-query raw budget exceeded')
            if torch.cuda.mem_get_info()[0]<8*1024**3:raise OSError('Protect concurrent policy work;native-query GPU free-memory guard')
    torch.set_num_threads(4);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407);torch.cuda.set_per_process_memory_fraction(.12)
    state.update(status='loading_frozen_truth_reader',first_admitted_utc=admission['first_admitted_utc'],absolute_deadline_utc=admission['absolute_deadline_utc']);save()
    reader=FrozenFactualEffectReader.from_source_runtime(read(sealed/'runtime_args.json'),read(sealed/'online_contract.json'));atomic_json(out/'reader_receipt.json',reader.receipt)
    probe=torch.load(sealed/'source_probe.pt',map_location='cpu',weights_only=False)['probe']['arrays']
    with np.load(sealed/'factual_prefix.npz',allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
    physical_prefix=[json.loads(x) for x in (sealed/'factual_physical_prefix.jsonl').read_text().splitlines()];entry=read(sealed/'source_roster_entry.json')
    sys.path[:0]=[str(runtime),str(runtime/'description/utils'),source_plan['curobo_source']];os.chdir(runtime)
    from generate_episode_instructions import generate_episode_descriptions
    cls=getattr(importlib.import_module('envs.'+seal['task']),seal['task']);config=resolved_config(runtime,seal['task'],seal['seed']);config['now_ep_num']=seal['case_id']
    records={};env=None;children_before={x.pid for x in mp.active_children()};query_id=f"final3000-dev-rb-{seal['case_id']}-{seal['condition']}-q8"
    def record_progress():
        state['completed_branches']=list(records);state['completed_candidate_branches']=sum(k.startswith('rank-') for k in records)
        state['independent_candidate_labels']=sum(k.startswith('rank-') and r['independent_label']['applicable'] is not None for k,r in records.items());save()
    try:
        for branch_id,commands in actions.items():
            budget();branch_root=out/branch_id;completed=list(branch_root.glob('attempt-*/complete.json'))
            if len(completed)>1:raise ValueError('Duplicate completed branch;never select a favorable attempt')
            if completed:
                row=read_verified_branch(completed[0].parent,query_id=query_id,branch_id=branch_id)
                if row['sealed_query_sha256']!=cli.expected_seal_sha256 or records and row['parent_hashes']!=records['anchor-a']['parent_hashes']:raise ValueError('Completed branch belongs to another seal/parent')
                records[branch_id]=row;record_progress()
                if branch_id.startswith('anchor') and row['endpoint_status']!='complete_horizon':
                    state.update(status='fixed_Full_H32_anchor_unavailable_no_new_pool_claim',stopped_on=branch_id,remaining_candidates_unknown=32-state['independent_candidate_labels']);save();return
                if branch_id in ['anchor-aa','anchor-final'] and (row['endpoint_hashes']!=records['anchor-a']['endpoint_hashes'] or row['observed_effect']!=records['anchor-a']['observed_effect']):raise RuntimeError('Previously mismatched Full anchors remain quarantined')
                continue
            old_attempts=list(branch_root.glob('attempt-*'))
            if old_attempts and (resume is None or not all(str(x) in resume['adjudicated_incomplete_attempts'] for x in old_attempts)):raise ValueError('Incomplete/unknown branch attempt must be archived and adjudicated before retry')
            attempt=branch_root/f'attempt-{len(old_attempts)+1:03d}';attempt.mkdir(parents=True,exist_ok=False);atomic_json(attempt/'attempt.json',dict(branch_id=branch_id,query_id=query_id,pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            random.seed(seal['seed']);np.random.seed(seal['seed']);torch.manual_seed(seal['seed']);torch.cuda.manual_seed_all(seal['seed']);env=cls();env.setup_demo(**config)
            if type(env.scene.physx_system).__name__!='PhysxCpuSystem' or env.eval_video_path is not None or env.save_data or env.render_freq:raise ValueError('Pinned native CPU physics/no render-recording required')
            instruction=str(np.random.choice(generate_episode_descriptions(seal['task'],[entry['expert_info']['info']],100)[0]['unseen']));env.set_instruction(instruction=instruction)
            if instruction!=entry['initial_identity']['instruction'] or plain(physical_joint_state(env))!=entry['physical_initial']:raise ValueError('Original fixed initial physical state or instruction differs')
            with np.load(sealed/'initial_observation.npz',allow_pickle=False) as z:require_observation({k:z[k] for k in z.files},observation_arrays(env.get_obs()))
            with OfficialQposGuard(env) as guard:
                for i,command in enumerate(trace['commands']):
                    budget();before=np.asarray(env.get_obs()['joint_action']['vector']).copy()
                    if plain(physical_joint_state(env))!=physical_prefix[2*i]['physical_before']:raise ValueError('Source measured prefix-before joints differ')
                    if not guard.step(command):raise RuntimeError('Original native prefix stopped')
                    require_trace_step(trace,i,before,np.asarray(env.get_obs()['joint_action']['vector']).copy())
                    if plain(physical_joint_state(env))!=physical_prefix[2*i+1]['physical_after']:raise ValueError('Source measured prefix-after joints differ')
            obs=env.get_obs()
            with np.load(sealed/'query_observation.npz',allow_pickle=False) as z:require_observation({k:z[k] for k in z.files},observation_arrays(obs))
            if plain(physical_joint_state(env))!=read(sealed/'expected_query_physical.json'):raise ValueError('Actual source q8 physical state differs')
            parent=native_parent(env,obs);ph=state_hashes(parent)
            if records and ph!=records['anchor-a']['parent_hashes']:raise RuntimeError('Fresh branch native/RNG parent differs;quarantine')
            before=reader.encode(obs);parent_outcome=observe_task_progress(env)
            if state_hashes(native_parent(env,env.get_obs()))!=ph:raise RuntimeError('Truth read mutated native parent')
            if records and not np.array_equal(before,np.load(out/'parent_dino_spatial.npy',allow_pickle=False)):raise RuntimeError('Frozen parent features differ')
            if not records:np.save(out/'parent_dino_spatial.npy',before,allow_pickle=False)
            (attempt/'parent.pickle').write_bytes(pickle.dumps(parent,protocol=4));np.save(attempt/'parent_dino_spatial.npy',before,allow_pickle=False)
            count=0;status='complete_horizon';error=None
            with (attempt/'commands.jsonl').open('x',buffering=1) as stream,OfficialQposGuard(env) as guard:
                for index,command in enumerate(commands):
                    budget();pb=plain(physical_joint_state(env))
                    try:accepted=guard.step(command)
                    except InvalidBranchExecution as exc:status='invalid_branch';error=str(exc);break
                    if not accepted:status='invalid_branch';error='official executor stopped';break
                    stream.write(json.dumps(dict(step=index,command=command.tolist(),physical_before=pb,physical_after=plain(physical_joint_state(env))))+'\n');count+=1
                    if env.eval_success and count<32:status='early_success';break
            endpoint=native_parent(env,env.get_obs());eh=state_hashes(endpoint);after=reader.encode(env.get_obs());outcome=observe_task_progress(env)
            if state_hashes(native_parent(env,env.get_obs()))!=eh:raise RuntimeError('Truth read mutated endpoint')
            effect=_spatial_effect_summary(torch.from_numpy(after-before)[None],torch.from_numpy(before)[None])[0].numpy() if status=='complete_horizon' else None
            row=dict(query_id=query_id,branch_id=branch_id,parent_recreated=True,parent_hashes=ph,endpoint_hashes=eh,endpoint_status=status,error=error,executed_steps=count,observed_effect=None if effect is None else effect.tolist(),parent_outcome=parent_outcome,endpoint_outcome=outcome,independent_label=label_next_interaction(parent_outcome,outcome,valid_branch=status!='invalid_branch'),sealed_query_sha256=cli.expected_seal_sha256)
            (attempt/'endpoint.pickle').write_bytes(pickle.dumps(endpoint,protocol=4));np.save(attempt/'endpoint_dino_spatial.npy',after,allow_pickle=False);seal_branch(attempt,row);records[branch_id]=read_verified_branch(attempt,query_id=query_id,branch_id=branch_id)
            env.close_env(clear_cache=True);env=None;record_progress()
            if branch_id.startswith('anchor') and status!='complete_horizon':
                state.update(status='fixed_Full_H32_anchor_unavailable_no_new_pool_claim',stopped_on=branch_id,unavailable_reason=error,remaining_candidates_unknown=32-state['independent_candidate_labels']);save();return
            if branch_id in ['anchor-aa','anchor-final'] and (eh!=records['anchor-a']['endpoint_hashes'] or row['observed_effect']!=records['anchor-a']['observed_effect']):raise RuntimeError('Repeated Full endpoint/effect differs;quarantine')
        assembled=dict(task=seal['task'],episode_id=seal['case_id'],query_id=query_id,required_effect=as_array(probe['required_effect'])[0].tolist(),candidates=[])
        for rank in range(32):
            row=records[f'rank-{rank}'];assembled['candidates'].append(dict(candidate_id=f'rank-{rank}',valid=True,horizon=32,predicted_effect=as_array(probe['predicted_effect'])[0,rank].tolist(),historical_effect=as_array(probe['historical_effect'])[0,rank].tolist(),observed_effect=row['observed_effect'],endpoint_status=row['endpoint_status'],executed_steps=row['executed_steps'],applicable=row['independent_label']['applicable'],label_provenance=row['independent_label']['label_provenance']))
        atomic_json(out/'candidate_effects.json',assembled);atomic_json(out/'centered_effects.json',centered_effect_metrics(assembled));state.update(status='complete_all35_fixed_native_branches',complete_H32_candidates=sum(r['endpoint_status']=='complete_horizon' for k,r in records.items() if k.startswith('rank-')),full_anchors_exact=True,actual_missing_skill_certified=False,noninferiority_claimed=False);save()
    except BaseException as exc:state.update(status='unknown_runtime_or_guard_failure',error=repr(exc),traceback=traceback.format_exc());save();raise
    finally:
        if env is not None:
            try:env.close_env(clear_cache=True)
            except BaseException as exc:state.update(cleanup_error=repr(exc));save()
        for child in mp.active_children():
            if child.pid not in children_before:child.terminate();child.join(timeout=5)
        if identity(core)['sha256']!=seal['evaluation_source_sha256'] or source_identity(runtime)['sha256']!=runtime_before:
            state.update(status='unknown_frozen_source_changed');save();raise RuntimeError('Frozen scientific source changed;do not admit the query result')
        state['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    print(json.dumps(state))

if __name__=='__main__':main()
