"""Freeze two original-PB expert-admitted namespace cases; no learned policy.

The ordinal/lower-bound distinction is explicit. All rejected candidates and
runtime errors remain recorded. A runtime error never admits a replacement.
"""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,importlib,importlib.util,json,os,sys,traceback

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('PB_namespace',ROOT/'src/fastwam/research/pb_namespace.py')
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)

def now():return datetime.datetime.now(datetime.timezone.utc)
def plain(value):
    if isinstance(value,dict):return {str(key):plain(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [plain(item) for item in value]
    if hasattr(value,'tolist'):return plain(value.tolist())
    if value is None or type(value) in (str,int,float,bool):return value
    raise TypeError('Unspecified expert metadata: '+type(value).__name__)

def task_state(env):
    keys=['card_id_1','card_id_2','press_cnt_1','press_cnt_2','press_cnt_check_button','press_flag_1','press_flag_2','press_flag_check_button']
    return {key:plain(getattr(env,key)) for key in keys}

def physical(env):return [dict(name=actor.name,qpos=actor.get_qpos().tolist(),qvel=actor.get_qvel().tolist()) for actor in env.scene.get_all_articulations()]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--expected-registration-sha256',required=True)
    parser.add_argument('--admission',type=Path);parser.add_argument('--expected-admission-sha256');parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();assert B.sha(args.registration)==args.expected_registration_sha256
    registration=B.validate_registration(B.read(args.registration))
    for reference in registration['roster_inputs']:assert B.sha(reference['path'])==reference['sha256']
    helper=Path(registration['native_setup_helper']);spec=importlib.util.spec_from_file_location('PB_roster_setup',helper);setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup)
    runtime=Path(registration['roster_runtime']);source=setup.source_identity(runtime)
    assert source['sha256']==registration['simulator_source_sha256']=='a97e308ca053c3aeb1bfec8e3c4c4f550aac7f6ae753e36bbce82e9522292379'
    if args.validate_only:
        print(json.dumps(dict(status='namespace_and_runtime_CPU_validated',next_candidate_seed=340800000,requested_admitted_cases=2,model_queries=0,physical_commands=0)));return
    assert args.admission and B.sha(args.admission)==args.expected_admission_sha256
    admission=B.read(args.admission);assert admission['schema']=='warm.PB-namespace2-roster-admission.v1' and admission['registration_sha256']==args.expected_registration_sha256
    first=datetime.datetime.fromisoformat(admission['first_admitted_utc']);end=datetime.datetime.fromisoformat(admission['absolute_deadline_utc'])
    assert 0<(end-first).total_seconds()<=1800 and first<=now()<end
    out=Path(registration['roster_output']);out.mkdir(parents=True,exist_ok=False)
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    legacy_lock=Path(registration['old_original_worker_lease']).open('a');fcntl.flock(legacy_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='expert_only_initializing',started_utc=now().isoformat(),registration_sha256=args.expected_registration_sha256,namespace_file_sha256=B.NAMESPACE_SHA256,learned_policy_queries=0,admission_records=[],entries=[],original_diagnostic_seed0_unchanged=True)
    def save():
        state['updated_utc']=now().isoformat();temp=out/'operator.atomic';temp.write_text(json.dumps(state,indent=2));temp.replace(out/'operator.json')
    def budget():
        if now()>=end:raise TimeoutError('Original namespace roster window expired;do not reset it')
    save();env=None
    try:
        import numpy as np,random,torch
        sys.path[:0]=[str(runtime),str(runtime/'description/utils'),registration['curobo_source']];os.chdir(runtime)
        from envs.utils.create_actor import UnStableError
        from generate_episode_instructions import generate_episode_descriptions
        cls=getattr(importlib.import_module('envs.press_button'),'press_button')
        def seed_all(seed):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        namespace=np.load(registration['namespace_file'],allow_pickle=False);bounds=B.namespace_bounds(namespace)
        for offset in range(2000):
            budget();seed=int(bounds[0])+offset;case=len(state['entries']);cfg=setup.resolved_config(runtime,'press_button',seed);cfg['now_ep_num']=case
            row=dict(seed=seed,learned_policy_queries=0,candidate_offset=offset,case_id_if_admitted=case)
            state.update(status='expert_setup_candidate',active_seed=seed);save()
            try:
                seed_all(seed);env=cls();env.setup_demo(**cfg);info=env.play_once()
                accepted=bool(env.plan_success and env.check_success())
                row.update(status='accepted' if accepted else 'expert_rejected',plan_success=bool(env.plan_success))
                env.close_env(clear_cache=True);env=None
            except UnStableError as exc:
                row.update(status='unstable_seed',error=repr(exc));accepted=False
                if env is not None:env.close_env(clear_cache=True);env=None
            except Exception as exc:
                row.update(status='runtime_unknown',error=repr(exc));state['admission_records'].append(row);save();raise
            state['admission_records'].append(row);save()
            with (out/'admission.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
            if not accepted:continue
            identities=[]
            for repeat in range(2):
                budget();seed_all(seed);env=cls();env.setup_demo(**cfg);assert env.step_lim==1500
                instruction=str(np.random.choice(generate_episode_descriptions('press_button',[info['info']],100)[0]['unseen']))
                env.set_instruction(instruction=instruction);obs=env.get_obs()
                identity=dict(observation_arrays=B.observation_identity(obs),instruction=instruction,physical_initial=physical(env),task_state_audit_only=task_state(env))
                arrays={key:np.asarray(value['rgb']) for key,value in obs['observation'].items() if 'rgb' in value};arrays['native_qpos']=np.asarray(obs['joint_action']['vector'])
                with (out/f'case{case:02d}-initial-{repeat}.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
                identities.append(identity);env.close_env(clear_cache=True);env=None
            assert identities[0]==identities[1],'Two independent native initial resets differ;no policy trial admitted'
            entry=dict(case_id=case,seed=seed,namespace_lower_bound=int(bounds[case]),expert_info=plain(info),initial_identity=identities[0],two_initial_resets_exact=True)
            state['entries'].append(entry);state.update(status='accepted_case_initial_resets_verified');save()
            if len(state['entries'])==2:break
        expected=B.first_admitted(state['admission_records'],namespace)
        assert len(state['entries'])==2 and all(all(entry[key]==expected[i][key] for key in expected[i]) for i,entry in enumerate(state['entries']))
        assert setup.source_identity(runtime)==source
        state.update(status='qualified_original_PB_namespace_roster',ended_utc=now().isoformat(),simulator_source_sha256=source['sha256'],policy_outcomes_used_for_selection=False)
        save();target=out/'roster.json';assert not target.exists();target.write_text(json.dumps(state,indent=2));print(json.dumps(dict(status=state['status'],entries=expected,model_queries=0)))
    except BaseException as exc:
        state.update(status='failed_or_interrupted_expert_roster',error=repr(exc),traceback=traceback.format_exc(),ended_utc=now().isoformat());save();raise
    finally:
        if env is not None:
            try:env.close_env(clear_cache=True)
            except Exception:pass

if __name__=='__main__':main()
