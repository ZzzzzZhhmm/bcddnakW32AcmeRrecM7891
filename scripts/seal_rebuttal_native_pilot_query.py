#!/usr/bin/env python3
"""Seal actual native DEV candidates and their factual prefix before branch truth.

This CPU-only export loads no policy, simulator or DINO model. It is not an
outcome experiment and never converts factual future actions into an H32 branch.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,os,shutil,sys,tarfile

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def write(p,v):
    with Path(p).open('x') as f:f.write(json.dumps(v,indent=2)+'\n')

def verify_archive(attempt,expected):
    receipt=read(attempt/'archive_receipt.json');archive=attempt/'verified_worker_evidence.tar.gz';worker=attempt/'worker'
    if receipt['sha256']!=expected or sha(archive)!=expected or archive.stat().st_size!=receipt['bytes']:raise ValueError('Fixed source archive differs')
    with tarfile.open(archive,'r:gz') as tar:
        names=[m.name for m in tar.getmembers()]
        if len(names)!=len(set(names)) or names.count('archive_manifest.json')!=1:raise ValueError('Ambiguous source archive')
        manifest=json.load(tar.extractfile('archive_manifest.json'))
        if set(names)!=set(manifest)|{'archive_manifest.json'} or len(manifest)!=receipt['files']:raise ValueError('Incomplete source archive')
        if {f.relative_to(worker).as_posix() for f in worker.rglob('*') if f.is_file()}!=set(manifest):raise ValueError('Source worker changed after archival')
        for name,item in manifest.items():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts or not tar.getmember(name).isfile():raise ValueError('Unsafe source archive member')
            value=tar.extractfile(name).read()
            if len(value)!=item['bytes'] or hashlib.sha256(value).hexdigest()!=item['sha256'] or sha(worker/name)!=item['sha256']:raise ValueError('Original source bytes differ')
    return receipt

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-attempt',required=True,type=Path);ap.add_argument('--source-plan',required=True,type=Path)
    ap.add_argument('--expected-plan-sha256',required=True);ap.add_argument('--expected-source-archive-sha256',required=True)
    ap.add_argument('--query-ordinal',type=int,choices=[8],required=True);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();source=args.source_attempt.resolve(strict=True);worker=source/'worker';out=args.output.resolve();ROOT=Path(__file__).resolve().parents[1]
    if out.exists() or out.is_relative_to(ROOT) or out.is_relative_to(source):raise ValueError('Fresh Git-external output outside the source role required')
    if sha(args.source_plan)!=args.expected_plan_sha256:raise ValueError('Source plan changed')
    plan=read(args.source_plan)
    if plan.get('schema')!='warm.rebuttal.native-family-dev-pilot.v1' or plan.get('phase')!='native_family_holdout_development_pilot' or plan.get('formal_TEST') is not False:raise ValueError('Separate fixed native DEV source required')
    if plan.get('registered_pilot_seeds')!=[3407,3408] or plan.get('inference')!=dict(horizon=32,execution_prefix=4,nfe=10,top_k=32,recent_capacity=20,summary_capacity=8):raise ValueError('Registered fixed DEV or inference budget differs')
    if plan['task']!='rearrange_blocks' or plan['roles']['full']['checkpoint_sha256']!='b0669047fbcb0f8f79fd25fea2de25f689cfd286d85ae899872c9b4435fff27b':raise ValueError('Different task or fixed-final model')
    receipt=verify_archive(source,args.expected_source_archive_sha256);r=read(worker/'result.json');typed=read(worker/'typed_policy_receipt.json');exit=read(source/'exit.json')
    if exit['exit_code']!=0 or exit['bounded_termination'] or r['status']!='complete_native':raise ValueError('Known complete native source required')
    if r['phase']!=plan['phase'] or r['plan_sha256']!=args.expected_plan_sha256 or r['role']!='full' or typed['factory_role']!='full' or typed['intervention_role']!='full':raise ValueError('Source is not natural Full under the fixed plan')
    if r['checkpoint_sha256']!=plan['roles']['full']['checkpoint_sha256'] or typed['model_class']!='WarmRetrospectionFastWAM' or typed['checkpoint_update']!=3000:raise ValueError('Actual source model binding differs')
    case=r['episode_id'];seed=r['seed'];query=args.query_ordinal;frame=query*4
    if case not in [0,1] or seed!=[3407,3408][case] or r['search_condition'] not in plan['conditions']:raise ValueError('Source case or search condition is not registered')
    if (r['success']==0 and r['executed_commands']!=700) or (r['success']==1 and not 0<r['executed_commands']<=700):raise ValueError('Incomplete source outcome cannot be repackaged')
    if len(r['queries'])<=query or r['queries'][query]['frame']!=frame:raise ValueError('Prespecified q8 was not reached;never replace it')
    # Install the independently frozen common core before importing identity helpers.
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    H=load('frozen_native_helper',Path(plan['native_helper_code'])/'scripts/run_rebuttal_matched_native.py');core,identity=H.install_core(plan)
    if out.is_relative_to(core) or out.is_relative_to(Path(plan['native_helper_code'])):raise ValueError('Output overlaps immutable scientific source')
    import numpy as np,torch
    from fastwam.memory.robotwin_artifacts import RobotwinQposZScore
    from fastwam.research.candidate_pool import prepare_candidate_pool,decode_branch_actions
    runtime=read(worker/'runtime_args.json');contract=read(runtime['warm_online_contract_path'])
    if sha(runtime['dataset_stats_path'])!=contract['normalization_stats_sha256']:raise ValueError('Original normalization contract differs')
    normalizer=RobotwinQposZScore.from_dataset_stats(read(runtime['dataset_stats_path']))
    probe_path=worker/f'pre-execution-probe-q{query:04d}.pt';payload=torch.load(probe_path,map_location='cpu',weights_only=False);probe=payload['probe'];arrays=probe['arrays']
    if payload['before_any_local_branch_truth'] is not True or probe['force_null'] is not False or probe['source_mode']!='full':raise ValueError('Natural source candidates were not sealed before truth')
    rows=np.asarray(payload['bank_rows']);qrow=r['queries'][query]
    if rows.shape!=(32,) or rows.dtype.kind not in 'iu' or rows.tolist()!=qrow['bank_rows']:raise ValueError('Actual allK event identities differ')
    if float(arrays['g'].float().reshape(-1)[0])!=qrow['effective_gate']:raise ValueError('Source probe and query receipt differ')
    commands,native,decode_receipt=prepare_candidate_pool(probe,normalizer)
    if list(commands)!=[f'rank-{i}' for i in range(32)]:raise ValueError('All32 original valid ranks are required;no favorable subset')
    generation=worker/f'generations/replan-{query:04d}/model_actions.npy';value=np.load(generation,allow_pickle=False)
    if value.shape==(1,32,14):value=value[0]
    full=decode_branch_actions(normalizer,value)
    with np.load(worker/'qpos_trace.npz',allow_pickle=False) as z:
        if not np.array_equal(full[:4],z['commands'][frame:frame+4]):raise ValueError('Original generated Full prefix differs from actual source commands')
        prefix={k:z[k][:frame].copy() for k in ['commands','before','after']}
    context_path=worker/f'context-frame{frame:04d}.pt';context=torch.load(context_path,map_location='cpu',weights_only=True)
    if {'target_action','target_effect','current_semantic_teacher','target_action_valid_mask','future_valid_mask'}&set(context):raise ValueError('Future supervision in an inference context')
    for a,b in [('source','actual_source'),('base_gaussian','actual_base_gaussian'),('conditioning','actual_conditioning_tokens')]:
        if not torch.equal(arrays[a],context[b]):raise ValueError('Probe and actual generation context differ: '+a)
    all_actions={'anchor-a':full,'anchor-aa':full,**commands,'anchor-final':full};out.mkdir()
    with (out/'sealed_native_actions.npz').open('xb') as f:np.savez_compressed(f,**all_actions)
    with (out/'factual_prefix.npz').open('xb') as f:np.savez_compressed(f,**prefix)
    physical=(worker/'physical_commands.jsonl').read_bytes().splitlines(keepends=True)
    if len(physical)<2*(frame+4):raise ValueError('Actual physical prefix unavailable')
    parsed=[json.loads(x) for x in physical[:2*frame]]
    for i in range(frame):
        a,b=parsed[i*2:i*2+2]
        if a['index']!=b['index'] or a['index']!=i or b['error'] is not None or b['observed_TOPP_errors']:raise ValueError('Actual source prefix was not accepted by the native executor')
    (out/'factual_physical_prefix.jsonl').write_bytes(b''.join(physical[:2*frame]));write(out/'expected_query_physical.json',json.loads(physical[2*frame])['physical_before'])
    roster=read(plan['roster']);entry=roster['entries'][case]
    if entry['seed']!=seed or entry['status']!='qualified_fixed_development_reset':raise ValueError('Source reset provenance differs')
    write(out/'source_roster_entry.json',entry)
    copies={'source_probe.pt':probe_path,'source_context.pt':context_path,'source_generation.npy':generation,'initial_observation.npz':worker/'initial_observation.npz','query_observation.npz':worker/f'observation-frame{frame:04d}.npz','runtime_args.json':worker/'runtime_args.json','online_contract.json':Path(runtime['warm_online_contract_path'])}
    for name,path in copies.items():shutil.copy2(path,out/name);assert sha(out/name)==sha(path)
    inputs={str(path):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in [args.source_plan,source/'archive_receipt.json',worker/'result.json',worker/'typed_policy_receipt.json',probe_path,context_path,generation,worker/'qpos_trace.npz',worker/'physical_commands.jsonl',Path(plan['roster']),Path(runtime['dataset_stats_path'])]}
    seal=dict(schema='warm.rebuttal.native-DEV-query-seal.v1',task=r['task'],case_id=case,seed=seed,condition=r['search_condition'],query=query,frame=frame,horizon=32,expected_branch_order=list(all_actions),candidate_bank_rows=rows.tolist(),candidate_slots=32,valid_candidates=32,source_checkpoint_sha256=r['checkpoint_sha256'],source_archive=receipt,source_plan_sha256=args.expected_plan_sha256,evaluation_code=str(core),evaluation_source_sha256=plan['evaluation_source_sha256'],native_helper_code=plan['native_helper_code'],native_helper_source_sha256=plan['controller_source_sha256'],runtime_source_sha256=plan['runtime_source_sha256'],effect_projection='Unchanged shared-stage _spatial_effect_summary from the frozen C core;not the original specialist uniform mean.',decode_receipt=decode_receipt,inputs=inputs,before_any_new_local_branch_truth=True,physical_branches_executed=0,independent_candidate_labels=0,candidate_null_generation_available=False,original_saved_retrieval_tensor_available=False,formal_TEST=False,policy_loaded=False,simulator_loaded=False,DINO_model_loaded=False,resource_window_admitted=False,sampling_scope='Prespecified q8 of one registered DEV seed/condition. A factual four-command prefix binds the generated H32;its full endpoint still needs independent execution.')
    write(out/'sealed_query.json',seal)
    manifest={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in out.iterdir() if f.is_file()};write(out/'manifest.json',manifest)
    if torch.cuda.is_initialized():raise RuntimeError('CPU-only sealing unexpectedly initialized CUDA')
    print(json.dumps(dict(status='all32_native_candidates_and_factual_prefix_sealed_no_new_truth',output=str(out),case_id=case,seed=seed,condition=r['search_condition'],query=query,branches=35,sealed_query_sha256=sha(out/'sealed_query.json'),manifest_sha256=sha(out/'manifest.json'),policy_loaded=False,simulator_loaded=False,resource_window_admitted=False)))

if __name__=='__main__':main()
