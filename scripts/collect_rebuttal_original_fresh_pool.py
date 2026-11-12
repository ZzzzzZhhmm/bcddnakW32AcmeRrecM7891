"""Collect presealed original-specialist query effects after fresh A-A-B-A.

Model source, descriptor, candidates, source trajectory and physical seed are
frozen before any branch truth. No policy is loaded or updated here. Unsupported
BRT applicability labels stay unknown. No Git access or worktree gate occurs.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import sys
import traceback



BT_POOL_SCHEMA='warm.original-BT-fixed-cohort-fresh-pool.v1'


def validate_original_pool_plan(plan):
    query=int(plan['query_replan']);selected=int(plan['selected_rank'])
    if plan['schema']=='warm.original-specialist-fresh-pool.v1':
        if query not in (8,48,96) or selected!={8:28,48:0,96:10}[query]:
            raise ValueError('Original BRT query/rank protocol differs')
        return dict(task='blocks_ranking_try',physical_seed=0,is_BT=False)
    if plan['schema']!=BT_POOL_SCHEMA:
        raise ValueError('Unknown original-specialist branch protocol')
    expected=dict(task_name='battery_try',physical_seed=340800000,root_seed=3407,
        cohort_case_id=0,checkpoint_step=12000,horizon=32,execution_prefix=4,
        checkpoint_sha256='27c137d689007b38f6204d3dd12bb038550b1e58403506d7a9894bd6fcd43a9f',
        roster_sha256='612cb5d1728b2bebef7c09ee61bd50933c64821be31e1df0a7aa475b61d666b8',
        factual_source_plan_sha256='9d479fa7d1dedcc120edff3e38561d84c46d8214a649a436d004ecbbce72aec1')
    if any(plan.get(k)!=v for k,v in expected.items()) or query not in (8,48,96) or not 0<=selected<32:
        raise ValueError('Original BT fixed cohort/checkpoint/query binding differs')
    if plan.get('next_interaction_rubric')!='warm.rmbench.physical-next-interaction.v1':
        raise ValueError('Independent BT physical-outcome rubric must be declared before branch truth')
    return dict(task='battery_try',physical_seed=340800000,is_BT=True)


def validate_BT_factual_source(operator,trace,plan):
    required=dict(status='complete_fixed_original_factual_episode',candidate_seed=340800000,
        cohort_case_id=0,source_sha256='ff32342d286929ddc5904d55814c66e965269f6b749c1e1b6df1060aaef0b7fe',
        source_sha256_after='ff32342d286929ddc5904d55814c66e965269f6b749c1e1b6df1060aaef0b7fe',
        checkpoint_sha256='27c137d689007b38f6204d3dd12bb038550b1e58403506d7a9894bd6fcd43a9f',
        initial_observation_exact_to_qualification=True,first_replan_exact_to_qualification=True,model_updates=0)
    if any(operator.get(k)!=v for k,v in required.items()):raise ValueError('Original BT factual source is not qualified and complete')
    count=operator['native_commands']
    if len(trace)!=count or not 0<count<=1000 or (count!=1000 and not operator.get('success_flag')):
        raise ValueError('Only an official BT native-limit or success-terminated source is complete')
    frame=plan['query_replan']*4
    if count<frame+4:raise ValueError('Fixed original BT query does not have its full executed prefix;no replacement')
    if [r['index'] for r in trace]!=list(range(count)) or any(r['take_action_cnt']!=r['index']+1 for r in trace):
        raise ValueError('Original BT trace is incomplete or its native counters differ')
    if not plan.get('factual_prefix_manifest') and any(r.get('observed_TOPP_errors') for r in trace[:frame]):
        raise ValueError('Original BT factual prefix requires a separately verified two-pass rejection audit')


def plain_physical_state(value):
    return [{k:v.tolist() if hasattr(v,'tolist') else v for k,v in item.items()} for item in value]


def paired_BT_repair_summary(records,selected):
    full=records['anchor-a'];raw=records[f'rank-{selected}']
    a,b=raw['independent_label']['applicable'],full['independent_label']['applicable']
    return dict(scope='One fixed source state;dependent candidate outcomes,not population success rate',
        selected_rank=selected,independent_source_states=1,raw_label=a,refined_label=b,
        raw_endpoint_status=raw['endpoint_status'],refined_endpoint_status=full['endpoint_status'],
        paired_known=a is not None and b is not None,
        repaired=(a==0 and b==1) if a is not None and b is not None else None,
        harmed=(a==1 and b==0) if a is not None and b is not None else None,
        next_interaction_rubric='warm.rmbench.physical-next-interaction.v1',
        automatic_withdrawal_causal_comparison_available=False,population_effect_estimated=False)


def verified_prefix_manifest(plan, source, frame):
    """Bind a separately audited factual context;never classify candidate truth."""
    path = plan.get('factual_prefix_manifest')
    if path is None:
        return None
    path = Path(path)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(path) != plan['factual_prefix_manifest_sha256']:
        raise ValueError('Factual prefix manifest changed')
    value = json.loads(path.read_text())
    required = ('all_per_command_source_API_states_exact',
                'all_replan_source_RGB_observations_exact',
                'all_between_pass_measured_joint_states_exact',
                'final_parent_rng_hashes_exact')
    if (value['status'] != 'two_factual_prefix_replays_exact_to_source_observations_and_each_other'
            or value['query'] != plan['query_replan'] or value['frame'] != frame
            or value['commands_each_pass'] != frame or value['new_candidate_endpoints'] != 0
            or value['query_plan_sha256'] != plan['factual_prefix_base_plan_sha256']
            or any(value.get(k) is not True for k in required)
            or value['source_trace_sha256'] != digest(source/'native_trace.jsonl')):
        raise ValueError('Factual prefix identity or exactness evidence differs')
    passes = value['pass_files']
    if len(passes) != 2 or passes[0]['sha256'] != passes[1]['sha256']:
        raise ValueError('Two byte-identical observed prefix passes required')
    expected = {v['index']:v['error'] for v in value['prefix_errors']}
    if len(expected) != len(value['prefix_errors']) or any(not 0 <= i < frame for i in expected):
        raise ValueError('Duplicate or foreign factual-prefix rejection index')
    source_rows = [json.loads(s) for s in (source/'native_trace.jsonl').read_text().splitlines()]
    for item in passes:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Factual-prefix raw replay changed')
        rows = [json.loads(s) for s in Path(item['path']).read_text().splitlines()]
        if len(rows) != frame:
            raise ValueError('Incomplete factual-prefix replay')
        for i,row in enumerate(rows):
            if (row['index'] != i or row['error'] != expected.get(i)
                    or row['guard_accepted'] != (i not in expected)
                    or any(row[k] != source_rows[i][k] for k in ('command','before','after','take_action_cnt'))):
                raise ValueError('Factual-prefix observed command or rejection differs')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--resume-known-terminal', action='store_true')
    parser.add_argument('--qualify-only', action='store_true', help='Stop after the fixed four-branch A-A-B-A;retain the same sealed plan for the later pool')
    parser.add_argument('--null-generation-seal', type=Path)
    parser.add_argument('--expected-null-generation-seal-sha256')
    parser.add_argument('--preflight-only', action='store_true')
    cli = parser.parse_args()
    import fcntl
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda p: json.loads(Path(p).read_text())
    stamp = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    assert sha(cli.plan) == cli.expected_plan_sha256
    plan = read(cli.plan)
    null_seal = None
    if cli.null_generation_seal is not None:
        assert sha(cli.null_generation_seal) == cli.expected_null_generation_seal_sha256
        null_seal = read(cli.null_generation_seal)
        assert null_seal['schema'] == 'warm.original-q8-explicit-null-action-seal.v1'
        assert null_seal['query'] == 8 and null_seal['frame'] == 32
        assert all(null_seal[k] is True for k in ['Full_actions_and_wrapper_source_fields_exact',
            'Full_null_Gaussian_exact','withdrawn_generator_candidate_invariance_exact',
            'final_Full_actions_and_source_exact','model_parameters_and_buffers_unchanged',
            'explicit_intervention','synthetic_variation_has_no_physical_branch','before_any_new_null_truth'])
        assert null_seal['physical_branches'] == 0 and null_seal['policy_inferences'] == 4
        assert null_seal['natural_withdrawal_established'] is False
        study = plan['null_study'];base_path = Path(study['base_plan'])
        assert sha(base_path) == study['base_plan_sha256']
        base = read(base_path)
        assert base['schema'] == BT_POOL_SCHEMA and base['query_replan'] == 8
        for key,value in base.items():
            if key not in ['output','first_admitted_utc','absolute_deadline_utc','inputs']:
                assert plan[key] == value, key
        assert null_seal['checkpoint_sha256'] == base['checkpoint_sha256']
        previous = Path(base['output'])
        assert read(previous/'operator.json')['status'] == 'complete_original_fixed_query_pool'
        assert sha(previous/'branches/anchor-a/attempt-001/complete.json') == study['original_anchor_complete_sha256']
        assert sha(Path(study['generation_archive'])) == study['generation_archive_sha256']
    elif plan.get('null_study') is not None:
        raise ValueError('Explicit null seal required for the registered withdrawal sub-study')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='true',DIFFSYNTH_MODEL_BASE_PATH=plan['model_root'],TOKENIZERS_PARALLELISM='false')
    protocol=validate_original_pool_plan(plan);task=protocol['task'];physical_seed=protocol['physical_seed'];is_BT=protocol['is_BT']
    query = int(plan['query_replan']); selected = int(plan['selected_rank']); frame = query * 4
    query_id = f'original-bt-cohort-case0-q{query}' if is_BT else f'original-brt-seed0-q{query}'
    C, H, runtime, source, out = [Path(plan[k]) for k in (
        'original_source', 'reference_controller', 'rmbench_runtime',
        'factual_source', 'output')]
    assert not out.is_relative_to(C) and not out.is_relative_to(H)
    if out.exists() and not cli.resume_known_terminal:
        raise FileExistsError('Inspect the previous invocation; explicit known-terminal resume required')
    out.mkdir(exist_ok=True)
    lock = (out/'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = datetime.datetime.fromisoformat(plan['absolute_deadline_utc'])
    def budget():
        if datetime.datetime.now(datetime.timezone.utc) >= deadline:
            raise TimeoutError('Original branch resource window expired; do not reset it')
    budget()
    sys.path[:0] = [str(C/'src'), str(C)]
    import fastwam
    fastwam.__path__.append(str(H/'src/fastwam'))
    import fastwam.research
    fastwam.research.__path__.insert(0, str(H/'src/fastwam/research'))
    import numpy as np
    import torch
    import multiprocessing as mp
    from fastwam.memory.manifest import sha256_array
    from fastwam.memory.robotwin_artifacts import RobotwinQposZScore
    from fastwam.models.warm.retrospection_model import _mean_effect
    from experiments.rmbench.warm_policy.policy_core import factual_cameras, factual_joint_state
    from fastwam.research.candidate_pool import decode_branch_actions, centered_effect_metrics
    from fastwam.research.factual_effect_reader import FrozenFactualEffectReader
    from fastwam.research.fresh_branches import OfficialQposGuard, require_repeated_endpoints
    from fastwam.research.fresh_pool import atomic_json, read_verified_branch, seal_branch
    from fastwam.research.native_replay import native_parent, state_hashes, physical_joint_state, require_observation
    from fastwam.research.branches import InvalidBranchExecution
    from fastwam.research.rmbench_outcomes import observe_task_progress,label_next_interaction
    spec = importlib.util.spec_from_file_location('frozen_source_identity', H/'scripts/run_rebuttal_training_with_runtime_trace.py')
    identity = importlib.util.module_from_spec(spec); spec.loader.exec_module(identity)
    spec = importlib.util.spec_from_file_location('frozen_native_setup', H/'scripts/qualify_rebuttal_rmbench_task.py')
    setup = importlib.util.module_from_spec(spec); spec.loader.exec_module(setup)
    assert identity.source_digest(C) == plan['original_source_sha256']
    assert identity.source_digest(H) == plan['reference_controller_sha256']
    assert setup.source_identity(runtime)['sha256'] == plan['simulator_source_sha256']
    for module in ('fastwam.memory.online_retrieval', 'fastwam.models.warm.retrospection_model', 'experiments.rmbench.warm_policy.policy_core'):
        assert Path(sys.modules[module].__file__).resolve().is_relative_to(C)
    assert sha(C/'src/fastwam/models/warm/retrospection_model.py') == plan['original_model_module_sha256']
    operator = dict(status='binding', started_utc=stamp(), plan_sha256=cli.expected_plan_sha256,
        descriptor='original_uniform_mean_DINO_delta_v1', query=query, selected_rank=selected,
        complete_branches=0, independent_candidate_endpoints=0, applicability_labels=0,
        source_checkpoint_sha256=plan['checkpoint_sha256'], new_efficacy_claimed=False)
    def save():
        operator['updated_utc'] = stamp(); atomic_json(out/'operator.json', operator)
    save(); env = None; children_before = {p.pid for p in mp.active_children()}
    try:
        for row in plan['inputs']:
            assert sha(row['path']) == row['sha256'], row['path']
        full = read(source/'operator.json')
        assert full['status'] == 'complete_fixed_original_factual_episode'
        if not is_BT:assert full['native_commands']==600
        args = read(source/'runtime_args.json'); contract = read(args['warm_online_contract_path'])
        assert args['task_name'] == task
        trace = [json.loads(s) for s in (source/'native_trace.jsonl').read_text().splitlines()]
        prefix_manifest = verified_prefix_manifest(plan, source, frame)
        expected_prefix_errors = {v['index']:v['error'] for v in prefix_manifest['prefix_errors']} if prefix_manifest else {}
        if is_BT:validate_BT_factual_source(full,trace,plan)
        else:assert len(trace)==600
        commands = np.asarray([r['command'] for r in trace], dtype=np.float32)
        before_trace = np.asarray([r['before'] for r in trace], dtype=np.float32)
        after_trace = np.asarray([r['after'] for r in trace], dtype=np.float32)
        probe = source/f'probes/q{query:04d}'
        event = torch.load(probe/'event_adapter.pt', map_location='cpu', weights_only=False)
        model = torch.load(probe/'model_output.pt', map_location='cpu', weights_only=False)
        valid = event['candidate_valid_mask'].numpy()[0]
        assert valid.shape == (32,) and valid.dtype == np.bool_ and valid[selected]
        assert model['warm_retrospection']['selected_candidate_index'] == selected
        adapted = event['adapted_action_mean'].float().numpy()[0]
        predicted = event['predicted_effect'].float().numpy()[0]
        assert adapted.shape == (32,32,14) and predicted.shape == (32,768)
        normalizer = RobotwinQposZScore.from_dataset_stats(read(args['dataset_stats_path']))
        refined = decode_branch_actions(normalizer, np.load(source/f'generations/q{query:04d}/model_actions.npy', allow_pickle=False))
        assert np.array_equal(refined[:4], commands[frame:frame+4])
        native = {f'rank-{r}': decode_branch_actions(normalizer, adapted[r]) for r in np.flatnonzero(valid)}
        order = ['anchor-a','anchor-aa',f'rank-{selected}','anchor-qual-final']
        order += [f'rank-{r}' for r in np.flatnonzero(valid) if r != selected]
        order += ['anchor-pool-final']
        actions = {name: refined if name.startswith('anchor-') else native[name] for name in order}
        if null_seal is not None:
            assert is_BT and query == 8
            native_file = cli.null_generation_seal.parent/'sealed_native_actions.npz'
            assert sha(native_file) == null_seal['sealed_native_actions_sha256']
            with np.load(native_file,allow_pickle=False) as sealed_null:
                assert set(sealed_null.files) == {'Full','candidate-null','Full-final'}
                assert np.array_equal(sealed_null['Full'],refined) and np.array_equal(sealed_null['Full-final'],refined)
                null_actions = sealed_null['candidate-null'].copy()
            assert null_actions.shape == (32,14) and null_actions.dtype == np.float32 and np.isfinite(null_actions).all()
            order = ['anchor-a','candidate-null','anchor-final']
            actions = {'anchor-a':refined,'candidate-null':null_actions,'anchor-final':refined}
            original_anchor = read_verified_branch(previous/'branches/anchor-a/attempt-001',query_id=query_id,branch_id='anchor-a')
        # Exact source bank is checked before any new physical truth.
        bank = Path(args['warm_bank_directory']); manifest = read(bank/'manifest.json')
        assert sha(bank/'manifest.json') == contract['bank_manifest_sha256']
        assert sha(bank/manifest['payload_file']) == manifest['content_hashes'][manifest['payload_file']]
        rows = np.asarray(model['warm_online_telemetry']['retrieval']['bank_rows'], dtype=np.int64)
        ids = model['warm_online_telemetry']['retrieval']['ranked_event_ids']
        with np.load(bank/manifest['payload_file'], allow_pickle=False) as loaded:
            offsets = loaded['_event_dataset_id_offsets']; utf8 = loaded['_event_dataset_id_utf8']
            ds = loaded['_event_dataset_index']; eps = loaded['_event_episode_index']; starts = loaded['_event_start_frame']
            for j,r in enumerate(rows):
                if not valid[j]: continue
                actual = dict(dataset_id=bytes(utf8[offsets[r]:offsets[r+1]]).decode(), dataset_index=int(ds[r]), episode_index=int(eps[r]), start_frame=int(starts[r]))
                assert actual == ids[j], (j,actual,ids[j])
            pre = loaded['effect_pre'][rows].copy(); post = loaded['effect_post'][rows].copy()
        historical = _mean_effect(torch.from_numpy(post-pre)).float().numpy()
        assert historical.shape == (32,768)
        seal = dict(query=query, frame=frame, source_plan_sha256=plan['factual_source_plan_sha256'],
            checkpoint_sha256=plan['checkpoint_sha256'], descriptor='original_uniform_mean_DINO_delta_v1',
            before_any_new_branch_truth=True, selected_rank=selected, original_valid_ranks=np.flatnonzero(valid).tolist(),
            branch_order=order, source_inputs=plan['inputs'], expected_source_observation=str(source/f'replan-{query:04d}.npz'),
            action_hashes={k:sha256_array(v) for k,v in actions.items()},
            predicted_sha256=sha256_array(predicted), historical_sha256=sha256_array(historical),
            required_effect_available=False, unsupported_applicability_is_unknown=not is_BT,
            independent_BT_rubric=plan.get('next_interaction_rubric') if is_BT else None,
            rng_protocol='Restore the sealed source pre-forward Python/NumPy/Torch RNG at each query parent;no physical unpack.')
        if not is_BT:seal.pop('independent_BT_rubric',None)
        if null_seal is not None:
            seal.update(study='original_same_parent_Full_explicit_null_Full',
                null_generation_seal_sha256=cli.expected_null_generation_seal_sha256,
                original_query_plan_sha256=study['base_plan_sha256'],
                original_anchor_complete_sha256=study['original_anchor_complete_sha256'],
                new_candidate_endpoints=0,natural_withdrawal_established=False,independent_source_states=1)
        if (out/'sealed_query.json').exists():
            assert read(out/'sealed_query.json') == seal
            with np.load(out/'sealed_arrays.npz', allow_pickle=False) as saved:
                assert np.array_equal(saved['predicted'],predicted) and np.array_equal(saved['historical'],historical)
                assert all(np.array_equal(saved[k],v) for k,v in actions.items())
        else:
            np.savez_compressed(out/'sealed_arrays.npz', predicted=predicted, historical=historical, valid=valid, **actions)
            atomic_json(out/'sealed_query.json', seal)
        if cli.preflight_only:
            assert not torch.cuda.is_initialized()
            operator.update(status='complete_CPU_original_branch_input_preflight',physical_branches=0,
                branch_order=order,CUDA_initialized=False,null_generation_seal_verified=null_seal is not None)
            save();print(json.dumps(operator));return
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='true',TOKENIZERS_PARALLELISM='false')
        torch.manual_seed(3407); torch.cuda.manual_seed_all(3407); torch.cuda.set_per_process_memory_fraction(.12)
        operator['status'] = 'loading_original_frozen_teacher'; save()
        reader = FrozenFactualEffectReader.from_source_runtime(args, contract)
        if (out/'reader_receipt.json').exists(): assert read(out/'reader_receipt.json') == reader.receipt
        else: atomic_json(out/'reader_receipt.json', reader.receipt)
        asset_plan = read(Path(plan['evidence_root'])/'resumable_rb_cases0003_0009_v1_plan.json')
        sys.path[:0] = [str(runtime), str(runtime/'description/utils'), asset_plan['curobo_source']]
        os.chdir(runtime)
        from generate_episode_instructions import generate_episode_descriptions
        cls = getattr(importlib.import_module('envs.'+task),task)
        cfg = setup.resolved_config(runtime,task,physical_seed)
        assert cfg == read(source/'native_config.json')
        expert = read(source/'expert_setup_metadata.json'); initial = read(source/'initial_identity.json')
        rng = pickle.loads((probe/'sampling_rng_before.pkl').read_bytes())
        def factual_arrays(obs): return {**factual_cameras(obs), 'native_qpos': factual_joint_state(obs)}
        def require_source(path, obs):
            with np.load(path,allow_pickle=False) as saved:
                require_observation({k:saved[k] for k in saved.files}, factual_arrays(obs))
        def require_step(index,before,after):
            if not np.array_equal(before,before_trace[index]) or not np.array_equal(after,after_trace[index]):
                raise RuntimeError('Native API replay mismatch at command '+str(index))
        records = {}
        for name in order:
            budget(); root = out/'branches'/name; root.mkdir(parents=True,exist_ok=True)
            completed = list(root.glob('attempt-*/complete.json'))
            assert len(completed) <= 1, 'Do not choose among duplicate terminal branches'
            if completed:
                row = read_verified_branch(completed[0].parent,query_id=query_id,branch_id=name)
                assert row['sealed_query_sha256'] == sha(out/'sealed_query.json')
            else:
                attempt = root/f'attempt-{len(list(root.glob("attempt-*")))+1:03d}'; attempt.mkdir()
                atomic_json(attempt/'attempt.json', dict(branch=name,pid=os.getpid(),utc=stamp()))
                operator.update(status='recreating_fixed_parent',active_branch=name); save()
                random.seed(physical_seed); np.random.seed(physical_seed); torch.manual_seed(physical_seed); torch.cuda.manual_seed_all(physical_seed)
                env = cls(); env.setup_demo(**cfg)
                assert type(env.scene.physx_system).__name__ == 'PhysxCpuSystem' and not env.eval_video_path and not env.save_data and not env.render_freq
                instruction = str(np.random.choice(generate_episode_descriptions(task,[expert['info']],100)[0]['unseen']))
                assert instruction == initial['instruction']; env.set_instruction(instruction=instruction)
                require_source(source/'initial_observation.npz',env.get_obs())
                observed_prefix_errors = [];seen_combinations=[];prefix_rows=[]
                if is_BT:
                    initial_progress=observe_task_progress(env)
                    if all(initial_progress['in_slots']):seen_combinations.append(initial_progress['combination'])
                with OfficialQposGuard(env) as guard:
                    for index,command in enumerate(commands[:frame]):
                        budget(); before = factual_joint_state(env.get_obs()).copy()
                        measured_before=plain_physical_state(physical_joint_state(env)) if is_BT else None
                        error = None
                        try:
                            assert guard.step(command)
                        except InvalidBranchExecution as exc:
                            error = str(exc)
                            if error != expected_prefix_errors.get(index):
                                raise
                            observed_prefix_errors.append(dict(index=index,error=error))
                        if error != expected_prefix_errors.get(index):
                            raise RuntimeError('Expected factual-prefix rejection did not recur')
                        assert env.take_action_cnt == trace[index]['take_action_cnt']
                        require_step(index,before,factual_joint_state(env.get_obs()))
                        if is_BT:
                            measured_after=plain_physical_state(physical_joint_state(env))
                            assert measured_before==trace[index]['physical_before'] and measured_after==trace[index]['physical_after'],'Original BT measured factual prefix differs'
                            progress=observe_task_progress(env)
                            if all(progress['in_slots']) and progress['combination'] not in seen_combinations:seen_combinations.append(progress['combination'])
                            prefix_rows.append(dict(index=index,command=command.tolist(),before=before.tolist(),after=factual_joint_state(env.get_obs()).tolist(),physical_before=measured_before,physical_after=measured_after,task_progress=progress,error=error))
                        if (prefix_manifest or is_BT) and (index+1)%4 == 0:
                            require_source(source/f'replan-{(index+1)//4:04d}.npz',env.get_obs())
                assert observed_prefix_errors == (prefix_manifest['prefix_errors'] if prefix_manifest else [])
                atomic_json(attempt/'factual_prefix_audit.json',dict(command_attempts=frame,
                    observed_rejections=observed_prefix_errors,
                    manifest_sha256=plan.get('factual_prefix_manifest_sha256'),
                    candidate_execution_validity_unchanged=True))
                if is_BT:
                    with (attempt/'BT_factual_prefix_trace.jsonl').open('x') as stream:
                        for row in prefix_rows:stream.write(json.dumps(row)+'\n')
                obs = env.get_obs(); require_source(source/f'replan-{query:04d}.npz',obs)
                random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch_cpu']); torch.cuda.set_rng_state_all(rng['torch_cuda'])
                parent = native_parent(env,obs); parent_hash = state_hashes(parent)
                if prefix_manifest:
                    assert parent_hash == prefix_manifest['parent_hashes'], 'Audited factual parent differs'
                if records: assert parent_hash == records['anchor-a']['parent_hashes'], 'Fresh native parents differ'
                before_feature = reader.encode(obs)
                assert state_hashes(native_parent(env,env.get_obs())) == parent_hash, 'Teacher read changed parent'
                if records: assert np.array_equal(before_feature,np.load(out/'parent_dino_spatial.npy',allow_pickle=False))
                else: np.save(out/'parent_dino_spatial.npy',before_feature,allow_pickle=False)
                (attempt/'parent.pickle').write_bytes(pickle.dumps(parent,protocol=4)); np.save(attempt/'parent_dino_spatial.npy',before_feature,allow_pickle=False)
                parent_progress=observe_task_progress(env) if is_BT else None
                if is_BT:assert state_hashes(native_parent(env,env.get_obs()))==parent_hash,'Independent task reader changed the parent'
                count=0; status='complete_horizon'; error=None
                operator['status']='executing_sealed_branch'; save()
                with (attempt/'commands.jsonl').open('x',buffering=1) as stream, OfficialQposGuard(env) as guard:
                    for index,command in enumerate(actions[name]):
                        budget(); native_api_before=factual_joint_state(env.get_obs()).copy(); physical_before=physical_joint_state(env)
                        try: executed=guard.step(command)
                        except InvalidBranchExecution as exc: status='invalid_branch'; error=str(exc); break
                        if not executed: status='invalid_branch'; error='Official executor stopped'; break
                        physical_after=physical_joint_state(env); count+=1
                        def plain(value):
                            return [{k:v.tolist() if hasattr(v,'tolist') else v for k,v in item.items()} for item in value]
                        stream.write(json.dumps(dict(index=index,command=command.tolist(),physical_before=plain(physical_before),physical_after=plain(physical_after)))+'\n')
                        if name.startswith('anchor-') and index<4:
                            require_step(frame+index,native_api_before,factual_joint_state(env.get_obs()))
                        if name.startswith('anchor-') and index==3: require_source(source/f'replan-{query+1:04d}.npz',env.get_obs())
                        if env.eval_success and count<32: status='early_success'; break
                endpoint=native_parent(env,env.get_obs()); endpoint_hash=state_hashes(endpoint)
                after_feature=reader.encode(env.get_obs())
                assert state_hashes(native_parent(env,env.get_obs())) == endpoint_hash, 'Teacher read changed endpoint'
                effect=_mean_effect(torch.from_numpy(after_feature-before_feature)[None])[0].numpy() if status=='complete_horizon' else None
                (attempt/'endpoint.pickle').write_bytes(pickle.dumps(endpoint,protocol=4)); np.save(attempt/'endpoint_dino_spatial.npy',after_feature,allow_pickle=False)
                row=dict(query_id=query_id,branch_id=name,parent_recreated=True,parent_hashes=parent_hash,endpoint_hashes=endpoint_hash,
                    executed_steps=count,endpoint_status=status,error=error,observed_effect=effect.tolist() if effect is not None else None,
                    independent_label=label_next_interaction(parent_progress,observe_task_progress(env),valid_branch=status!='invalid_branch',seen_combinations=seen_combinations) if is_BT else dict(applicable=None,label_provenance='BRT rubric not yet qualified;unknown preserved'),
                    BT_parent_progress=parent_progress,BT_seen_combinations_before_query=seen_combinations if is_BT else None,BT_endpoint_progress=observe_task_progress(env) if is_BT else None,
                    endpoint_native_success=bool(env.eval_success),sealed_query_sha256=sha(out/'sealed_query.json'))
                if not is_BT:
                    for k in ('BT_parent_progress','BT_seen_combinations_before_query','BT_endpoint_progress'):row.pop(k,None)
                if is_BT:assert state_hashes(native_parent(env,env.get_obs()))==endpoint_hash,'Independent task reader changed the endpoint'
                seal_branch(attempt,row); row=read_verified_branch(attempt,query_id=row['query_id'],branch_id=name)
                env.close_env(clear_cache=True); env=None
            records[name]=row
            if null_seal is not None and name == 'anchor-a':
                assert all(row[k] == original_anchor[k] for k in ['parent_hashes','endpoint_hashes','observed_effect',
                    'endpoint_status','executed_steps','BT_parent_progress','BT_endpoint_progress']), 'Original Full control differs;stop before null truth'
            if name.startswith('anchor-') and name!='anchor-a':
                assert row['endpoint_status']=='complete_horizon' and row['endpoint_hashes']==records['anchor-a']['endpoint_hashes'] and row['observed_effect']==records['anchor-a']['observed_effect'], 'Repeated Full endpoint/effect differs'
            if name=='anchor-qual-final':
                proof=[]
                for key,role in zip(order[:4],('refined','refined','raw','refined')):
                    item=records[key]; proof.append(dict(role=role,valid=item['endpoint_status']=='complete_horizon',commands=item['executed_steps'],parent_hashes=item['parent_hashes'],endpoint_hashes=item['endpoint_hashes']))
                require_repeated_endpoints(proof)
                atomic_json(out/'AABA_qualification.json',dict(status=f'qualified_original_BT_cohort_case0_fresh_q{query}_AABA' if is_BT else f'qualified_original_BRT_fresh_q{query}_AABA',selected_rank=selected,branches=order[:4],independent_policy_samples=0,candidate_truth_only=True,no_physical_unpack=True,utc=stamp()))
            if name=='anchor-qual-final' and is_BT:
                atomic_json(out/'selected_raw_refined_BT_contrast.json',paired_BT_repair_summary(records,selected))
            if name=='anchor-qual-final' and cli.qualify_only:
                assert identity.source_digest(C)==plan['original_source_sha256'] and identity.source_digest(H)==plan['reference_controller_sha256']
                operator.update(status='qualified_original_query_AABA_pool_pending',complete_branches=len(records),completed_branches=list(records),independent_candidate_endpoints=1,ended_utc=stamp(),full_pool_pending=True)
                save(); print(json.dumps(operator),flush=True); return
            operator.update(status='collecting_fixed_pool',complete_branches=len(records),independent_candidate_endpoints=sum(k.startswith('rank-') and r['endpoint_status']=='complete_horizon' for k,r in records.items()),completed_branches=list(records)); save()
            print(json.dumps(dict(branch=name,status=row['endpoint_status'],steps=row['executed_steps'],completed=len(records))),flush=True)
        if null_seal is not None:
            full_row,null_row,final_row = (records[n] for n in order)
            summary = dict(scope='One prespecified original specialist query;dependent branches,not population SR',
                query=query,source_states=1,branch_order=order,branch_endpoints={n:r['endpoint_status'] for n,r in records.items()},
                strict_labels={n:r['independent_label']['applicable'] for n,r in records.items()},
                object_fell_below_table={n:r['BT_endpoint_progress']['object_fell_below_table'] for n,r in records.items()},
                original_Full_parent_endpoint_effect_exact=True,final_Full_endpoint_effect_exact=True,
                Full_null_endpoint_physics_different=full_row['endpoint_hashes']['physics']!=null_row['endpoint_hashes']['physics'],
                natural_withdrawal_established=False,population_efficacy_estimated=False,policy_generations=0,
                null_generation_seal_sha256=cli.expected_null_generation_seal_sha256)
            atomic_json(out/'same_parent_withdrawal_summary.json',summary)
            assert identity.source_digest(C)==plan['original_source_sha256'] and identity.source_digest(H)==plan['reference_controller_sha256']
            operator.update(status='complete_original_same_parent_withdrawal',ended_utc=stamp(),complete_branches=len(records),
                independent_source_states=1,new_candidate_endpoints=0,physical_commands=sum(r['executed_steps'] for r in records.values()),
                original_source_sha256_after=identity.source_digest(C),reference_controller_sha256_after=identity.source_digest(H))
            save();print(json.dumps(dict(operator=operator,summary=summary)),flush=True);return
        assembled=dict(task=task,cohort=f'original_BT12000_cohort_case0_q{query}' if is_BT else f'original_fixed14k_seed0_q{query}',descriptor='uniform_mean_DINO_delta',required_effect_available=False,candidates=[])
        for rank in np.flatnonzero(valid):
            row=records[f'rank-{rank}']; assembled['candidates'].append(dict(candidate_id=f'rank-{rank}',valid=True,horizon=32,predicted_effect=predicted[rank].tolist(),historical_effect=historical[rank].tolist(),observed_effect=row['observed_effect'],endpoint_status=row['endpoint_status'],executed_steps=row['executed_steps'],applicable=row['independent_label']['applicable'] if is_BT else None))
        metrics=centered_effect_metrics(assembled)
        if 'constant_query' in metrics['references']:
            metrics['references']['constant_query']['absolute_mse']=None
            metrics['references']['constant_query']['absolute_mse_unavailable_reason']='Actual query requirement not captured;only the centered constant reference is identified.'
        atomic_json(out/'candidate_effects.json',assembled); atomic_json(out/'centered_effects.json',metrics)
        if is_BT:assert json.loads((out/'selected_raw_refined_BT_contrast.json').read_text())==paired_BT_repair_summary(records,selected)
        operator.update(status='complete_original_fixed_query_pool',ended_utc=stamp(),complete_candidate_endpoints=metrics['complete_endpoints'],unknown_candidate_endpoints=metrics['valid_candidates']-metrics['complete_endpoints'],original_source_sha256_after=identity.source_digest(C),reference_controller_sha256_after=identity.source_digest(H))
        assert operator['original_source_sha256_after']==plan['original_source_sha256'] and operator['reference_controller_sha256_after']==plan['reference_controller_sha256']
        save(); print(json.dumps(dict(operator=operator,metrics=metrics)),flush=True)
    except BaseException as exc:
        operator.update(status='failed_or_interrupted',error=repr(exc),traceback=traceback.format_exc(),ended_utc=stamp()); save(); raise
    finally:
        if env is not None:
            try: env.close_env(clear_cache=True)
            except Exception: pass
        for child in mp.active_children():
            if child.pid not in children_before: child.terminate(); child.join(timeout=5)
        fcntl.flock(lock,fcntl.LOCK_UN); lock.close()


if __name__=='__main__': main()
