"""Collect original-specialist q8 effects after exact fresh-scene A-A-B-A.

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--resume-known-terminal', action='store_true')
    cli = parser.parse_args()
    import fcntl
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    read = lambda p: json.loads(Path(p).read_text())
    stamp = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    assert sha(cli.plan) == cli.expected_plan_sha256
    plan = read(cli.plan)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='true',DIFFSYNTH_MODEL_BASE_PATH=plan['model_root'],TOKENIZERS_PARALLELISM='false')
    assert plan['schema'] == 'warm.original-specialist-fresh-pool.v1'
    assert plan['query_replan'] == 8 and plan['selected_rank'] == 28
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
        descriptor='original_uniform_mean_DINO_delta_v1', query=8, selected_rank=28,
        complete_branches=0, independent_candidate_endpoints=0, applicability_labels=0,
        source_checkpoint_sha256=plan['checkpoint_sha256'], new_efficacy_claimed=False)
    def save():
        operator['updated_utc'] = stamp(); atomic_json(out/'operator.json', operator)
    save(); env = None; children_before = {p.pid for p in mp.active_children()}
    try:
        for row in plan['inputs']:
            assert sha(row['path']) == row['sha256'], row['path']
        full = read(source/'operator.json')
        assert full['status'] == 'complete_fixed_original_factual_episode' and full['native_commands'] == 600
        args = read(source/'runtime_args.json'); contract = read(args['warm_online_contract_path'])
        assert args['task_name'] == 'blocks_ranking_try'
        trace = [json.loads(s) for s in (source/'native_trace.jsonl').read_text().splitlines()]
        assert len(trace) == 600
        commands = np.asarray([r['command'] for r in trace], dtype=np.float32)
        before_trace = np.asarray([r['before'] for r in trace], dtype=np.float32)
        after_trace = np.asarray([r['after'] for r in trace], dtype=np.float32)
        probe = source/'probes/q0008'
        event = torch.load(probe/'event_adapter.pt', map_location='cpu', weights_only=False)
        model = torch.load(probe/'model_output.pt', map_location='cpu', weights_only=False)
        valid = event['candidate_valid_mask'].numpy()[0]
        assert valid.shape == (32,) and valid.dtype == np.bool_ and valid[28]
        assert model['warm_retrospection']['selected_candidate_index'] == 28
        adapted = event['adapted_action_mean'].float().numpy()[0]
        predicted = event['predicted_effect'].float().numpy()[0]
        assert adapted.shape == (32,32,14) and predicted.shape == (32,768)
        normalizer = RobotwinQposZScore.from_dataset_stats(read(args['dataset_stats_path']))
        refined = decode_branch_actions(normalizer, np.load(source/'generations/q0008/model_actions.npy', allow_pickle=False))
        assert np.array_equal(refined[:4], commands[32:36])
        native = {f'rank-{r}': decode_branch_actions(normalizer, adapted[r]) for r in np.flatnonzero(valid)}
        order = ['anchor-a','anchor-aa','rank-28','anchor-qual-final']
        order += [f'rank-{r}' for r in np.flatnonzero(valid) if r != 28]
        order += ['anchor-pool-final']
        actions = {name: refined if name.startswith('anchor-') else native[name] for name in order}
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
        seal = dict(query=8, frame=32, source_plan_sha256=plan['factual_source_plan_sha256'],
            checkpoint_sha256=plan['checkpoint_sha256'], descriptor='original_uniform_mean_DINO_delta_v1',
            before_any_new_branch_truth=True, selected_rank=28, original_valid_ranks=np.flatnonzero(valid).tolist(),
            branch_order=order, source_inputs=plan['inputs'], expected_source_observation=str(source/'replan-0008.npz'),
            action_hashes={k:sha256_array(v) for k,v in actions.items()},
            predicted_sha256=sha256_array(predicted), historical_sha256=sha256_array(historical),
            required_effect_available=False, unsupported_applicability_is_unknown=True,
            rng_protocol='Restore the sealed source pre-forward Python/NumPy/Torch RNG at each query parent;no physical unpack.')
        if (out/'sealed_query.json').exists():
            assert read(out/'sealed_query.json') == seal
            with np.load(out/'sealed_arrays.npz', allow_pickle=False) as saved:
                assert np.array_equal(saved['predicted'],predicted) and np.array_equal(saved['historical'],historical)
                assert all(np.array_equal(saved[k],v) for k,v in actions.items())
        else:
            np.savez_compressed(out/'sealed_arrays.npz', predicted=predicted, historical=historical, valid=valid, **actions)
            atomic_json(out/'sealed_query.json', seal)
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
        cls = getattr(importlib.import_module('envs.blocks_ranking_try'), 'blocks_ranking_try')
        cfg = setup.resolved_config(runtime,'blocks_ranking_try',0)
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
                row = read_verified_branch(completed[0].parent,query_id='original-brt-seed0-q8',branch_id=name)
                assert row['sealed_query_sha256'] == sha(out/'sealed_query.json')
            else:
                attempt = root/f'attempt-{len(list(root.glob("attempt-*")))+1:03d}'; attempt.mkdir()
                atomic_json(attempt/'attempt.json', dict(branch=name,pid=os.getpid(),utc=stamp()))
                operator.update(status='recreating_fixed_parent',active_branch=name); save()
                random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
                env = cls(); env.setup_demo(**cfg)
                assert type(env.scene.physx_system).__name__ == 'PhysxCpuSystem' and not env.eval_video_path and not env.save_data and not env.render_freq
                instruction = str(np.random.choice(generate_episode_descriptions('blocks_ranking_try',[expert['info']],100)[0]['unseen']))
                assert instruction == initial['instruction']; env.set_instruction(instruction=instruction)
                require_source(source/'initial_observation.npz',env.get_obs())
                with OfficialQposGuard(env) as guard:
                    for index,command in enumerate(commands[:32]):
                        budget(); before = factual_joint_state(env.get_obs()).copy()
                        assert guard.step(command)
                        require_step(index,before,factual_joint_state(env.get_obs()))
                obs = env.get_obs(); require_source(source/'replan-0008.npz',obs)
                random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch_cpu']); torch.cuda.set_rng_state_all(rng['torch_cuda'])
                parent = native_parent(env,obs); parent_hash = state_hashes(parent)
                if records: assert parent_hash == records['anchor-a']['parent_hashes'], 'Fresh native parents differ'
                before_feature = reader.encode(obs)
                assert state_hashes(native_parent(env,env.get_obs())) == parent_hash, 'Teacher read changed parent'
                if records: assert np.array_equal(before_feature,np.load(out/'parent_dino_spatial.npy',allow_pickle=False))
                else: np.save(out/'parent_dino_spatial.npy',before_feature,allow_pickle=False)
                (attempt/'parent.pickle').write_bytes(pickle.dumps(parent,protocol=4)); np.save(attempt/'parent_dino_spatial.npy',before_feature,allow_pickle=False)
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
                            require_step(32+index,native_api_before,factual_joint_state(env.get_obs()))
                        if name.startswith('anchor-') and index==3: require_source(source/'replan-0009.npz',env.get_obs())
                        if env.eval_success and count<32: status='early_success'; break
                endpoint=native_parent(env,env.get_obs()); endpoint_hash=state_hashes(endpoint)
                after_feature=reader.encode(env.get_obs())
                assert state_hashes(native_parent(env,env.get_obs())) == endpoint_hash, 'Teacher read changed endpoint'
                effect=_mean_effect(torch.from_numpy(after_feature-before_feature)[None])[0].numpy() if status=='complete_horizon' else None
                (attempt/'endpoint.pickle').write_bytes(pickle.dumps(endpoint,protocol=4)); np.save(attempt/'endpoint_dino_spatial.npy',after_feature,allow_pickle=False)
                row=dict(query_id='original-brt-seed0-q8',branch_id=name,parent_recreated=True,parent_hashes=parent_hash,endpoint_hashes=endpoint_hash,
                    executed_steps=count,endpoint_status=status,error=error,observed_effect=effect.tolist() if effect is not None else None,
                    independent_label=dict(applicable=None,label_provenance='BRT rubric not yet qualified;unknown preserved'),
                    endpoint_native_success=bool(env.eval_success),sealed_query_sha256=sha(out/'sealed_query.json'))
                seal_branch(attempt,row); row=read_verified_branch(attempt,query_id=row['query_id'],branch_id=name)
                env.close_env(clear_cache=True); env=None
            records[name]=row
            if name.startswith('anchor-') and name!='anchor-a':
                assert row['endpoint_status']=='complete_horizon' and row['endpoint_hashes']==records['anchor-a']['endpoint_hashes'] and row['observed_effect']==records['anchor-a']['observed_effect'], 'Repeated Full endpoint/effect differs'
            if name=='anchor-qual-final':
                proof=[]
                for key,role in zip(order[:4],('refined','refined','raw','refined')):
                    item=records[key]; proof.append(dict(role=role,valid=item['endpoint_status']=='complete_horizon',commands=item['executed_steps'],parent_hashes=item['parent_hashes'],endpoint_hashes=item['endpoint_hashes']))
                require_repeated_endpoints(proof)
                atomic_json(out/'AABA_qualification.json',dict(status='qualified_original_BRT_fresh_q8_AABA',selected_rank=28,branches=order[:4],independent_policy_samples=0,candidate_truth_only=True,no_physical_unpack=True,utc=stamp()))
            operator.update(status='collecting_fixed_pool',complete_branches=len(records),independent_candidate_endpoints=sum(k.startswith('rank-') and r['endpoint_status']=='complete_horizon' for k,r in records.items()),completed_branches=list(records)); save()
            print(json.dumps(dict(branch=name,status=row['endpoint_status'],steps=row['executed_steps'],completed=len(records))),flush=True)
        assembled=dict(task='blocks_ranking_try',cohort='original_fixed14k_seed0_q8',descriptor='uniform_mean_DINO_delta',required_effect_available=False,candidates=[])
        for rank in np.flatnonzero(valid):
            row=records[f'rank-{rank}']; assembled['candidates'].append(dict(candidate_id=f'rank-{rank}',valid=True,horizon=32,predicted_effect=predicted[rank].tolist(),historical_effect=historical[rank].tolist(),observed_effect=row['observed_effect'],endpoint_status=row['endpoint_status'],executed_steps=row['executed_steps'],applicable=None))
        metrics=centered_effect_metrics(assembled)
        if 'constant_query' in metrics['references']:
            metrics['references']['constant_query']['absolute_mse']=None
            metrics['references']['constant_query']['absolute_mse_unavailable_reason']='Actual query requirement not captured;only the centered constant reference is identified.'
        atomic_json(out/'candidate_effects.json',assembled); atomic_json(out/'centered_effects.json',metrics)
        operator.update(status='complete_original_fixed_q8_pool',ended_utc=stamp(),complete_candidate_endpoints=metrics['complete_endpoints'],unknown_candidate_endpoints=metrics['valid_candidates']-metrics['complete_endpoints'],original_source_sha256_after=identity.source_digest(C),reference_controller_sha256_after=identity.source_digest(H))
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
