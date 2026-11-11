#!/usr/bin/env python3
"""Reproduce one sealed native Full H32, then seal a same-context null H32.

No simulator, retrieval search, consumed online capability, Git access, or new
physical truth is used. A changed Full generation is preserved as an unknown
engineering outcome and blocks the null branch; it never changes the query.
"""
from pathlib import Path
import argparse
from copy import deepcopy
from dataclasses import fields
import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
read = lambda path: json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(value)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seal', required=True, type=Path)
    ap.add_argument('--expected-seal-sha256', required=True)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--admission', type=Path)
    ap.add_argument('--expected-admission-sha256')
    ap.add_argument('--preflight-only', action='store_true')
    cli = ap.parse_args()
    seal = read(cli.seal)
    sealed = cli.seal.resolve(strict=True).parent
    out = cli.output.resolve()
    if sha(cli.seal) != cli.expected_seal_sha256:
        raise ValueError('original native query seal changed')
    if (seal['schema'], seal['task'], seal['case_id'], seal['seed'], seal['condition'], seal['query'], seal['frame']) != (
            'warm.rebuttal.native-DEV-query-seal.v1', 'rearrange_blocks', 0, 3407, 'clean_size_matched', 8, 32):
        raise ValueError('only the original prespecified clean q8 source is supported')
    manifest = read(sealed / 'manifest.json')
    if {p.name for p in sealed.iterdir() if p.is_file()} != set(manifest) | {'manifest.json'}:
        raise ValueError('original export acquired extra or missing files')
    for name, row in manifest.items():
        if Path(name).name != name or sha(sealed / name) != row['sha256']:
            raise ValueError('sealed original bytes changed: ' + name)
    for name, row in seal['inputs'].items():
        if sha(name) != row['sha256']:
            raise ValueError('original factual evidence changed: ' + name)
    plan = read(next(n for n in seal['inputs'] if Path(n).name == 'plan.json'))
    if out.exists() or out.is_relative_to(ROOT) or out.is_relative_to(sealed):
        raise ValueError('fresh Git-external generation output required')
    admission = None
    if cli.preflight_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        if cli.admission is None or cli.expected_admission_sha256 is None or sha(cli.admission) != cli.expected_admission_sha256:
            raise ValueError('explicit immutable null-study admission required')
        admission = read(cli.admission)
        start = datetime.datetime.fromisoformat(admission['first_admitted_utc'])
        end = datetime.datetime.fromisoformat(admission['absolute_deadline_utc'])
        if (admission['schema'] != 'warm.native-DEV-q8-null-admission.v1'
                or not 0 < (end - start).total_seconds() <= 3600
                or not start <= datetime.datetime.now(datetime.timezone.utc) < end
                or admission['sealed_query_sha256'] != cli.expected_seal_sha256
                or Path(admission['generation_output']).resolve() != out):
            raise ValueError('null study query, output or original resource window differs')
        marker = read(admission['external_worker_reservation_path'])
        if (marker['status'] != 'reserved_native_null_followup_before_first_process'
                or marker['generation_output'] != str(out)
                or marker['sealed_query_sha256'] != cli.expected_seal_sha256):
            raise ValueError('unique cross-client null-study reservation required')
        ref = admission['previous_stage_terminal_archive_receipt']
        proof = read(ref['path'])
        if (sha(ref['path']) != ref['sha256'] or not proof['actual_processes_absent']
                or not proof['all_existing_raw_archived'] or proof['actual_exit']['exit_code'] != 0):
            raise ValueError('complete known q8 pool terminal and raw archive required')
        if sha(proof['archive']) != proof['archive_sha256']:
            raise ValueError('previous complete pool archive changed')
    H = load('native_helper', Path(seal['native_helper_code']) / 'scripts/run_rebuttal_matched_native.py')
    core, identity = H.install_core(plan)
    if out.is_relative_to(core) or out.is_relative_to(Path(seal['native_helper_code'])):
        raise ValueError('generation output overlaps immutable scientific source')
    if admission and identity(ROOT)['sha256'] != admission['controller_source_sha256']:
        raise ValueError('null generation controller source changed')
    import numpy as np
    import torch
    from fastwam.models.warm.retrospection_model import RetrospectiveSourceContext
    helper = load('native_null_rules', ROOT / 'src/fastwam/research/native_null_replay.py')
    replica_rules = load('native_null_replica', ROOT / 'src/fastwam/research/inference_replica.py')
    source_result = read(next(n for n in seal['inputs'] if Path(n).name == 'result.json'))
    query = source_result['queries'][8]
    original_probe = torch.load(sealed / 'source_probe.pt', map_location='cpu', weights_only=False)['probe']
    payload = torch.load(sealed / 'source_context.pt', map_location='cpu', weights_only=True)
    helper.restore_factual_context(payload, RetrospectiveSourceContext, device='cpu')
    original = np.load(sealed / 'source_generation.npy', allow_pickle=False).reshape(32, 14)
    if original.dtype != np.float32 or not np.isfinite(original).all():
        raise ValueError('original normalized H32 is invalid')
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status='authenticated_factual_context_ready', pid=os.getpid(), sealed_query_sha256=cli.expected_seal_sha256,
                 simulator_loaded=False, counterfactual_branches_executed=0, physical_unpack=False, formal_TEST=False)
    def save():
        state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        H.M.atomic_json(out / 'operator.json', state)
    save()
    if cli.preflight_only:
        state.update(status='CPU_factual_context_verified_no_policy_or_null_generation', source_sha256=identity(ROOT)['sha256'])
        save()
        if torch.cuda.is_initialized():
            raise RuntimeError('preflight unexpectedly initialized CUDA')
        print(json.dumps(state))
        return
    (out / 'admission_snapshot.json').write_bytes(cli.admission.read_bytes())
    interrupted = False
    def stop(*_):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def budget(*, before_load=False):
        if interrupted or datetime.datetime.now(datetime.timezone.utc) >= end:
            raise TimeoutError('original null-study window ended;never reset it')
        if shutil.disk_usage(out).free < admission['minimum_free_bytes']:
            raise OSError('preserve concurrent raw evidence: storage guard')
        if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 32 * 1024**2:
            raise OSError('bounded generation raw budget exceeded')
        if torch.cuda.mem_get_info()[0] < (32 if before_load else 8) * 1024**3:
            raise OSError('protect concurrent policy work: actual free GPU memory guard')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', DIFFSYNTH_SKIP_DOWNLOAD='true', TOKENIZERS_PARALLELISM='false')
    policy = None
    try:
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.45)
        budget(before_load=True)
        from experiments.rmbench.warm_policy import deploy_policy as deployment
        from experiments.rmbench.warm_policy.deploy_policy import RMBenchWarmPolicy, _normalize_state, DEFAULT_PROMPT
        from fastwam.memory.manifest import sha256_array
        from fastwam.models.wan22.fastwam import FastWAM
        runtime = read(sealed / 'runtime_args.json')
        runtime['warm_telemetry_path'] = str(out / 'policy_telemetry.jsonl')
        H.M.atomic_json(out / 'runtime_args.json', runtime)
        asset_root = helper.asset_working_directory(runtime)
        H.M.atomic_json(out / 'asset_resolution.json', dict(previous_cwd=str(Path.cwd()), asset_working_directory=str(asset_root), immutable_model_code=str(core), bound_asset_paths_unchanged=True))
        os.chdir(asset_root)
        state.update(status='loading_one_typed_policy_and_cold_readonly_replica')
        save()
        with helper.bound_asset_metadata(deployment, runtime,
                lambda witness: H.M.atomic_json(out / 'loaded_asset_metadata.json', witness)):
            policy = RMBenchWarmPolicy(runtime)
        typed = H.M.typed_policy_receipt(policy, plan, 'full')
        H.M.atomic_json(out / 'typed_policy_receipt.json', typed)
        model = policy.model
        null_model, weights = replica_rules.shared_weight_inference_replica(model)
        model.configure_research_probe(capture=True, force_null=False)
        null_model.configure_research_probe(capture=True, force_null=True)
        with np.load(sealed / 'query_observation.npz', allow_pickle=False) as z:
            cameras = {k: z[k].copy() for k in z.files if k != 'native_qpos'}
            qpos = z['native_qpos'].copy()
        raw_batch, _ = policy.retriever._prepare_raw_cameras(cameras)
        image = np.ascontiguousarray(policy.retriever._image_adapter.prepare(raw_batch).vae_frames, dtype=np.float32)
        proprio = _normalize_state(qpos, policy.processor).numpy()
        if sha256_array(image) != query['model_input_sha256'] or sha256_array(proprio) != query['proprio_sha256']:
            raise ValueError('reconstructed current image or normalized proprio differs from factual generation')
        prompt = DEFAULT_PROMPT.format(task=source_result['initial_identity']['instruction'])
        ctx = helper.restore_factual_context(payload, RetrospectiveSourceContext, device=model.device)
        options = {k: runtime[k] for k in ('negative_prompt', 'text_cfg_scale', 'num_inference_steps', 'sigma_shift', 'rand_device', 'tiled')}
        if options['num_inference_steps'] != 10:
            raise ValueError('original NFE changed')
        context_records = lambda context: {f.name: H.M.tensor_record(getattr(context, f.name)) for f in fields(context) if torch.is_tensor(getattr(context, f.name))}
        ctx_before = context_records(ctx)
        H.M.atomic_json(out / 'reconstructed_inputs.json', dict(prompt=prompt, model_input_sha256=sha256_array(image), proprio_sha256=sha256_array(proprio), derived_seed=query['derived_seed'], sampler_options=options, factual_context=ctx_before, no_BoundOnlineStep_constructed=True))
        outputs = {}
        for role, selected_model in [('full-replay', model), ('candidate-null', null_model), ('full-replay-final', model)]:
            budget()
            state.update(status='generating_' + role)
            save()
            with torch.inference_mode():
                output = FastWAM.infer_action(selected_model, prompt=prompt,
                    input_image=torch.tensor(image, device=model.device, dtype=model.torch_dtype),
                    proprio=torch.tensor(proprio, device=model.device, dtype=model.torch_dtype), action_horizon=32,
                    action_source_context=ctx, seed=int(query['derived_seed']),
                    memory_sigma=model._require_retrospection().source_sigma_min, **options)
            action = output['action'].detach().float().cpu().numpy().reshape(32, 14).copy()
            probe = deepcopy(selected_model._last_research_probe)
            np.save(out / (role + '_model_actions.npy'), action, allow_pickle=False)
            torch.save(probe, out / (role + '_probe.pt'))
            if context_records(ctx) != ctx_before:
                raise RuntimeError('production generation mutated the sealed factual context')
            replica_rules.assert_shared_weights_unchanged(weights)
            if role.startswith('full-replay'):
                helper.require_reproduced_full(action, original, probe, original_probe, payload)
            else:
                helper.require_explicit_null_probe(probe, original_probe)
            outputs[role] = action
            del output
        if not np.array_equal(outputs['full-replay'], outputs['full-replay-final']):
            raise ValueError('interleaved null generation changed the Full H32')
        from fastwam.research.candidate_pool import decode_branch_actions
        native = {k: decode_branch_actions(policy.action_normalizer, value) for k, value in outputs.items()}
        with np.load(sealed / 'sealed_native_actions.npz', allow_pickle=False) as z:
            if not np.array_equal(native['full-replay'], z['anchor-a']):
                raise ValueError('original native Full commands differ after exact model replay')
        np.savez_compressed(out / 'sealed_native_actions.npz', **{'anchor-a': native['full-replay'], 'candidate-null': native['candidate-null'], 'anchor-final': native['full-replay-final']})
        seal_null = dict(schema='warm.rebuttal.sealed-native-same-parent-null.v1', original_seal_sha256=cli.expected_seal_sha256,
            original_pool_terminal_receipt=admission['previous_stage_terminal_archive_receipt'],
            checkpoint_sha256=policy.checkpoint_sha256, controller_source_sha256=admission['controller_source_sha256'],
            task=seal['task'], case_id=0, seed=3407, query=8, frame=32, horizon=32,
            expected_branch_order=['anchor-a', 'candidate-null', 'anchor-final'],
            model_input_sha256=query['model_input_sha256'], proprio_sha256=query['proprio_sha256'], derived_seed=query['derived_seed'],
            original_factual_context_sha256=sha(sealed / 'source_context.pt'), full_replay_exact=True,
            full_source_Gaussian_conditioning_and_prediction_exact=True, cold_null_replica_shared_readonly_weights=True,
            same_factual_context=True, Gaussian_exact=True, explicit_candidate_null=True,
            before_any_generated_null_execution=True, counterfactual_branches_executed=0,
            no_historical_BoundOnlineStep_capability_constructed_or_consumed=True, natural_skill_absence_certified=False,
            formal_TEST=False, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        seal_null['files'] = {name: dict(bytes=(out / name).stat().st_size, sha256=sha(out / name)) for name in [
            'sealed_native_actions.npz', 'full-replay_model_actions.npy', 'candidate-null_model_actions.npy', 'full-replay-final_model_actions.npy',
            'full-replay_probe.pt', 'candidate-null_probe.pt', 'full-replay-final_probe.pt', 'reconstructed_inputs.json', 'typed_policy_receipt.json']}
        H.M.atomic_json(out / 'sealed_null.json', seal_null)
        state.update(status='complete_exact_Full_then_same_context_null_sealed_before_null_truth', null_seal_sha256=sha(out / 'sealed_null.json'), full_replay_exact=True, explicit_null=True)
        save()
    except BaseException as exc:
        state.update(status='unknown_full_reproduction_or_generation_guard_failure', error=repr(exc), traceback=traceback.format_exc())
        save()
        raise
    finally:
        if policy is not None:
            policy.close()
        if identity(core)['sha256'] != seal['evaluation_source_sha256']:
            raise RuntimeError('immutable scientific source changed')
        state['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
    print(json.dumps(state))


if __name__ == '__main__':
    main()
