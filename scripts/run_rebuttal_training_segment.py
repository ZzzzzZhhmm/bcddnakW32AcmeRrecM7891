#!/usr/bin/env python3
"""Continue an attested immutable Full lineage and preserve its entire state.

This operator never selects a TEST checkpoint, changes a scientific recipe,
contacts a Git remote, deletes a prior artifact, or retries an unknown run.
ACP owns the resource; the existing logged runner owns the training timeout.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nonreal_job import file_hash, source_identity


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def segment_config(before, *, output, resume, start, end):
    """Only storage/restart controls may differ from the original recipe."""
    if not 0 < start < end <= 3000:
        raise ValueError('segment must advance toward the fixed update3000')
    required = dict(max_steps=30000, batch_size=8,
                    gradient_accumulation_steps=16, seed=3407)
    if any(before.get(key) != value for key, value in required.items()):
        raise ValueError('original seed/batch/scheduler budget differs')
    # One complete checkpoint at the segment end avoids unbounded AFS growth.
    # Evaluation cadence and every mathematical training setting stay intact.
    after = dict(before, output_dir=str(output), resume=str(resume),
                 run_steps=end-start, save_every=end)
    allowed = {'output_dir', 'resume', 'run_steps', 'save_every'}
    if any(before[key] != after[key] for key in before.keys() - allowed):
        raise ValueError('scientific recipe was changed')
    return after


def validate_metrics(rows, *, start, end):
    if [row['step'] for row in rows] != list(range(start+1, end+1)):
        raise ValueError('missing, duplicated or out-of-order optimizer update')
    if not all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in rows):
        raise ValueError('nonfinite training metric')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-plan-sha256', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if file_hash(args.plan) != args.expected_plan_sha256:
        raise ValueError('immutable segment plan differs')
    plan = read(args.plan)
    if plan['schema'] != 'warm.rebuttal.full-training-segment.v1':
        raise ValueError('unsupported plan')
    if plan['selected_final_update'] != 3000 or plan['role'] != 'full':
        raise ValueError('this operator only continues the original Full reference')
    code = Path(plan['training_code'])
    if source_identity(code)['sha256'] != plan['training_source_sha256']:
        raise ValueError('frozen training implementation differs')
    for item in plan['inputs']:
        path = Path(item['path'])
        if not path.is_file() or file_hash(path) != item['sha256']:
            raise ValueError('input mismatch: ' + str(path))
    parent = Path(plan['parent_state'])
    inventory = plan['parent_state_files']
    actual_names = sorted(str(f.relative_to(parent)) for f in parent.rglob('*') if f.is_file())
    if actual_names != sorted(item['path'] for item in inventory):
        raise ValueError('parent optimizer/scheduler/RNG inventory differs')
    for item in inventory:
        f = parent / item['path']
        if f.stat().st_size != item['bytes'] or file_hash(f) != item['sha256']:
            raise ValueError('parent state file differs: ' + str(f))
    parent_marker = read(plan['parent_complete'])
    if parent_marker['step'] != plan['start_update'] or parent_marker['state_path'] != str(parent):
        raise ValueError('parent completion marker differs')
    attestation = read(plan['parent_attestation'])
    if (attestation['actual_global_step'] != plan['start_update'] or
            attestation['checkpoint_sha256'] != plan['parent_checkpoint_sha256']):
        raise ValueError('parent checkpoint attestation differs')
    import yaml
    before = yaml.safe_load(Path(plan['base_config']).read_text())
    output = Path(plan['output'])
    after = segment_config(before, output=output, resume=parent,
                           start=plan['start_update'], end=plan['end_update'])
    artifact = Path(plan['artifact_root'])
    if output.exists() or artifact.exists():
        raise FileExistsError('existing output or operator receipt: inspect it before any recovery')
    if shutil.disk_usage(output.parent.parent).free < plan['minimum_free_bytes']:
        raise OSError('insufficient AFS capacity; preserve prior states')
    if args.preflight_only:
        print(json.dumps(dict(status='attested_original_parent_and_recipe_admitted',
                              physical_training_updates=0, config_overrides=['output_dir', 'resume', 'run_steps', 'save_every'],
                              start_update=plan['start_update'], end_update=plan['end_update'])), flush=True)
        return

    artifact.mkdir(parents=True, exist_ok=False)
    state = dict(status='admitted', start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 selected_final_update=3000, formal_performance_eligible=False,
                 plan_sha256=args.expected_plan_sha256, operator_source_sha256=source_identity(ROOT)['sha256'],
                 training_source_sha256=plan['training_source_sha256'],
                 parent_checkpoint_sha256=plan['parent_checkpoint_sha256'])
    state_path = artifact / 'operator.json'
    write(state_path, state)
    try:
        config_path = artifact / 'full-segment-s3407.yaml'
        config_path.write_text(yaml.safe_dump(after, sort_keys=False))
        spec = read(plan['template_spec'])
        argv = spec['argv']
        argv[argv.index('--config-path')+1] = str(artifact)
        argv[argv.index('--config-name')+1] = config_path.stem
        argv = [('hydra.run.dir=' + str(artifact / 'hydra')) if s.startswith('hydra.run.dir=') else s for s in argv]
        spec.update(argv=argv, max_seconds=plan['max_training_seconds'],
                    evidence_type='original_full_fullstate_resume_segment',
                    claim='Continue original seed3407 and unchanged mathematical recipe toward fixed3000; no TEST performance claim.',
                    inputs=[dict(path=str(p), sha256=file_hash(p)) for p in (config_path, args.plan)], qualification=[])
        spec_path = artifact / 'training.job.json'
        spec_path.write_text(json.dumps(spec, indent=2) + '\n')
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                   HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                   DIFFSYNTH_SKIP_DOWNLOAD='true', TOKENIZERS_PARALLELISM='false',
                   DIFFSYNTH_MODEL_BASE_PATH=plan['model_base_path'], WARM_IMMUTABLE_ASSET_HASH_CACHE='1',
                   PYTHONPATH=str(code/'src') + os.pathsep + str(code))
        state.update(status='running_training', config_sha256=file_hash(config_path), spec_sha256=file_hash(spec_path))
        write(state_path, state)
        with (artifact/'launcher.log').open('xb') as log:
            result = subprocess.run([plan['train_python'], str(code/'scripts/nonreal_job.py'),
                                     '--spec', str(spec_path), '--output-root', str(artifact/'job_logs')],
                                    cwd=code, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('training runner exit ' + str(result.returncode) + '; retain partial state')
        end = plan['end_update']
        complete = read(output/'training_complete.json')
        weight = output / f'checkpoints/weights/step_{end:06d}.pt'
        attest = read(weight.with_suffix('.training.json'))
        if (complete['step'] != end or attest['actual_global_step'] != end or
                attest['resume_step'] != plan['start_update'] or
                attest['parent_checkpoint_sha256'] != plan['parent_checkpoint_sha256'] or
                file_hash(weight) != attest['checkpoint_sha256']):
            raise ValueError('completed state/weight/lineage failed attestation')
        metrics = [json.loads(row) for row in (output/'training_metrics.jsonl').read_text().splitlines() if row.strip()]
        validate_metrics(metrics, start=plan['start_update'], end=end)
        if source_identity(code)['sha256'] != plan['training_source_sha256']:
            raise ValueError('training source changed during execution')
        state.update(status='archiving_complete_training', checkpoint_sha256=attest['checkpoint_sha256'], optimizer_updates=end)
        write(state_path, state)
        scratch = Path(plan['scratch_root'])
        scratch.mkdir(parents=True, exist_ok=False)
        archive = scratch / f'full_resume_update{end}_evidence.tar'
        with tarfile.open(archive, 'w') as tar:
            tar.add(output, arcname='training')
            tar.add(config_path, arcname='config.yaml')
            tar.add(spec_path, arcname='job_spec.json')
        loader = importlib.util.spec_from_file_location('warm_pinned_lossless_packer', plan['packer'])
        packer = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(packer)
        manifest = packer.pack(archive, Path(plan['cas_store']), file_hash(archive))
        evidence = packer.verify(manifest)
        evidence.update(role='full', optimizer_updates=end, start_update=plan['start_update'],
                        parent_checkpoint_sha256=plan['parent_checkpoint_sha256'], checkpoint_sha256=attest['checkpoint_sha256'],
                        source_sha256=plan['training_source_sha256'], selected_final_update=3000, formal_performance_eligible=False)
        write(artifact/'archive_receipt.json', evidence)
        state.update(status='complete_training_and_byte_exact_archive', archive_receipt=str(artifact/'archive_receipt.json'),
                     end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(state_path, state)
        import fcntl
        ledger = Path(plan['experiment_ledger'])
        with ledger.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write('\n\n### Automatic collection: original Full matched seed3407 update' + str(end) + '\n\n' +
                         'UTC ' + state['end_utc'] + '; full optimizer/scheduler/RNG/dataloader resume ' + str(plan['start_update']) +
                         ' to ' + str(end) + ', exactly ' + str(len(metrics)) + ' finite optimizer updates. Training source SHA `' +
                         plan['training_source_sha256'] + '`, checkpoint SHA `' + evidence['checkpoint_sha256'] +
                         '`. Complete byte-exact archive receipt `' + str(artifact/'archive_receipt.json') + '`, TAR SHA `' +
                         evidence['raw_sha256'] + '`, ' + str(evidence['raw_bytes']) +
                         ' bytes. Only checkpoint storage cadence changed to one end-of-segment save; the original mathematical recipe remains. ' +
                         '**Training execution and preservation only. Fixed TEST selection remains3000; no matched efficacy result.**\n')
            stream.flush()
            os.fsync(stream.fileno())
            fcntl.flock(stream, fcntl.LOCK_UN)
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        state.update(status='failed_or_interrupted', error=repr(exc), traceback=traceback.format_exc(),
                     end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(state_path, state)
        raise


if __name__ == '__main__':
    main()
