#!/usr/bin/env python3
"""One bounded submission for qualified continuation and new-checkpoint probes.

This does not claim to implement the missing simulator/history/trained controls.
Their blockers remain explicit in every report. No network or Git operations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]

from nonreal_job import file_hash, run as run_logged, source_identity


BLOCKERS = {
    'tables_4_17_18': 'Real RMBench full-state restore and independent outcome labels are not qualified; a renderer pass alone is insufficient.',
    'table_23': 'Requirement-only semantic history intervention is not implemented/qualified.',
    'table_20': 'Matched retrieval-residual training and closed-loop evaluation are missing.',
    'table_22': 'Independent matched Gaussian/scale-only training and closed-loop evaluation are missing; source probes are inference interventions.',
    'tables_11_12_14': 'Original run/checkpoint linkage, donor telemetry and paired outcomes remain missing.',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def tree_files(root):
    return {p.relative_to(root).as_posix(): file_hash(p)
            for p in sorted(root.rglob('*')) if p.is_file()}


def validate_plan(path):
    plan = read(path)
    if plan.get('schema') != 'warm.nonreal.bundle.v1':
        raise ValueError('unsupported bundle plan')
    if not 1 <= plan['max_seconds'] <= 23 * 3600:
        raise ValueError('bundle budget must be within 23 hours')
    if ROOT != Path(plan['code']).resolve():
        raise ValueError('wrong immutable bundle checkout')
    output = Path(plan['output']).resolve()
    if output.is_relative_to(ROOT) or not output.is_absolute():
        raise ValueError('output must be outside immutable source')
    if source_identity(ROOT)['sha256'] != plan['source_sha256']:
        raise ValueError('bundle source changed')
    if source_identity(Path(plan['resume_code']))['sha256'] != plan['resume_source_sha256']:
        raise ValueError('prepared training source changed')
    for item in plan['inputs']:
        if file_hash(Path(item['path'])) != item['sha256']:
            raise ValueError('prepared input changed: ' + item['path'])
    resume = read(plan['resume_plan'])
    if resume['source_sha256'] != plan['resume_source_sha256']:
        raise ValueError('resume source identity differs')
    if resume['target_step'] != 15000 or resume['parent_step'] != 300:
        raise ValueError('this handoff binds the prepared 300-to-15000 continuation')
    if not Path(plan['asset_root']).is_dir():
        raise ValueError('missing offline assets')
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(resume['config'])
    if cfg.data.val.episode_split != 'dev' or cfg.data.val.is_training_set:
        raise ValueError('post-training probes require the declared DEV split')
    return plan, resume


def verified_target(plan, resume):
    """Do not accept exit=0, an old summary or an unverified target by itself."""
    parent = Path(plan['resume_plan']).parent
    summary = read(parent / 'summary.json')
    job = read(parent / 'job/run_manifest.json')
    if summary['status'] != 'complete' or summary['exit_code'] != 0 or summary.get('reason') is not None:
        raise ValueError('continuation summary is incomplete')
    if job['status'] != 'complete' or job['exit_code'] != 0:
        raise ValueError('continuation process did not complete')
    if job['source_sha256'] != resume['source_sha256'] or job['source_sha256_after'] != resume['source_sha256']:
        raise ValueError('continuation source identity mismatch')
    from nonreal_resume import collect_metrics
    from fastwam.models.warm.training_attestation import verify_training_attestation
    training = Path(resume['training_output'])
    metrics_path = training / 'training_metrics.jsonl'
    rows = collect_metrics(metrics_path)
    if not rows or rows[-1]['step'] != resume['target_step'] or file_hash(metrics_path) != summary['metrics_sha256']:
        raise ValueError('target training metrics are missing or changed')
    tag = f"step_{resume['target_step']:06d}"
    weights = training / 'checkpoints/weights' / (tag + '.pt')
    state = training / 'checkpoints/state' / tag
    if read(state / 'trainer_state.json')['global_step'] != resume['target_step']:
        raise ValueError('target state step mismatch')
    for rank in range(4):
        for file in (state / f'random_states_{rank}.pkl', state / 'pytorch_model' / f'bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'):
            if not file.is_file() or file.stat().st_size == 0:
                raise ValueError('missing final rank state: ' + str(file))
    if not (state / 'scheduler.bin').is_file():
        raise ValueError('missing target scheduler state')
    att = verify_training_attestation(weights, weights.with_suffix('.training.json'))
    from omegaconf import OmegaConf
    from fastwam.models.warm.training_attestation import training_config_hashes
    canonical, _ = training_config_hashes(OmegaConf.to_container(OmegaConf.load(resume['config']), resolve=True))
    if (att.checkpoint_step != resume['target_step'] or att.resume_step != resume['parent_step']
            or att.parent_checkpoint_sha256 != resume['parent_weights_sha256']
            or att.shared_recipe_sha256 != resume['shared_recipe_sha256']
            or att.resolved_train_config_sha256 != canonical
            or att.world_size != 4 or att.effective_batch_size != 128):
        raise ValueError('target checkpoint lineage/config/runtime mismatch')
    return dict(weights=str(weights), state=str(state), checkpoint_sha256=att.checkpoint_sha256,
                checkpoint_step=att.checkpoint_step, metrics_sha256=summary['metrics_sha256'],
                training_summary=str(parent / 'summary.json'), training_record=str(parent / 'EXPERIMENT_RECORD.generated.md'))


def full_gate_report(root, expected):
    """Verify every natural Full probe and its arrays before triggering controls."""
    import numpy as np
    manifest, summary = read(root / 'manifest.json'), read(root / 'summary.json')
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError('Full probe identity mismatch: ' + key)
    prefixes = read(root / 'frozen_prefixes.json')
    if file_hash(root / 'frozen_prefixes.json') != expected['prefix_manifest_sha256']:
        raise ValueError('frozen prefix bytes changed')
    rows = [json.loads(line) for line in (root / 'source_results.jsonl').read_text().splitlines()]
    if summary['status'] != 'complete' or len(rows) != 50 or len(prefixes) != 50:
        raise ValueError('expected all 50 predeclared DEV prefixes')
    alpha, gates, scales, eligible = [], [], [], []
    for number, (prefix, row) in enumerate(zip(prefixes, rows)):
        query = f'prefix-{number:04d}'
        if row['query_id'] != query or row['mode'] != 'full' or any(row[k] != v for k, v in prefix.items()):
            raise ValueError('Full prefix identity mismatch')
        directory = root / 'probes' / (query + '-full')
        for name, key in [('proposal.npz', 'array_sha256'), ('proposal.json', 'metadata_sha256')]:
            if file_hash(directory / name) != row['probe'][key]:
                raise ValueError('probe hash mismatch')
        meta = read(directory / 'proposal.json')
        if meta['force_null'] or meta['source_mode'] != 'full':
            raise ValueError('requires natural Full gate')
        with np.load(directory / 'proposal.npz', allow_pickle=False) as arrays:
            values = [arrays[k].reshape(-1) for k in ('alpha', 'g', 'v_det')]
            if any(len(v) != 1 or not np.isfinite(v).all() for v in values):
                raise ValueError('finite batch-one gate fields required')
            a, g, e = (float(v[0]) for v in values)
            c = float(np.mean(arrays['source_noise_scale'].astype(np.float64) ** 2))
            if not np.isfinite(c) or bool(g) != row['nonzero_gate'] or not np.isclose(g, row['mean_g']):
                raise ValueError('saved gate metrics differ from arrays')
        with np.load(root / (query + '-source-actions.npz'), allow_pickle=False) as actions:
            if set(actions.files) != {'full', 'repeat'} or any(actions[k].shape != (32, 14) or not np.isfinite(actions[k]).all() for k in actions.files):
                raise ValueError('invalid or missing Full action outputs')
        alpha.append(a); gates.append(g); scales.append(c); eligible.append(bool(e))
    nonzero = sum(g != 0 for g in gates)
    if summary['queries'] != 50 or summary['nonzero_gate_queries'] != nonzero:
        raise ValueError('Full summary differs from raw arrays')
    return dict(queries=50, episodes=len({(p['task'], p['episode_index']) for p in prefixes}),
                nonzero_g=nonzero, mean_g=float(np.mean(gates)), mean_noise_scale_squared=float(np.mean(scales)),
                alpha_min=min(alpha), alpha_max=max(alpha), alpha_mean=float(np.mean(alpha)),
                deterministically_eligible=sum(eligible), identity=manifest,
                scope='New-checkpoint DEV inference diagnosis; no independent applicability labels, task SR, or training-control evidence.')


class Bundle:
    def __init__(self, plan, resume, plan_path):
        self.plan, self.resume = plan, resume
        self.output = Path(plan['output'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.deadline = time.monotonic() + plan['max_seconds']
        self.identity = file_hash(plan_path)
        self.report = dict(schema='warm.nonreal.bundle-result.v1', status='running',
                           plan_sha256=self.identity, source_sha256=plan['source_sha256'],
                           paper_evidence_complete=False, paper_blockers=BLOCKERS, stages={},
                           gpu_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
                           scope='Qualified stages only; never interpreted as completion of all manuscript experiments.')

    def save(self):
        self.report['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write(self.output / 'bundle_summary.json', self.report)
        lines = ['# WARM one-command bundle record', '', f"Status: {self.report['status']}",
                 'Paper evidence complete: **false**', '', self.report['scope'], '',
                 f'Plan SHA256: {self.identity}', f"Source SHA256: {self.plan['source_sha256']}",
                 f'Raw directory: {self.output}', '', '| Stage | Status | Evidence / reason |', '|---|---|---|']
        for name, item in self.report['stages'].items():
            lines.append(f"| {name} | {item['status']} | {item.get('path', item.get('reason', ''))} |")
        if 'target' in self.report:
            lines += ['', '## Verified checkpoint', '', '```json', json.dumps(self.report['target'], indent=2), '```']
        for key in ('natural_gate', 'source_comparison'):
            if key in self.report:
                lines += ['', '## ' + key, '', '```json', json.dumps(self.report[key], indent=2), '```']
        lines += ['', '## Still missing from the paper', '']
        lines += [f'- {key}: {reason}' for key, reason in BLOCKERS.items()]
        lines += ['', 'Do not overwrite Tables 19/21/24 or old main-table results with this new checkpoint.',
                  'Archive these raw records and append verified values/scope to docs/nonreal72h/EXPERIMENT_RECORD_ZH.md.']
        temporary = self.output / 'EXPERIMENT_RECORD.generated.md.tmp'
        temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        os.replace(temporary, self.output / 'EXPERIMENT_RECORD.generated.md')

    def stage(self, name, argv, data, *, seconds=3600, optional=False):
        receipt_path = self.output / (name + '.receipt.json')
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt['plan_sha256'] != self.identity or receipt['argv'] != argv or receipt['files'] != tree_files(data):
                raise ValueError('completed stage evidence changed: ' + name)
            job = read(Path(receipt['job']) / 'run_manifest.json')
            code = receipt['exit_code']
            expected_status = 'complete' if code == 0 else 'failed'
            if (job['status'] != expected_status or job['exit_code'] != code
                    or job['source_sha256_after'] != self.plan['source_sha256'] or (code and not optional)):
                raise ValueError('completed stage execution record changed: ' + name)
            self.report['stages'][name] = dict(status='reused_verified' if code == 0 else 'blocked_recorded', path=str(data), exit_code=code)
            self.save()
            return code
        if data.exists():
            raise ValueError('unfinished stage output preserved; inspect and prepare a new attempt: ' + str(data))
        remaining = int(self.deadline - time.monotonic()) - 120
        if remaining < min(seconds, 600):
            raise TimeoutError('insufficient remaining wall time for ' + name)
        job = self.output / 'jobs' / (name + '-' + uuid.uuid4().hex[:8])
        self.report['stages'][name] = dict(status='running', path=str(job))
        self.save()
        try:
            code = run_logged(dict(max_seconds=min(seconds, remaining), evidence_type='engineering', claim=name),
                              argv, job, source=source_identity(ROOT))
        except BaseException as error:
            self.report['stages'][name] = dict(status='failed', path=str(job), reason=f'{type(error).__name__}: {error}')
            self.save()
            raise
        self.report['stages'][name] = dict(status='complete' if code == 0 else 'failed', path=str(job), exit_code=code)
        self.save()
        if code == 0 or (optional and data.exists()):
            if not data.exists() or not tree_files(data):
                raise ValueError('successful process did not publish evidence: ' + name)
            write(receipt_path, dict(plan_sha256=self.identity, argv=argv, files=tree_files(data), job=str(job), exit_code=code))
        elif not optional:
            raise RuntimeError(f'{name} failed with exit {code}; logs preserved at {job}')
        return code

    def execute(self):
        # The original launcher owns full parent-state hashing and four-rank restore.
        training_root = Path(self.plan['resume_plan']).parent
        if not (training_root / 'summary.json').is_file() or read(training_root / 'summary.json')['status'] != 'complete':
            if (training_root / 'started.json').exists():
                raise ValueError('original continuation has started but is incomplete; do not restart or overwrite it')
            from nonreal_resume import live_preflight
            live_preflight(self.resume)
        renderer = self.output / 'renderer'
        self.stage('renderer', [sys.executable, str(ROOT / 'scripts/nonreal_bundle.py'), '--renderer-output', str(renderer)],
                   renderer, seconds=120, optional=True)
        if not (training_root / 'summary.json').is_file() or read(training_root / 'summary.json')['status'] != 'complete':
            # A bounded outer runner also covers hashing/initialization beyond the trainer's own timeout.
            job = self.output / 'jobs' / ('training-' + uuid.uuid4().hex[:8])
            self.report['stages']['training'] = dict(status='running', path=str(job))
            self.save()
            code = run_logged(dict(max_seconds=min(22 * 3600, int(self.deadline - time.monotonic()) - 600),
                                   evidence_type='training', claim='Original 300-to-15000 full-state continuation'),
                              ['bash', str(Path(self.plan['resume_code']) / 'scripts/acp_nonreal_resume.sh'), self.plan['resume_plan']],
                              job, source=source_identity(ROOT))
            if code:
                self.report['stages']['training'] = dict(status='failed', path=str(job), exit_code=code)
                self.save()
                raise RuntimeError('continuation failed; preserve the original state and inspect ' + str(job))
        target = verified_target(self.plan, self.resume)
        self.report['target'] = target
        self.report['stages']['training'] = dict(status='verified_complete', path=target['training_summary'])
        self.save()
        checkpoint_dir = self.output / 'checkpoint_inspection'
        self.stage('checkpoint_inspection', [sys.executable, str(ROOT / 'scripts/inspect_nonreal_gate_checkpoint.py'),
                   '--checkpoint', target['weights'], '--output', str(checkpoint_dir / 'gate_tensors.json')], checkpoint_dir, seconds=300)
        source = self.output / 'source'
        source.mkdir(exist_ok=True)
        forwarded = ['--train-config', self.resume['config'], '--checkpoint', target['weights'],
                     '--asset-root', self.plan['asset_root'], '--tasks', 'press_button', 'put_back_block',
                     '--episodes-per-task', '5', '--prefixes-per-episode', '5', '--nfe', '20', '--seed', '3407', '--experiment', 'source']
        def collect(mode):
            output = source / ('mode-' + mode)
            self.stage('source_' + mode, [sys.executable, str(ROOT / 'scripts/probe_nonreal_full_null.py'), *forwarded,
                       '--source-mode', mode, '--output', str(output)], output, seconds=3600)
        collect('full')
        expected = dict(self.plan['dev_identity'], checkpoint_sha256=target['checkpoint_sha256'],
                        train_config_sha256=self.resume['config_sha256'], source_mode='full', experiment='source', nfe=20)
        gate = full_gate_report(source / 'mode-full', expected)
        self.report['natural_gate'] = gate
        write(self.output / 'natural_gate.json', gate)
        self.save()
        if gate['nonzero_g']:
            for mode in ('scale_only', 'gaussian'):
                collect(mode)
            merged = self.output / 'paired_source'
            self.stage('paired_source', [sys.executable, str(ROOT / 'scripts/nonreal_bundle.py'),
                       '--merge-source', str(source), '--merge-output', str(merged)], merged, seconds=600)
            self.report['source_comparison'] = read(merged / 'summary.json')
        else:
            self.report['stages']['source_controls'] = dict(status='not_run_zero_activation',
                reason='Natural Full g is zero for all 50 new-checkpoint prefixes; do not rerun degenerate controls.')
        self.report['status'] = 'completed_available_stages'
        self.save()


def renderer_check(output):
    output.mkdir(parents=True, exist_ok=False)
    from check_warm_rmbench_eval_runtime import renderer_smoke
    report = dict(status='running', scope='Rendering smoke only; not full-state restore qualification.')
    code = 1
    try:
        renderer_smoke()
        report['status'] = 'passed'
        code = 0
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        write(output / 'result.json', report)
    return code


def merge_sources(source, output):
    # Reuse the Full process already completed; never rerun it to obtain pairing.
    output.mkdir(parents=True, exist_ok=False)
    for mode in ('full', 'scale_only', 'gaussian'):
        (output / ('mode-' + mode)).symlink_to((source / ('mode-' + mode)).resolve(), target_is_directory=True)
    from run_nonreal_source_modes import merge
    result = merge(output)
    result['limitation'] = 'Fixed-checkpoint DEV source intervention; no independent training, task SR, or applicability labels.'
    write(output / 'summary.json', result)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--renderer-output', type=Path)
    parser.add_argument('--merge-source', type=Path)
    parser.add_argument('--merge-output', type=Path)
    args = parser.parse_args()
    if args.renderer_output:
        return renderer_check(args.renderer_output)
    if args.merge_source:
        return merge_sources(args.merge_source, args.merge_output)
    if not args.plan:
        parser.error('--plan is required')
    plan, resume = validate_plan(args.plan)
    if args.validate:
        print(json.dumps(dict(status='cpu_preflight_passed', source_sha256=plan['source_sha256'],
                              paper_evidence_complete=False, remaining_live_checks='Parent full-state bytes, four-rank runtime/restore, new-checkpoint DEV inference.')))
        return 0
    import fcntl
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'bundle.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (Path(plan['resume_plan']).parent / 'run.lock').open('a') as training_lock:
            fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Release before handing the original launcher its own lock.
        bundle = Bundle(plan, resume, args.plan)
        try:
            bundle.execute()
            return 0
        except Exception as error:
            bundle.report.update(status='failed_or_incomplete', error=f'{type(error).__name__}: {error}')
            bundle.save()
            print(bundle.report['error'], file=sys.stderr, flush=True)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
