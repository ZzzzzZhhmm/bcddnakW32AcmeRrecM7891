#!/usr/bin/env python3
"""Apply an explicit orchestration-only time budget to an unchanged frozen plan.

The original collector, model, samples, commands and resource deadline remain
bound to the original plan. Failed raw results are never edited or imputed.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import sys


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def qualify(amendment, plan, plan_sha, now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if amendment.get('schema') != 'warm.rebuttal.execution-time-amendment.v1':
        raise ValueError('explicit execution amendment required')
    if (plan_sha != amendment['original_plan_sha256']
            or plan['first_admitted_utc'] != amendment['first_admitted_utc']
            or plan['absolute_deadline_utc'] != amendment['absolute_deadline_utc']
            or plan['controller_source_sha256'] != amendment['frozen_controller_sha256']):
        raise ValueError('original plan, controller or resource window changed')
    if (plan['phase'] != 'formal' or plan['task'] != 'battery_try'
            or plan['worker_max_seconds'] != 2400
            or amendment['effective_worker_max_seconds'] != 3600
            or amendment['applies_to_roles'] != ['full', 'pooled_memory_prior']
            or amendment['scientific_fields_changed'] != []
            or amendment['maximum_parallel_formal_slots'] != 6
            or amendment['original_campaign_gpu_hour_ceiling'] != 120):
        raise ValueError('amendment must change only the common worker wall limit')
    end = datetime.datetime.fromisoformat(plan['absolute_deadline_utc'])
    if now.tzinfo is None or end <= now or end > datetime.datetime(2026, 11, 12, 18, tzinfo=datetime.timezone.utc):
        raise ValueError('original resource deadline expired or changed')
    if not amendment['reason'].startswith('Three predeclared BT attempts'):
        raise ValueError('observed infrastructure reason required')
    return dict(plan, worker_max_seconds=3600)


def terminal_view(row, worker, archive):
    """Classify an authenticated unfinished status in a new read-only view."""
    if row['status'] in ('complete_native', 'identifiable_policy_failure', 'complete_recorded_DEV'):
        return row
    key = str(Path(worker).parent.resolve())
    proof = archive['attempts'].get(key)
    if (proof is None or proof['actual_exit_code'] == 0 or row.get('success') is not None
            or sha(Path(worker) / 'result.json') != proof['result_sha256']
            or sha(Path(worker).parent / 'exit.json') != proof['exit_sha256']):
        raise ValueError('unknown attempt lacks its actual terminal/raw archive proof')
    return dict(row, status='unknown_terminal_preserved_under_execution_amendment',
                original_raw_status=row['status'], success=None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--amendment', type=Path, required=True)
    ap.add_argument('--expected-amendment-sha256', required=True)
    ap.add_argument('--max-new-roles', type=int, default=1)
    ap.add_argument('--validate-only', action='store_true')
    cli = ap.parse_args()
    if sha(cli.amendment) != cli.expected_amendment_sha256 or not 1 <= cli.max_new_roles <= 2:
        raise ValueError('amendment bytes or invocation bound differs')
    a = read(cli.amendment);planfile = Path(a['original_plan']);plan = read(planfile)
    if sha(Path(__file__)) != a['execution_entry_sha256']:
        raise ValueError('frozen execution entry changed')
    effective = qualify(a, plan, sha(planfile))
    proof_ref = a['known_terminal_archive_receipt'];proof = read(proof_ref['path'])
    if sha(proof_ref['path']) != proof_ref['sha256'] or not proof['all_existing_terminal_raw_archived']:
        raise ValueError('terminal evidence is not completely archived')
    if sha(proof['archive']) != proof['archive_sha256']:
        raise ValueError('terminal archive changed')
    root = Path(a['frozen_controller']).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('frozen_campaign', root / 'scripts/run_rebuttal_matched_campaign.py')
    frozen = importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    sys.path.insert(0, str(root))
    from scripts.nonreal_job import source_identity
    if source_identity(root)['sha256'] != a['frozen_controller_sha256']:
        raise ValueError('frozen scientific controller changed')
    original_validate = frozen.M.validate_plan
    original_result = frozen.verified_result
    original_write = frozen.M.atomic_json
    def validate(value):
        return qualify(a, original_validate(value), a['original_plan_sha256'])
    def result(worker, *args):
        return terminal_view(original_result(worker, *args), worker, proof)
    def write(path, value):
        if Path(path).name in ('attempt.json', 'campaign_state.json'):
            value = dict(value, execution_amendment=dict(path=str(cli.amendment.resolve()),
                sha256=cli.expected_amendment_sha256, effective_worker_max_seconds=3600,
                original_frozen_plan_unchanged=True))
        return original_write(path, value)
    # Only the orchestration view changes. Child collectors read original bytes.
    frozen.M.validate_plan = validate
    frozen.verified_result = result
    frozen.M.atomic_json = write
    argv = sys.argv
    try:
        sys.argv = ['frozen_campaign', '--plan', str(planfile), '--expected-plan-sha256',
                    a['original_plan_sha256'], '--resume-known-terminal', '--max-new-roles', str(cli.max_new_roles)]
        if cli.validate_only:
            sys.argv.append('--validate-only')
        frozen.main()
    finally:
        sys.argv = argv
        frozen.M.validate_plan = original_validate
        frozen.verified_result = original_result
        frozen.M.atomic_json = original_write
        if source_identity(root)['sha256'] != a['frozen_controller_sha256']:
            raise RuntimeError('immutable scientific controller changed')


if __name__ == '__main__':
    main()
