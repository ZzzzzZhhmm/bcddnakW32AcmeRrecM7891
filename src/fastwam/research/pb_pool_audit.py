"""Read-only CPU audit of the three fixed original-PB candidate pools.

This module never imports a model, simulator, Torch, Git, or a network client.
Invalid and interrupted endpoints remain missing on the common H32 support.
"""
from pathlib import Path
import hashlib
import json
import tarfile

import numpy as np

from .candidate_pool import centered_effect_metrics
from .pb_mechanism import label, repair, WEIGHT_SHA256, TRACE_PLAN_SHA256


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(a.dtype.str.encode('ascii') + b'\0')
    h.update(json.dumps(list(a.shape), separators=(',', ':')).encode('ascii') + b'\0')
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def verify_archive(root, receipt):
    root = Path(root).resolve()
    archive = Path(receipt['archive'])
    manifest = read(archive.with_name(archive.name.removesuffix('.tar.gz') + '.manifest.json'))
    if sha(archive) != receipt['sha256'] or not receipt['all_raw_archived']:
        raise ValueError('Raw archive identity differs')
    actual = {f.relative_to(root).as_posix() for f in root.rglob('*') if f.is_file()}
    if actual != set(manifest) or len(manifest) != receipt['files']:
        raise ValueError('Terminal tree and manifest membership differ')
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        if len(names) != len(set(names)) or set(names) != set(manifest):
            raise ValueError('Duplicate or incomplete archive members')
        for name, item in manifest.items():
            f = root / name
            if (not f.resolve().is_relative_to(root) or f.is_symlink()
                    or not tar.getmember(name).isfile()):
                raise ValueError('Unsafe evidence member')
            data = tar.extractfile(name).read()
            if (len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']
                    or f.stat().st_size != item['bytes'] or sha(f) != item['sha256']):
                raise ValueError('Raw evidence changed: ' + name)
    return dict(path=str(archive), sha256=sha(archive), members=len(manifest),
                manifest_sha256=sha(archive.with_name(archive.name.removesuffix('.tar.gz') + '.manifest.json')))


def verify_branch(root, name, actions, seal_hash, query_id):
    directory = Path(root) / 'branches' / name / 'attempt-001'
    receipt = read(directory / 'complete.json')
    if receipt['status'] != 'sealed_terminal_branch' or receipt['query_id'] != query_id or receipt['branch_id'] != name:
        raise ValueError('Foreign branch receipt')
    for key, value in receipt['files'].items():
        f = directory / key
        if (not f.resolve().is_relative_to(directory.resolve()) or f.is_symlink()
                or f.stat().st_size != value['bytes'] or sha(f) != value['sha256']):
            raise ValueError('Changed branch member')
    row = read(directory / 'result.json')
    if not row['parent_recreated'] or row['query_id'] != query_id or row['branch_id'] != name or row['sealed_query_sha256'] != seal_hash:
        raise ValueError('Branch is not bound to the frozen query')
    commands = [json.loads(x) for x in (directory / 'commands.jsonl').read_text().splitlines()]
    if len(commands) != row['executed_steps'] or not 0 <= len(commands) <= 32:
        raise ValueError('Accepted command count differs')
    for i, command in enumerate(commands):
        if (command['index'] != i or not np.array_equal(np.asarray(command['command'], dtype=np.float32), actions[i])
                or not np.isfinite(command['command']).all()):
            raise ValueError('Accepted command differs from sealed action')
        if i and (command['physical_before'] != commands[i-1]['physical_after'] or command['task_before'] != commands[i-1]['task_after']):
            raise ValueError('Accepted physical/task chain is discontinuous')
    if commands and (commands[0]['task_before'] != row['parent_progress'] or commands[-1]['task_after'] != row['endpoint_progress']):
        raise ValueError('Stored task endpoint differs')
    status = row['endpoint_status']
    if status not in ('complete_horizon', 'invalid_branch', 'early_success'):
        raise ValueError('Not a sealed physical terminal')
    if row['independent_label'] != label(row['parent_progress'], row['endpoint_progress'], valid_branch=status != 'invalid_branch'):
        raise ValueError('Blind native count label differs')
    error = None
    if status == 'complete_horizon':
        if len(commands) != 32 or row['observed_effect'] is None:
            raise ValueError('Incomplete H32 endpoint')
        a = np.load(directory / 'parent_dino_spatial.npy', allow_pickle=False)
        b = np.load(directory / 'endpoint_dino_spatial.npy', allow_pickle=False)
        if a.shape != (4, 768) or b.shape != (4, 768) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Original uniform DINO descriptor differs')
        effect = (b.astype(np.float64) - a.astype(np.float64)).mean(axis=0)
        if not np.allclose(effect, row['observed_effect'], rtol=1e-6, atol=1e-6):
            raise ValueError('Stored H32 effect differs from endpoint data')
        error = float(np.max(np.abs(effect - row['observed_effect'])))
    elif row['observed_effect'] is not None or (status == 'invalid_branch' and row['independent_label']['applicable'] is not None):
        raise ValueError('Invalid or censored endpoint was imputed')
    return row, error


def verify_metrics(data, frozen):
    result = centered_effect_metrics(data)
    if 'constant_query' in result['references']:
        result['references']['constant_query']['absolute_mse'] = None
        result['references']['constant_query']['absolute_mse_unavailable_reason'] = 'Required query effect not recorded;only centered constant comparison is identified'
    if result != frozen:
        raise ValueError('Frozen metrics differ from the identical complete-H32 support')
    return result


def audit_pool(root, *, query, plan_sha256):
    root = Path(root)
    operator = read(root / 'operator.json')
    if operator['status'] != 'complete_original_PB_fixed_query_pool' or operator['query'] != query or operator['plan_sha256'] != plan_sha256:
        raise ValueError('A live, incomplete, or foreign query cannot release metrics')
    seal = read(root / 'sealed_query.json')
    if (query not in (8, 48, 96) or seal['query'] != query or seal['primary_horizon'] != 32
            or seal['descriptor'] != 'original_uniform_mean_DINO_delta_v1'
            or seal['required_effect_available'] or seal['population_success_rate_available']
            or seal['checkpoint_sha256'] != WEIGHT_SHA256 or seal['source_plan_sha256'] != TRACE_PLAN_SHA256
            or seal['before_any_new_branch_truth'] is not True):
        raise ValueError('Fixed query or measurement scope differs')
    seal_hash = sha(root / 'sealed_query.json')
    with np.load(root / 'sealed_arrays.npz', allow_pickle=False) as z:
        predicted, historical, valid = [z[k].copy() for k in ['predicted', 'historical', 'valid']]
        if (predicted.shape != (32, 768) or historical.shape != (32, 768) or valid.shape != (32,)
                or valid.dtype != np.bool_ or array_sha(predicted) != seal['predicted_sha256']
                or array_sha(historical) != seal['historical_sha256'] or np.flatnonzero(valid).tolist() != seal['original_valid_ranks']):
            raise ValueError('Candidate prediction or coverage seal differs')
        actions = {name: z[name].copy() for name in seal['branch_order']}
        for name, a in actions.items():
            if a.shape != (32, 14) or a.dtype != np.float32 or not np.isfinite(a).all() or array_sha(a) != seal['action_hashes'][name]:
                raise ValueError('Candidate action seal differs')
    continuation = read(root / 'continuation_receipt.json')
    sequence = ['anchor-continuation-initial'] + seal['branch_order'] + ['anchor-continuation-final']
    actions.update({name: actions['anchor-a'] for name in ['anchor-continuation-initial', 'anchor-continuation-final']})
    records, delta_errors, statuses = {}, {}, {}
    for name in sequence:
        if name in continuation['retained_unknown']:
            row = read(root / (name + '_retained_unknown.json'))
            if row['endpoint_status'] != 'unknown_terminal_infrastructure' or row['observed_effect'] is not None or row['independent_label']['applicable'] is not None or not row['resource_interruption_preserved']:
                raise ValueError('An old interrupted branch was replaced or imputed')
        else:
            row, error = verify_branch(root, name, actions[name], seal_hash, f'original-pb12000-seed0-q{query}')
            delta_errors[name] = error
        records[name] = row
        statuses[name] = dict(status=row['endpoint_status'], accepted_commands=row['executed_steps'], label=row['independent_label']['applicable'])
    base = records['anchor-a']
    same = ['parent_hashes', 'endpoint_hashes', 'endpoint_status', 'executed_steps', 'observed_effect', 'independent_label', 'error', 'parent_progress', 'endpoint_progress', 'secondary_prefix4']
    for name, row in records.items():
        if row['endpoint_status'] == 'unknown_terminal_infrastructure':
            continue
        if row['parent_hashes'] != base['parent_hashes'] or (name.startswith('anchor-') and any(row[k] != base[k] for k in same)):
            raise ValueError('Parents or repeated Full controls differ')
    data = read(root / 'candidate_effects.json')
    if (data['task'] != 'press_button' or data['cohort'] != f'originalPB12000_seed0_q{query}'
            or data['descriptor'] != 'uniform_mean_DINO_delta' or data['required_effect_available'] is not False):
        raise ValueError('Candidate table measurement scope differs')
    candidates = data['candidates']
    if [c['candidate_id'] for c in candidates] != [f'rank-{r}' for r in np.flatnonzero(valid)]:
        raise ValueError('Fixed candidate order/denominator differs')
    for rank, item in zip(np.flatnonzero(valid), candidates):
        row = records[item['candidate_id']]
        if (item['valid'] is not True or item['horizon'] != 32
                or not np.array_equal(np.asarray(item['predicted_effect'], dtype=np.float32), predicted[rank])
                or not np.array_equal(np.asarray(item['historical_effect'], dtype=np.float32), historical[rank])
                or any(item[k] != row[k] for k in ['observed_effect', 'endpoint_status', 'executed_steps'])
                or item['applicable'] != row['independent_label']['applicable']):
            raise ValueError('Candidate table differs from sealed physical evidence')
    metrics = verify_metrics(data, read(root / 'centered_effects.json'))
    contrast = repair(records, seal['selected_rank'])
    if contrast != read(root / 'selected_raw_refined_PB_contrast.json'):
        raise ValueError('Selected repair contrast differs')
    return dict(query=query, query_plan_sha256=plan_sha256, sealed_query_sha256=seal_hash,
                all_fixed_candidates_accounted=True, candidate_statuses={k:v for k,v in statuses.items() if k.startswith('rank-')},
                full_control_statuses={k:v for k,v in statuses.items() if k.startswith('anchor-')},
                old_unknown_branches_preserved=continuation['retained_unknown'],
                original_AABA_complete=continuation['original_AABA_complete'],
                same_parents_and_repeated_Full_terminals=True, metrics=metrics,
                selected_contrast=contrast, DINO_recomputed_max_absolute_errors=delta_errors,
                independent_source_episodes=1, independent_source_states=1,
                population_SR_available=False, population_or_training_seed_CI=None,
                no_new_policy_or_physical_actions=True, primary_descriptor_unchanged=True)
