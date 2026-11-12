"""Adopt sealed PB branches and preserve interrupted branches without reruns."""
from pathlib import Path
import hashlib
import json
import shutil

NEW_CONTROL='anchor-continuation-final'
PRE_CONTROL='anchor-continuation-initial'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require_resource_only_plan(base, current):
    allowed = {'output', 'collector_commit', 'collector_sha256',
               'helper_sha256', 'continuation_resource_stage'}
    changed = {k for k in set(base)|set(current) if base.get(k) != current.get(k)}
    if not changed <= allowed or 'output' not in changed:
        raise ValueError('Continuation may change only its output and executor/resource metadata')
    a, b = Path(base['output']).resolve(), Path(current['output']).resolve()
    if a.is_relative_to(b) or b.is_relative_to(a):
        raise ValueError('A continuation must preserve a disjoint original output tree')
    return changed


def require_inventory(manifest, order):
    if (manifest.get('schema') != 'warm.original-PB-remaining-branches.v1'
            or manifest.get('new_work') != 'remaining_original_branches_only'
            or manifest.get('prior_clock_reset') is not False):
        raise ValueError('Not a prospectively registered remaining-only continuation')
    complete, unknown, pending = (manifest[k] for k in ['completed', 'unknown', 'unstarted'])
    groups = [set(complete), set(unknown), set(pending)]
    if (len(order) != len(set(order)) or len(pending) != len(groups[2])
            or set.union(*groups) != set(order)
            or any(groups[i]&groups[j] for i in range(3) for j in range(i))):
        raise ValueError('Every original branch must be accounted for exactly once')
    if not set(order[:2]) <= groups[0]:
        raise ValueError('Two original repeated Full controls are required before new qualification')
    if pending != [n for n in order if n in groups[2]]:
        raise ValueError('Remaining branch order cannot depend on outcomes')
    for item in unknown.values():
        count=item.get('complete_command_records',0)
        if count is not None and (type(count) is not int or not 0<=count<=32):
            raise ValueError('Interrupted command accounting must remain bounded and explicit')
    return manifest


def require_old_full_repeats(a, aa):
    keys=['parent_hashes','endpoint_hashes','endpoint_status','executed_steps',
          'observed_effect','independent_label','error','parent_progress',
          'endpoint_progress','secondary_prefix4']
    if any(a[k]!=aa[k] for k in keys):
        raise ValueError('Original repeated Full controls differ; do not qualify a new pool')
    return True


def bind_manifest(path, expected, plan, order, sealed_query_sha256):
    if sha(path) != expected:
        raise ValueError('Continuation inventory changed')
    m = require_inventory(read(path), order)
    if sha(m['base_plan']) != m['base_plan_sha256']:
        raise ValueError('Original query plan changed')
    require_resource_only_plan(read(m['base_plan']), plan)
    if m['sealed_query_sha256'] != sealed_query_sha256:
        raise ValueError('Continuation must retain original candidates and predictions')
    for item in m['bound_files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Old terminal/raw evidence changed: '+item['path'])
    terminal = read(m['old_stage_operator'])
    if terminal['status'] not in ('terminal_fixed_PB_stage_all_existing_output_archived',
                                  'failed_or_interrupted_preserve_all'):
        raise ValueError('The preceding physical stage is not terminal')
    return m


def adopt_completed(manifest, output, *, query_id, verifier):
    """Copy only complete immutable records; never invent an old operator."""
    base = Path(read(manifest['base_plan'])['output']).resolve()
    output = Path(output)
    for name, item in manifest['completed'].items():
        source = Path(item['directory']).resolve()
        if not source.is_relative_to(base/'branches'):
            raise ValueError('Foreign branch directory')
        if sha(source/'complete.json') != item['complete_sha256']:
            raise ValueError('Original complete receipt changed')
        row = verifier(source, query_id=query_id, branch_id=name)
        if row['sealed_query_sha256'] != manifest['sealed_query_sha256']:
            raise ValueError('Foreign candidate/prediction seal')
        destination = output/'branches'/name/'attempt-001'
        if not destination.exists():
            if any(f.is_symlink() for f in source.rglob('*')):
                raise ValueError('Branch adoption requires original regular files')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
        adopted = verifier(destination, query_id=query_id, branch_id=name)
        if adopted != row or sha(destination/'complete.json') != item['complete_sha256']:
            raise ValueError('Adopted complete record differs')


def unknown_record(manifest, name, query_id):
    """An interrupted old attempt has no invented endpoint or success label."""
    item = manifest['unknown'][name]
    return dict(query_id=query_id, branch_id=name, parent_recreated=False,
        parent_hashes=None, endpoint_hashes=None,
        endpoint_status='unknown_terminal_infrastructure',
        executed_steps=item.get('complete_command_records', 0),
        observed_effect=None, independent_label=dict(applicable=None,
        reason='Archived interrupted attempt; no automatic rerun or imputation'),
        error=item['reason'], original_attempt_directory=item['directory'],
        secondary_prefix4=None, resource_interruption_preserved=True)
