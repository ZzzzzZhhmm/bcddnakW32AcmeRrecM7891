"""Admit immutable pool records without turning pending labels into failures."""
from pathlib import Path

from .fresh_pool import read_verified_branch
from .resumable_pairs import file_sha


def load_pool_dependencies(pool, *, query_id, selected_rank, allow_pending=False):
    """Two exact initial anchors are mandatory, including for pipelined work.

    Pipelining may defer only the already-selected raw branch and final pool
    anchor. Both remain explicit dependencies of the complete query report.
    Existing terminal records are always verified, even in pending mode.
    """
    pool = Path(pool)
    if type(selected_rank) is not int or not 0 <= selected_rank < 32:
        raise ValueError('original selected K32 rank required')
    seal_sha = file_sha(pool / 'sealed_query.json')
    records, markers, pending = {}, {}, []
    for name in ('anchor-a', 'anchor-aa', 'rank-' + str(selected_rank), 'anchor-final'):
        found = list((pool / name).glob('attempt-*/complete.json'))
        if len(found) > 1:
            raise ValueError('duplicate terminal branch; never select by outcome: ' + name)
        if not found:
            if allow_pending and (name.startswith('rank-') or name == 'anchor-final'):
                pending.append(name)
                continue
            raise ValueError('required completed pool dependency missing: ' + name)
        marker = found[0]
        record = read_verified_branch(marker.parent, query_id=query_id, branch_id=name)
        if record['sealed_query_sha256'] != seal_sha:
            raise ValueError('pool dependency belongs to a different query seal')
        records[name], markers[name] = record, marker
    anchor = records['anchor-a']
    for name, record in records.items():
        if record['parent_hashes'] != anchor['parent_hashes']:
            raise ValueError('completed dependency parent differs: ' + name)
        if name.startswith('anchor') and (
            record['endpoint_status'] != 'complete_horizon'
            or record['endpoint_hashes'] != anchor['endpoint_hashes']
            or record['observed_effect'] != anchor['observed_effect']
        ):
            raise ValueError('existing exact Full anchor differs: ' + name)
    return dict(anchor=anchor, raw=records.get('rank-' + str(selected_rank)),
                markers=markers, pending=pending, complete=not pending)
