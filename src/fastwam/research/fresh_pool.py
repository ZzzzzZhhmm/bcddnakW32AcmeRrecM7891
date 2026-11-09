"""Small durable branch records for outcome-independent fresh-scene pools."""
from __future__ import annotations

import json
from pathlib import Path

from .resumable_pairs import file_sha


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False))
    temporary.replace(path)


def seal_branch(directory, result):
    directory = Path(directory)
    if not result.get("parent_recreated") or result["endpoint_status"] not in ("complete_horizon","invalid_branch","early_success"):
        raise ValueError("only a parent-verified terminal branch can be sealed")
    complete = result["endpoint_status"] == "complete_horizon"
    if complete != (result["executed_steps"] == 32 and result["observed_effect"] is not None):
        raise ValueError("only complete H32 branches have numerical observed effects")
    if not complete and result.get("observed_effect") is not None:
        raise ValueError("do not impute an effect for an incomplete branch")
    if result["endpoint_status"] == "invalid_branch" and result["independent_label"]["applicable"] is not None:
        raise ValueError("invalid branch must retain unknown applicability")
    if (directory/'complete.json').exists():
        raise FileExistsError('never replace a sealed branch result')
    atomic_json(directory/"result.json",result)
    files={p.relative_to(directory).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p))
           for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='complete.json'}
    receipt=dict(status='sealed_terminal_branch',query_id=result['query_id'],branch_id=result['branch_id'],files=files)
    atomic_json(directory/'complete.json',receipt)
    return receipt


def read_verified_branch(directory, *, query_id, branch_id):
    directory=Path(directory);receipt=json.loads((directory/'complete.json').read_text())
    if receipt['status']!='sealed_terminal_branch' or receipt['query_id']!=query_id or receipt['branch_id']!=branch_id:
        raise ValueError('sealed branch identity differs')
    for name,row in receipt['files'].items():
        path=directory/name
        if not path.resolve().is_relative_to(directory.resolve()) or path.stat().st_size!=row['bytes'] or file_sha(path)!=row['sha256']:
            raise ValueError('sealed branch artifact changed: '+name)
    return json.loads((directory/'result.json').read_text())


def next_branch(sequence, completed):
    """Take the first missing declared branch, independent of any outcome."""
    if len(set(sequence))!=len(sequence) or not set(completed)<=set(sequence):
        raise ValueError('duplicate or foreign branch identity')
    return next((b for b in sequence if b not in completed),None)
