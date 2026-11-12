import io
import json
import tarfile
from pathlib import Path

import pytest

from test_matched_resource_amendment import amended, bind, run, save
from fastwam.research import matched_native as M


def close_inventory(p,r,path):
    root=Path(r['old_inventory_roots'][0]);unknown=[]
    for role in M.ROLES:
        A=root/f'0000-{role}'
        save(A/'attempt.json',dict(case_id=0,role=role))
        save(A/'exit.json',dict(exit_code=0,ended_utc='2026-11-12T17:00:00+00:00'))
    A=root/'0001-pooled-unknown'
    save(A/'attempt.json',dict(case_id=1,role='pooled_memory_prior'))
    save(A/'exit.json',dict(exit_code=-15,ended_utc='2026-11-12T17:00:00+00:00'))
    result=save(A/'worker/result.json',dict(episode_id=1,role='pooled_memory_prior',status='loading_typed_policy',success=None))
    r['old_known_role_records'].append(result);unknown.append(A.name)
    r.update(schema='warm.matched-prospective-resource-amendment.v3',new_roles_only_never_attempted=True,all_old_invocations_terminal=True,old_unknown_attempts_preserved=unknown,fixed_role_sequence=[[1,'full'],[2,'full']])
    r['predecessor_terminal']=save(Path(r['predecessor_terminal']['path']),dict(status='failed_or_interrupted',ended_utc='2026-11-12T17:00:00+00:00'))
    r['predecessor_runner_exit']=save(Path(r['predecessor_runner_exit']['path']),dict(exit_code=1))
    seal_inventory(p,r,path)
    return A


def seal_inventory(p,r,path):
    roots=[Path(v) for v in r['old_inventory_roots']];files={}
    archive=path.parent/'inventory.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        for i,root in enumerate(roots):
            for f in sorted(root.rglob('*')):
                if f.is_file():
                    member=f'{i}/{f.relative_to(root).as_posix()}'
                    files[str(f.resolve())]=dict(bytes=f.stat().st_size,sha256=M.file_sha(f),member=member)
                    stream.add(f,arcname=member,recursive=False)
    manifest=save(path.parent/'inventory.json',dict(schema='warm.matched-terminal-inventory.v1',roots=r['old_inventory_roots'],files=files))
    r['terminal_inventory']=dict(manifest=manifest,archive=dict(path=str(archive),sha256=M.file_sha(archive)))
    bind(p,r,path)


def test_closed_unknown_stays_unknown_while_untouched_original_schedule_continues(amended):
    p,r,path,*_=amended;A=close_inventory(p,r,path);before=M.file_sha(A/'worker/result.json')
    assert run(p)>0 and M.file_sha(A/'worker/result.json')==before
    assert M.read_json(A/'worker/result.json')['success'] is None
    # The ordinary campaign remains blocked; only the explicit stage advances.
    rows=[M.read_json(v['path']) for v in r['old_known_role_records']]
    with pytest.raises(RuntimeError,match='adjudicated'):M.next_request(rows,list(range(10)))


@pytest.mark.parametrize('change',['retry_unknown','repeat_known','skip_unattempted','omit_unknown','active_old','missing_raw','changed_archive','false_flag','wrong_member','missing_metadata'])
def test_terminal_continuation_cannot_hide_unknowns_or_choose_later_samples(amended,change):
    p,r,path,*_=amended;A=close_inventory(p,r,path)
    if change=='retry_unknown':r['fixed_role_sequence']=[[1,'pooled_memory_prior']]
    elif change=='repeat_known':r['fixed_role_sequence']=[[0,'full']]
    elif change=='skip_unattempted':r['fixed_role_sequence']=[[2,'full']]
    elif change=='omit_unknown':r['old_unknown_attempts_preserved']=[]
    elif change=='active_old':
        save(A/'exit.json',dict(exit_code=None,ended_utc='2026-11-12T17:00:00+00:00'));seal_inventory(p,r,path)
    elif change=='missing_raw':(A/'new_unarchived.log').write_text('retained')
    elif change=='changed_archive':Path(r['terminal_inventory']['archive']['path']).write_bytes(b'changed')
    elif change=='false_flag':r['new_roles_only_never_attempted']=False
    elif change=='missing_metadata':
        (A/'attempt.json').unlink();seal_inventory(p,r,path)
    else:
        archive=Path(r['terminal_inventory']['archive']['path'])
        with tarfile.open(archive,'w:gz') as stream:
            entry=tarfile.TarInfo('unrelated');entry.size=1;stream.addfile(entry,io.BytesIO(b'x'))
        r['terminal_inventory']['archive']['sha256']=M.file_sha(archive)
    bind(p,r,path)
    with pytest.raises((ValueError,KeyError,RuntimeError)):run(p)


def test_attempt_without_result_is_still_censored_and_never_replaced(amended):
    p,r,path,*_=amended;A=close_inventory(p,r,path)
    r['old_known_role_records']=[v for v in r['old_known_role_records'] if Path(v['path'])!=A/'worker/result.json']
    (A/'worker/result.json').unlink();seal_inventory(p,r,path)
    assert run(p)>0
    r['fixed_role_sequence']=[[1,'pooled_memory_prior']];bind(p,r,path)
    with pytest.raises(ValueError,match='never-attempted'):run(p)
