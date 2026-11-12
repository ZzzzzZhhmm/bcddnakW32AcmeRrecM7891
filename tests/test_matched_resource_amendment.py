from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path

import pytest
from fastwam.research import matched_native as M


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return dict(path=str(path),sha256=M.file_sha(path))


@pytest.fixture
def amended(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';rel='src/fastwam/research/matched_native.py';source=Path(M.__file__).read_text()
    for root in [old,new]:p=root/rel;p.parent.mkdir(parents=True);p.write_text(source)
    base=dict(phase='formal',first_admitted_utc='2026-11-11T10:00:00+00:00',absolute_deadline_utc='2026-11-11T22:00:00+00:00',controller_code=str(old),controller_source_sha256='a'*64,case_ids=list(range(10)),roles={'full':{'checkpoint_sha256':'full'},'pooled_memory_prior':{'checkpoint_sha256':'pooled'}},task='battery_try')
    ref=save(tmp_path/'original.json',base);prior=tmp_path/'previous';bound=[]
    for role in ['full','pooled_memory_prior']:bound.append(save(prior/f'0000-{role}/worker/result.json',dict(episode_id=0,role=role,status='complete_native')))
    terminal=save(prior/'operator.json',dict(status='terminal_fixed_invocation_complete_or_window_bound',ended_utc='2026-11-12T17:00:00+00:00'));exit_ref=save(prior/'run_manifest.json',dict(exit_code=0));budget=save(tmp_path/'old_budget.json',dict(authoritative_active_execution=dict(max_concurrent_GPU_slots=6,reserved_upper_GPU_hours=100)))
    plan={**base,'first_admitted_utc':'2026-11-12T17:20:00+00:00','absolute_deadline_utc':'2026-11-12T20:00:00+00:00','controller_code':str(new),'controller_source_sha256':'b'*64}
    proof=dict(original_controller=str(old),amended_controller=str(new),original_controller_sha256='a'*64,amended_controller_sha256='b'*64,files={rel:dict(original_sha256=M.file_sha(old/rel),amended_sha256=M.file_sha(new/rel))});pref=save(tmp_path/'equivalence.json',proof)
    reg=dict(schema='warm.matched-prospective-resource-amendment.v1',original_windows_unchanged=True,prior_unknowns_not_retried=True,stage_first_admitted_utc=plan['first_admitted_utc'],stage_absolute_deadline_utc=plan['absolute_deadline_utc'],old_master_deadline_unchanged='2026-11-12T18:00:00+00:00',original_plan=ref,original_controller_root=str(old),predecessor_terminal=terminal,predecessor_runner_exit=exit_ref,previous_budget_state=budget,max_concurrent_GPUs=6,reuses_terminal_GPU_slot=True,prior_reserved_upper_GPU_hours=100,combined_reserved_GPU_hours_upper_bound=100+8/3,old_known_role_records=bound,old_inventory_roots=[str(prior)],fixed_role_sequence=[[1,'pooled_memory_prior'],[1,'full']],old_unknown_attempts_preserved=[],controller_resource_equivalence=pref)
    rpath=tmp_path/'registration.json';plan['prospective_resource_registration']=save(rpath,reg)
    return plan,reg,rpath,new,proof,pref


def bind(plan,reg,path):plan['prospective_resource_registration']=save(path,reg)


def run(plan):return M.require_deadline(plan,datetime(2026,11,12,17,30,tzinfo=timezone.utc))


def test_explicit_new_stage_keeps_old_ceiling_and_science_intact(amended):
    p,r,_,_,_,_=amended;old=M.read_json(r['original_plan']['path']);assert run(p)==datetime(2026,11,12,20,tzinfo=timezone.utc).timestamp();assert M.read_json(r['original_plan']['path'])==old
    unregistered=deepcopy(p);unregistered.pop('prospective_resource_registration')
    with pytest.raises(ValueError,match='fixed campaign ceiling'):run(unregistered)


def test_original_manifests_without_a_controller_path_keep_their_explicit_source_identity(amended):
    p,r,path,_,_,_=amended;base=M.read_json(r['original_plan']['path']);base.pop('controller_code');r['original_plan']=save(Path(r['original_plan']['path']),base);bind(p,r,path)
    assert run(p)==datetime(2026,11,12,20,tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize('key,value',[('max_concurrent_GPUs',7),('reuses_terminal_GPU_slot',False),('prior_reserved_upper_GPU_hours',99),('combined_reserved_GPU_hours_upper_bound',121),('old_master_deadline_unchanged','2026-11-12T20:00:00+00:00'),('original_windows_unchanged',False),('prior_unknowns_not_retried',False),('fixed_role_sequence',[[0,'full']]),('fixed_role_sequence',[[1,'full']])])
def test_amendment_rejects_budget_reset_active_overlap_and_outcome_replacement(amended,key,value):
    p,r,path,_,_,_=amended;r[key]=value;bind(p,r,path)
    with pytest.raises(ValueError):run(p)


def test_changed_checkpoint_cannot_use_a_resource_amendment(amended):
    p,*_=amended;p['roles']=deepcopy(p['roles']);p['roles']['full']['checkpoint_sha256']='replacement'
    with pytest.raises(ValueError,match='scientific plan'):run(p)


def test_running_predecessor_is_not_a_released_slot(amended):
    p,r,path,_,_,_=amended;r['predecessor_terminal']=save(Path(r['predecessor_terminal']['path']),dict(status='running_0003-full',ended_utc='2026-11-12T17:00:00+00:00'));bind(p,r,path)
    with pytest.raises(ValueError,match='still owns'):run(p)


def test_old_unknown_attempt_cannot_be_omitted_from_inventory(amended):
    p,r,_,_,_,_=amended;save(Path(r['old_inventory_roots'][0])/'0001-full-unknown/worker/result.json',dict(status='loading_typed_policy',episode_id=1,role='full'))
    with pytest.raises(ValueError,match='omitted'):run(p)


def test_changing_scientific_ast_is_rejected_even_with_fresh_hashes(amended):
    p,r,path,new,proof,pref=amended;rel='src/fastwam/research/matched_native.py';f=new/rel;f.write_text(f.read_text().replace("ROLES=('full','pooled_memory_prior')","ROLES=('full','replacement')"));proof['files'][rel]['amended_sha256']=M.file_sha(f);r['controller_resource_equivalence']=save(Path(pref['path']),proof);bind(p,r,path)
    with pytest.raises(ValueError,match='scientific collector code'):run(p)


def test_same_byte_source_files_are_required_outside_resource_module(amended):
    p,r,path,new,proof,pref=amended;old=Path(proof['original_controller']);(old/'policy.py').write_text('value=1');(new/'policy.py').write_text('value=2');proof['files']['policy.py']=dict(original_sha256=M.file_sha(old/'policy.py'),amended_sha256=M.file_sha(new/'policy.py'));r['controller_resource_equivalence']=save(Path(pref['path']),proof);bind(p,r,path)
    with pytest.raises(ValueError,match='scientific collector implementation'):run(p)
