"""Unit fixtures below are statistical/archival checks, not robot evidence."""
from copy import deepcopy
from pathlib import Path
import hashlib, importlib.util, io, json, tarfile
import pytest

from fastwam.research.matched_report import summarize_fixed_pairs

IDENTITY=dict(design_sha256="d"*64,evaluation_source_sha256="e"*64,
              full_checkpoint_sha256="a"*64,pooled_checkpoint_sha256="b"*64,
              root_seed=3407,checkpoint_update=3000)


def row(task="rearrange_blocks",index=0,a=1,b=0):
    return dict(task=task,episode_id=index,phase="formal",study_identity=IDENTITY.copy(),
                plan_sha256=f"legitimate-shard-{index//10}",
                paired_initial_and_noise_verified=True,
                roles={"full":dict(status="complete_native",success=a),
                       "pooled_memory_prior":dict(status="complete_native",success=b)})


def test_complete_two_task_counts_and_legitimate_shard_plans():
    rows=[row(t,i,int(i<30),int(i>=20)) for t in ("rearrange_blocks","battery_try") for i in range(50)]
    report=summarize_fixed_pairs(rows)
    assert report["cohort_complete"] and report["primary"] is not None
    for x in report["tasks"].values():
        assert x["cells"]==dict(both_success=10,both_failure=0,full_only=20,pooled_only=20)
        assert x["full_successes"]==x["pooled_successes"]==30
    assert report["primary"]["equal_task_macro_difference"]==0
    assert not report["missing_skill_noninferiority_test_performed"]


def test_zero_discordances_never_give_zero_width_or_missing_skill_NI():
    rows=[row(t,i,0,0) for t in ("rearrange_blocks","battery_try") for i in range(50)]
    p=summarize_fixed_pairs(rows)["primary"]
    lo,hi=p["macro_ci95_conservative"]
    assert lo<0<hi


def test_incomplete_and_runtime_unknowns_never_become_failures():
    x=row();x["roles"]["pooled_memory_prior"]=dict(status="loading_typed_policy",success=None)
    r=summarize_fixed_pairs([x])
    assert r["primary"] is None and r["descriptive_macro_difference"] is None
    assert r["tasks"]["rearrange_blocks"]["complete_pairs"]==0
    assert r["tasks"]["rearrange_blocks"]["known_role_outcomes"]==dict(full=1,pooled_memory_prior=0)


def test_identified_invalid_output_is_zero_only_with_native_proof_status():
    x=row(a=0,b=1);x["roles"]["full"]["status"]="identifiable_policy_failure"
    r=summarize_fixed_pairs([x])["tasks"]["rearrange_blocks"]
    assert r["cells"]["pooled_only"]==1 and r["identifiable_policy_failures"]["full"]==1


@pytest.mark.parametrize("change",["foreign_case","bool_outcome","runtime_zero","DEV","same_checkpoint","missing_pair_proof","foreign_role"])
def test_protocol_or_label_corruption_is_rejected(change):
    x=row()
    if change=="foreign_case":x["episode_id"]=50
    elif change=="bool_outcome":x["roles"]["full"]["success"]=True
    elif change=="runtime_zero":x["roles"]["full"]["status"]="unknown_runtime";x["roles"]["full"]["success"]=0
    elif change=="DEV":x["phase"]="recorded_dev"
    elif change=="same_checkpoint":x["study_identity"]["pooled_checkpoint_sha256"]="a"*64
    elif change=="missing_pair_proof":x["paired_initial_and_noise_verified"]=False
    else:x["roles"]["candidate_null"]=x["roles"].pop("pooled_memory_prior")
    with pytest.raises(ValueError):summarize_fixed_pairs([x])


def test_duplicate_and_mixed_models_rejected():
    x,y=row(),row(index=1)
    with pytest.raises(ValueError):summarize_fixed_pairs([x,deepcopy(x)])
    y["study_identity"]["full_checkpoint_sha256"]="c"*64
    with pytest.raises(ValueError):summarize_fixed_pairs([x,y])


def cli():
    p=Path(__file__).resolve().parents[1]/"scripts/summarize_rebuttal_matched_final3000.py"
    s=importlib.util.spec_from_file_location("matched_report_cli",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


def make_archive(tmp_path):
    a=tmp_path/"attempt";w=a/"worker";w.mkdir(parents=True)
    data=b"unit-only-raw-record"
    (w/"record.txt").write_bytes(data)
    manifest={"record.txt":dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())}
    archive=a/"verified_worker_evidence.tar.gz"
    with tarfile.open(archive,"w:gz") as tar:
        for name,value in {"record.txt":data,"archive_manifest.json":json.dumps(manifest).encode()}.items():
            member=tarfile.TarInfo(name);member.size=len(value);tar.addfile(member,io.BytesIO(value))
    receipt=dict(path=str(archive),files=1,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (a/"archive_receipt.json").write_text(json.dumps(receipt))
    return a


def test_archive_checks_original_bytes_and_rejects_unarchived_extra(tmp_path):
    m=cli();a=make_archive(tmp_path)
    assert m.verify_role_archive(a)["files"]==1
    (a/"worker/record.txt").write_bytes(b"changed")
    with pytest.raises(ValueError):m.verify_role_archive(a)
    (a/"worker/record.txt").write_bytes(b"unit-only-raw-record")
    (a/"worker/extra.txt").write_text("not archived")
    with pytest.raises(ValueError):m.verify_role_archive(a)


def test_primary_latex_unavailable_for_incomplete_cohort():
    with pytest.raises(ValueError):cli().latex_table(summarize_fixed_pairs([]))
