#!/usr/bin/env python3
"""Audit frozen Full/pooled worker archives and emit the fixed-cohort report.

Read-only with respect to every experiment. No model, simulator, Git or network
access. A main-paper result table is emitted only for all 100 verified pairs.
"""
from pathlib import Path
import argparse, csv, datetime, hashlib, importlib.util, json, sys, tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from fastwam.research.matched_report import TASKS, ROLES, KNOWN, summarize_fixed_pairs

CONTROLLER = "861004fa79c2d454f0718e429729405fc783cc31d1997abbb5ffb43f35e9e6d9"
CORE = "87199cc6f815da5599fcb0ee9c0a9ffb67181f3750e34d16acc810f6218ffe80"
DESIGN = "3484ad1477a868607a916638c631c80df25501db59dd82edbf6651c768238e33"
read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_role_archive(attempt):
    """Verify all immutable TAR members against the receipt and original worker."""
    receipt = read(attempt / "archive_receipt.json")
    archive, worker = attempt / "verified_worker_evidence.tar.gz", attempt / "worker"
    if Path(receipt["path"]).resolve() != archive.resolve() or sha(archive) != receipt["sha256"]:
        raise ValueError("Role archive identity differs")
    if archive.stat().st_size != receipt["bytes"]:
        raise ValueError("Role archive length differs")
    with tarfile.open(archive) as tar:
        names = [m.name for m in tar.getmembers()]
        if len(names) != len(set(names)) or names.count("archive_manifest.json") != 1:
            raise ValueError("Duplicate/missing archive members")
        manifest = json.load(tar.extractfile("archive_manifest.json"))
        if set(names) != set(manifest) | {"archive_manifest.json"} or receipt["files"] != len(manifest):
            raise ValueError("Incomplete/extra raw evidence members")
        actual = {p.relative_to(worker).as_posix() for p in worker.rglob("*") if p.is_file()}
        if actual != set(manifest):
            raise ValueError("Worker evidence changed after archival")
        for name, item in manifest.items():
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts or not tar.getmember(name).isfile():
                raise ValueError("Unsafe/non-file evidence member")
            h = hashlib.sha256()
            count = 0
            with tar.extractfile(name) as f:
                for b in iter(lambda: f.read(8*1024**2), b""):
                    h.update(b)
                    count += len(b)
            original = worker / name
            if count != item["bytes"] or h.hexdigest() != item["sha256"] or sha(original) != item["sha256"]:
                raise ValueError("Original or archived raw evidence hash differs")
    return receipt


def latex_table(report):
    if report["primary"] is None:
        raise ValueError("Incomplete cohorts cannot supply a primary result table")
    rows = []
    for task, label in [("rearrange_blocks", "Rearrange Blocks"), ("battery_try", "Battery Try")]:
        x = report["tasks"][task]; b = x["complete_case_bounds"]
        lo, hi = b["ci95_conservative"]
        rows.append(f'{label} & {x["full_successes"]}/50 & {x["pooled_successes"]}/50 & '
                    f'{100*b["difference"]:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] ' + r"\\")
    p = report["primary"]; lo, hi = p["macro_ci95_conservative"]
    rows.append(f'Equal-task macro & & & {100*p["equal_task_macro_difference"]:+.1f} '
                f'[{100*lo:+.1f}, {100*hi:+.1f}] ' + r"\\")
    return "\n".join([
        r"\begin{table}[t]\centering\small",
        r"\caption{Fixed-final matched Full/Pooled shared-stage comparison. "
        r"Each task has 50 paired native trials at one training seed. "
        r"Differences are percentage points, with conservative simultaneous 95\% intervals. "
        r"The methods include their declared mechanism-specific supervision; "
        r"this does not replace the task-specialist benchmark.}",
        r"\label{tab:matched_final3000}",
        r"\begin{tabular}{lccc}\toprule",
        r"Task & Full & Pooled & Difference [95\% interval] \\ \midrule",
        *rows, r"\bottomrule\end{tabular}\end{table}", ""])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--campaign-root", type=Path, required=True)
    ap.add_argument("--controller-code", type=Path, required=True)
    ap.add_argument("--evaluation-code", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root, h, c, out = [p.resolve() for p in (args.campaign_root, args.controller_code, args.evaluation_code, args.output)]
    if out.exists() or any(out.is_relative_to(p) for p in (ROOT, h, c, root)):
        raise ValueError("Use a fresh evidence directory outside source and experiment roots")
    identity_reader = load("frozen_identity_reader", h/"scripts/nonreal_job.py")
    if identity_reader.source_identity(h)["sha256"] != CONTROLLER or identity_reader.source_identity(c)["sha256"] != CORE:
        raise ValueError("Frozen runtime sources differ")
    campaign = load("frozen_typed_campaign", h/"scripts/run_rebuttal_matched_campaign.py")
    M = campaign.M
    sys.path[:0] = [str(c/"src"), str(c)]
    evidence, rows, attempts_audit = {}, [], []
    def bind(path):
        evidence[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
    if not root.exists():
        report = summarize_fixed_pairs([])
        report["execution_status"] = "not_admitted_no_formal_clock_or_policy_result"
    else:
        master = read(root/"master.json"); bind(root/"master.json")
        if master["design_sha256"] != DESIGN or master["planned_pairs"] != 100:
            raise ValueError("Different registered study")
        ready = read(root/"verified_inputs.json"); bind(root/"verified_inputs.json")
        if sha(root/"verified_inputs.json") != master["verified_inputs_sha256"]:
            raise ValueError("Master input evidence changed")
        study = dict(design_sha256=DESIGN, evaluation_source_sha256=CORE, root_seed=3407,
                     checkpoint_update=3000,
                     full_checkpoint_sha256=ready["training"]["full"]["checkpoint_sha256"],
                     pooled_checkpoint_sha256=ready["training"]["pooled_memory_prior"]["checkpoint_sha256"])
        expected = {f"{t}-{i:02d}" for t in ("rb", "bt") for i in range(0,50,10)}
        present = {d.name for d in (root/"shards").glob("*") if d.is_dir()}
        if present-expected:
            raise ValueError("Unregistered shard directory")
        for key in sorted(present):
            d = root/"shards"/key
            if not (d/"plan.json").exists():
                continue
            plan = read(d/"plan.json"); plan_sha = sha(d/"plan.json"); bind(d/"plan.json")
            M.validate_plan(plan); M.verify_inputs(plan)
            prefix,start=key.split("-");start=int(start)
            if plan["task"]!={"rb":"rearrange_blocks","bt":"battery_try"}[prefix] or plan["case_ids"]!=list(range(start,start+10)):
                raise ValueError("Shard name differs from its fixed task/case range")
            if plan["evaluation_source_sha256"] != CORE or plan["controller_source_sha256"] != CONTROLLER:
                raise ValueError("Different role/evaluation source")
            if [plan["roles"][r]["checkpoint_sha256"] for r in ROLES] != [study["full_checkpoint_sha256"], study["pooled_checkpoint_sha256"]]:
                raise ValueError("Different final weights across shards")
            groups = {}
            for a in sorted((d/"campaign/attempts").glob("*")):
                if not a.is_dir():
                    continue
                meta = read(a/"attempt.json"); bind(a/"attempt.json")
                if meta["plan_sha256"] != plan_sha or meta["case_id"] not in plan["case_ids"] or meta["role"] not in ROLES:
                    raise ValueError("Foreign attempt")
                group = groups.setdefault((meta["case_id"], meta["role"]), [])
                audit = dict(attempt=str(a), case_id=meta["case_id"], role=meta["role"], task=plan["task"])
                if not (a/"exit.json").exists():
                    audit["status"] = "active_or_unknown_exit"
                    group.append((None, audit)); attempts_audit.append(audit); continue
                bind(a/"exit.json"); audit["exit"] = read(a/"exit.json")
                if not (a/"archive_receipt.json").exists():
                    audit["status"] = "known_exit_archive_pending"
                    group.append((None, audit)); attempts_audit.append(audit); continue
                receipt = verify_role_archive(a)
                bind(a/"archive_receipt.json"); audit["archive"] = receipt
                worker = a/"worker"
                if (worker/"result.json").exists():
                    result = campaign.verified_result(worker, plan, plan_sha, meta["case_id"], meta["role"])
                    bind(worker/"result.json")
                    if result["status"] in KNOWN:
                        typed=read(worker/"typed_policy_receipt.json");binding=plan["roles"][meta["role"]]
                        for name in ("model_class","checkpoint_schema","checkpoint_update","checkpoint_sha256"):
                            if typed[name]!=binding[name] or result[name]!=typed[name]:
                                raise ValueError("Actual typed model receipt differs from its final role")
                        if typed["common_contract"]!=result["common_contract"]:
                            raise ValueError("Typed information contract changed")
                        import numpy as np
                        generations=sorted((worker/"generations").glob("*/generation.json"))
                        if result["status"]=="complete_native" and len(generations)!=len(result["queries"]):
                            raise ValueError("Missing complete H32 generation artifacts")
                        for generation in generations:
                            g=read(generation)
                            if g.get("returned_action_missing") is True:
                                if result["status"]!="identifiable_policy_failure":raise ValueError("Missing action in a complete native role")
                                continue
                            action_file=generation.parent/"model_actions.npy"
                            if sha(action_file)!=g["action_file_sha256"]:raise ValueError("Action generation hash differs")
                            action=np.load(action_file,allow_pickle=False)
                            if list(action.shape)!=g["shape"] or bool(np.isfinite(action).all())!=g["all_finite"]:
                                raise ValueError("Action generation metadata differs")
                            if result["status"]=="complete_native" and (tuple(action.shape) not in ((32,14),(1,32,14)) or not g["all_finite"]):
                                raise ValueError("Incomplete/nonfinite H32 generation")
                        if result["status"]=="complete_native":
                            commands=[json.loads(line) for line in (worker/"physical_commands.jsonl").read_text().splitlines() if line.strip()]
                            n=result["executed_commands"]
                            if len(commands)!=2*n:raise ValueError("Incomplete physical command trace")
                            with np.load(worker/"qpos_trace.npz",allow_pickle=False) as trace:
                                for i in range(n):
                                    before,after=commands[2*i:2*i+2]
                                    if (before["kind"],after["kind"],before["index"],after["index"],before["take_action_cnt_before"],after["take_action_cnt_after"])!=("command_begin","command_end",i,i,i,i+1):
                                        raise ValueError("Native command identity/counter differs")
                                    if after["error"] is not None or not np.array_equal(np.asarray(before["command"],dtype=np.float32),trace["commands"][i]):
                                        raise ValueError("Native command evidence differs")
                        roster = read(plan["roster"])["entries"][meta["case_id"]]
                        M.require_initial_native_identity(roster, result["initial_identity"], result["initial_measured_physical"])
                        audit["status"] = "verified_known_native_outcome"
                        group.append((result, audit))
                    else:
                        audit["status"] = "archived_unknown_native_outcome"
                        group.append((None, audit))
                else:
                    audit["status"] = "archived_unknown_before_result"
                    group.append((None, audit))
                attempts_audit.append(audit)
            for index in plan["case_ids"]:
                role_values, role_records = {}, {}
                for role in ROLES:
                    history = groups.get((index,role), [])
                    known = [r for r,a in history if r is not None]
                    if len(known)>1 or len(history)>plan.get("max_attempts_per_role",2):
                        raise ValueError("Duplicate known outcomes or exceeded retry bound")
                    if known and any(a["status"] in ("active_or_unknown_exit","known_exit_archive_pending") for r,a in history):
                        raise ValueError("Unadjudicated invocation cannot be hidden by a later outcome")
                    if known:
                        role_records[role] = known[0]
                        role_values[role] = dict(status=known[0]["status"],success=known[0]["success"])
                    else:
                        role_values[role] = dict(status="pending_or_unknown",success=None)
                paired = len(role_records)==2
                if paired:
                    check=M.pair_receipts(role_records["full"],role_records["pooled_memory_prior"])
                    if check["status"]!="paired_native_known_outcomes":
                        raise ValueError("Actual native pair verification unavailable")
                rows.append(dict(task=plan["task"],episode_id=index,phase="formal",study_identity=study,
                                 plan_sha256=plan_sha,roles=role_values,paired_initial_and_noise_verified=paired))
        report = summarize_fixed_pairs(rows)
        report["execution_status"] = "all100_native_pairs_verified" if report["cohort_complete"] else "incomplete_fixed_cohort_descriptive_only"
    if identity_reader.source_identity(h)["sha256"] != CONTROLLER or identity_reader.source_identity(c)["sha256"] != CORE:
        raise ValueError("Immutable runtime changed during analysis")
    out.mkdir(parents=True)
    report.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempt_audit=attempts_audit,
                  publication_authorized=False)
    (out/"REPORT.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    (out/"INPUT_MANIFEST.json").write_text(json.dumps(evidence,indent=2)+"\n")
    with (out/"PAIRS.csv").open("w",newline="") as f:
        fields=["task","episode_id","full_status","full_success","pooled_status","pooled_success","paired_verified"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        bykey={(r["task"],r["episode_id"]):r for r in rows}
        for task in TASKS:
            for index in range(50):
                row=bykey.get((task,index),{});roles=row.get("roles",{})
                writer.writerow(dict(task=task,episode_id=index,
                    full_status=roles.get("full",{}).get("status","not_observed"),
                    full_success=roles.get("full",{}).get("success"),
                    pooled_status=roles.get("pooled_memory_prior",{}).get("status","not_observed"),
                    pooled_success=roles.get("pooled_memory_prior",{}).get("success"),
                    paired_verified=row.get("paired_initial_and_noise_verified",False)))
    if report["primary"] is not None:
        (out/"matched_final3000_table.tex").write_text(latex_table(report))
    print(json.dumps(dict(status=report["execution_status"],complete_pairs=sum(x["complete_pairs"] for x in report["tasks"].values()),
                          primary_table_available=report["primary"] is not None,output=str(out))))


if __name__ == "__main__":
    main()
