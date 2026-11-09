"""Verify immutable official role trials before admitting resumable evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROLES = ("full", "candidate_null")
BINDINGS = ("task", "phase", "checkpoint_sha256", "plan_sha256", "search_condition")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_role(directory, row, role, plan, seeds):
    """A completed flag alone never admits a truncated or differently bound run."""
    import numpy as np

    directory = Path(directory)
    identity = read_json(directory / "identity.json")
    simulator = read_json(directory / "runtime_source_manifest.json")
    if simulator["sha256"] != plan["runtime_source_sha256"]:
        raise ValueError("official simulator source differs from the admitted roster")
    expected = dict(task=plan["task"], phase="formal", checkpoint_sha256=plan["checkpoint_sha256"],
                    plan_sha256=plan["evidence_plan_sha256"], search_condition="full_bank")
    if any(identity.get(k) != v or row.get(k) != v for k, v in expected.items()):
        raise ValueError("role trial differs from fixed scientific bindings")
    if any(identity.get(k) != plan["core_source_sha256"] for k in
           ("core_source_sha256", "collector_source_sha256")):
        raise ValueError("role was not executed by the qualified immutable core")
    index = row["episode_id"]
    if type(index) is not int or index not in seeds or row["seed"] != seeds[index]:
        raise ValueError("role does not match the outcome-independent reset roster")
    if role not in ROLES or role not in row["roles"]:
        raise ValueError("unknown or missing role")
    result = row["roles"][role]
    if result.get("status") != "complete" or type(result.get("success")) is not int or result["success"] not in (0, 1):
        raise ValueError("only complete official roles have known success labels")
    case = directory / f"case-{index:04d}-seed-{seeds[index]}" / role
    if read_json(case / "result.json") != result:
        raise ValueError("role row and durable result disagree")
    if any(result.get(k) != row[k] for k in ("task", "phase", "episode_id", "seed")) or result.get("role") != role:
        raise ValueError("result carries a different role or episode")
    count = result["executed_commands"]
    if type(count) is not int or not 0 < count <= plan["official_step_limit"] or (not result["success"] and count != plan["official_step_limit"]):
        raise ValueError("failed role did not reach the original official limit")
    trace_path = case / "qpos_trace.npz"
    with np.load(trace_path, allow_pickle=False) as trace:
        for key in ("commands", "before", "after"):
            if trace[key].shape != (count, 14) or not np.isfinite(trace[key]).all():
                raise ValueError("truncated or nonfinite official command/state trace")
        first_qpos = trace["before"][0].copy()
    initial_path = case / "initial_identity.json"
    initial = read_json(initial_path)
    from fastwam.memory.manifest import sha256_array
    with np.load(case / "initial_observation.npz", allow_pickle=False) as obs:
        if not np.array_equal(obs["native_qpos"], first_qpos):
            raise ValueError("initial observation differs from the executed trace")
        if {k: sha256_array(obs[k]) for k in obs.files} != initial["observation_arrays"]:
            raise ValueError("sealed initial camera/qpos hashes differ from saved arrays")
    receipt = dict(row=row, identity=identity, role=role, initial_identity=initial,
                   worker_evidence=str(case), worker_root=str(directory),
                   files={str(f): file_sha(f) for f in (case / "result.json", trace_path, initial_path, case / "initial_observation.npz")})
    return receipt


def read_roles(directory, plan, seeds):
    directory = Path(directory)
    trial_path = directory / "role_trials.jsonl"
    if not trial_path.exists():
        trial_path = directory / "pairs.jsonl"
    if not trial_path.exists():
        return []
    text=trial_path.read_text()
    # Append records commit with a newline. Preserve a killed writer's partial
    # tail on disk, while admitting only its fully committed predecessor rows.
    if text and not text.endswith('\n'):text=text[:text.rfind('\n')+1]
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    result = []
    for row in rows:
        for role, value in row["roles"].items():
            if value.get("status") == "complete":
                result.append(verify_role(directory, row, role, plan, seeds))
    return result


def unique_roles(records):
    result = {}
    for record in records:
        key = record["row"]["episode_id"], record["role"]
        if key in result:
            raise ValueError("duplicate complete role: adjudicate attempts explicitly, never select by score")
        result[key] = record
    return result


def paired_rows(records, seeds, indices):
    accepted = unique_roles(records)
    rows = []
    for index in indices:
        parts = [accepted.get((index, role)) for role in ROLES]
        if all(parts):
            if parts[0]["initial_identity"] != parts[1]["initial_identity"]:
                raise ValueError("fresh-reset camera/qpos/task/instruction identities differ")
            for key in ("core_source_sha256", "collector_source_sha256", *BINDINGS):
                if parts[0]["identity"][key] != parts[1]["identity"][key]:
                    raise ValueError("paired role identities differ")
        roles = {role: part["row"]["roles"][role] if part else None for role, part in zip(ROLES, parts)}
        complete = all(parts)
        rows.append(dict(episode_id=index, seed=seeds[index], status="complete" if complete else "pending",
                         roles=roles, full_minus_null=roles["full"]["success"] - roles["candidate_null"]["success"] if complete else None,
                         evidence=[part["worker_evidence"] for part in parts if part]))
    return rows


def next_role(accepted, seeds, indices):
    for index in indices:
        order = ROLES if index % 2 == 0 else tuple(reversed(ROLES))
        for role in order:
            if (index, role) not in accepted:
                return index, role, seeds[index]
    return None


def summarize(rows):
    complete = [r for r in rows if r["status"] == "complete"]
    return dict(requested_pairs=len(rows), complete_pairs=len(complete), pending_pairs=len(rows)-len(complete),
                full_successes=sum(r["roles"]["full"]["success"] for r in complete),
                null_successes=sum(r["roles"]["candidate_null"]["success"] for r in complete),
                full_only=sum(r["roles"]["full"]["success"] == 1 and r["roles"]["candidate_null"]["success"] == 0 for r in complete),
                null_only=sum(r["roles"]["full"]["success"] == 0 and r["roles"]["candidate_null"]["success"] == 1 for r in complete),
                scope="Descriptive counts for the fixed cohort; pending and interrupted trials are not failures. No efficacy or noninferiority conclusion from partial counts.")


def archive_verified_role(worker):
    """Keep a complete immutable worker snapshot and hashes outside the source."""
    import io
    import tarfile
    worker=Path(worker);target=worker.parent/'verified_worker_evidence.tar.gz'
    receipt_path=worker.parent/'archive_receipt.json'
    if target.exists():
        if not receipt_path.exists():
            with tarfile.open(target) as tar:
                manifest=json.load(tar.extractfile('archive_manifest.json'))
                for name,item in manifest.items():
                    data=tar.extractfile(name).read()
                    if len(data)!=item['bytes'] or hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('existing role archive differs')
                if len(tar.getmembers())!=len(manifest)+1:raise ValueError('unexpected archive members')
            receipt=dict(path=str(target),files=len(manifest),bytes=target.stat().st_size,sha256=file_sha(target))
            receipt_path.write_text(json.dumps(receipt,indent=2))
        receipt=read_json(receipt_path)
        if file_sha(target)!=receipt['sha256']:raise ValueError('existing role archive differs')
        return receipt
    contents={str(f.relative_to(worker)):f.read_bytes() for f in sorted(worker.rglob('*')) if f.is_file()}
    manifest={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in contents.items()}
    import uuid
    temporary=target.with_suffix(target.suffix+'.partial-'+uuid.uuid4().hex)
    with tarfile.open(temporary,'x:gz') as tar:
        for name,data in {**contents,'archive_manifest.json':json.dumps(manifest,indent=2).encode()}.items():
            item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
    temporary.replace(target)
    receipt=dict(path=str(target),files=len(contents),bytes=target.stat().st_size,sha256=file_sha(target))
    receipt_path.write_text(json.dumps(receipt,indent=2))
    return receipt
