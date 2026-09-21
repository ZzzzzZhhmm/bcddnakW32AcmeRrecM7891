"""Small append-only evidence writer and immutable pre-rollout probe bundles."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def probe_query_id(*, experiment_id: str, task: str, bound_step_sha256: str) -> str:
    """Unique across tasks and interventions when episode counters are reused."""
    if not experiment_id or not task or not bound_step_sha256:
        raise ValueError("query identity requires experiment, task and bound step")
    return "query-" + hashlib.sha256(canonical({
        "experiment_id": experiment_id, "task": task, "bound_step_sha256": bound_step_sha256,
    })).hexdigest()


def append_record(path: Path, record: dict) -> None:
    """One writer per file. Flush every record so interruption is observable."""
    data = canonical(record) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def write_probe(root: Path, query_id: str, probe: dict, identity: dict) -> dict:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", query_id) is None:
        raise ValueError("unsafe query_id")
    arrays = {}
    for key, value in probe["arrays"].items():
        if hasattr(value, "detach"):
            value = value.detach().cpu()
            if value.is_floating_point():
                value = value.float()
            value = value.numpy()
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"probe array {key} is nonnumeric or nonfinite")
        arrays[key] = array
    if not arrays:
        raise ValueError("empty probe")
    directory = root / query_id
    directory.mkdir(parents=True, exist_ok=False)
    payload = directory / "proposal.npz"
    with payload.open("xb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    metadata = {k: v for k, v in probe.items() if k != "arrays"}
    metadata.update({"query_id": query_id, "identity": identity,
                     "array_sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
                     "capture_phase": "before_any_branch_truth",
                     "dtype": {k: str(v.dtype) for k, v in arrays.items()}})
    with (directory / "proposal.json").open("xb") as f:
        f.write(canonical(metadata))
        f.flush()
        os.fsync(f.fileno())
    return {"path": str(directory), "array_sha256": metadata["array_sha256"],
            "metadata_sha256": hashlib.sha256(canonical(metadata)).hexdigest()}
