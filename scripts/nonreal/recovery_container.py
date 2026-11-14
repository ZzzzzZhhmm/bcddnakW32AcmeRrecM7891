"""Self-contained, lossless recovery containers; never modify input files.

Payload chunks are addressed by their SHA256. TAR payload boundaries are kept
so the same checkpoint bytes can be shared by its raw and TAR representations.
Every file, including original TAR headers and padding, is reconstructed exactly.
This module has no model, network, Git, or checkpoint-retention side effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import time
import zipfile


class Budget:
    def __init__(self, seconds: float, min_free_bytes: int = 0, volume: Path | None = None):
        if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                or not math.isfinite(seconds) or seconds <= 0
                or type(min_free_bytes) is not int or min_free_bytes < 0):
            raise ValueError("Invalid recovery resource bound")
        self.deadline = time.monotonic() + seconds
        self.min_free = min_free_bytes
        self.volume = volume
        self.last_disk_check = 0.0

    def check(self):
        now = time.monotonic()
        if now >= self.deadline:
            raise TimeoutError("Original recovery-container time bound expired")
        if self.volume is not None and now - self.last_disk_check >= 1:
            if shutil.disk_usage(self.volume).free < self.min_free:
                raise OSError("Recovery-container free-space guard rejected further work")
            self.last_disk_check = now


def file_sha256(path: Path, budget: Budget) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            budget.check()
            h.update(b)
    return h.hexdigest()


def _key(value: str) -> str:
    p = PurePosixPath(value)
    if (not value or p.is_absolute() or ".." in p.parts or "\\" in value
            or not re.fullmatch(r"[A-Za-z0-9_.\-/]+", value) or str(p) != value):
        raise ValueError("Recovery entry key must be a normalized relative path")
    return value


def _identity(path: Path):
    s = path.stat()
    return s.st_size, s.st_mtime_ns, s.st_ino


def _intervals(path: Path, size: int):
    boundaries = {0, size}
    if path.suffix.lower() == ".tar":
        with tarfile.open(path, "r:") as pack:
            for member in pack.getmembers():
                if member.isfile():
                    end = member.offset_data + member.size
                    if not 0 <= member.offset_data <= end <= size:
                        raise ValueError("Invalid original TAR member boundary")
                    boundaries.update((member.offset_data, end))
    points = sorted(boundaries)
    return list(zip(points, points[1:]))


def build_container(plan: dict, output: Path, budget: Budget) -> dict:
    budget.check()
    scope = Path(plan["source_scope"]).resolve(strict=True)
    rows = plan["files"]
    chunk_bytes = plan.get("chunk_bytes", 1024**2)
    if not rows or type(chunk_bytes) is not int or not 4096 <= chunk_bytes <= 8 * 1024**2:
        raise ValueError("Invalid recovery input roster or chunk size")
    prepared = []
    keys = set()
    for row in rows:
        name = _key(row["key"])
        if name in keys:
            raise ValueError("Duplicate recovery entry key")
        keys.add(name)
        original = Path(row["path"])
        path = original.resolve(strict=True)
        if original.is_symlink() or not path.is_relative_to(scope) or not path.is_file():
            raise ValueError("Recovery input escapes its declared source scope")
        identity = _identity(path)
        if list(identity) != row["stat"]:
            raise ValueError("Recovery input changed after registration")
        prepared.append((row, path, identity))
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists() or output.is_symlink():
        raise FileExistsError("Existing recovery output must be reconciled")
    if any(path == output.resolve() or path == partial.resolve() for _, path, _ in prepared):
        raise ValueError("Recovery output cannot replace an input")
    files, objects = {}, {}
    with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as pack:
        for row, path, identity in prepared:
            budget.check()
            if _identity(path) != identity:
                raise ValueError("Recovery input changed before reading")
            parts, digest, count = [], hashlib.sha256(), 0
            with path.open("rb") as f:
                for begin, end in _intervals(path, identity[0]):
                    f.seek(begin)
                    cursor = begin
                    while cursor < end:
                        budget.check()
                        b = f.read(min(chunk_bytes, end - cursor))
                        if not b:
                            raise EOFError("Original recovery input was truncated")
                        h = hashlib.sha256(b).hexdigest()
                        if h not in objects:
                            pack.writestr("objects/" + h, b)
                            objects[h] = len(b)
                        elif objects[h] != len(b):
                            raise ValueError("Recovery object identity collision")
                        parts.append(h)
                        digest.update(b)
                        count += len(b)
                        cursor += len(b)
            actual = digest.hexdigest()
            if count != identity[0] or _identity(path) != identity:
                raise ValueError("Recovery input changed during reading")
            if row.get("expected_sha256") and actual != row["expected_sha256"]:
                raise ValueError("Original recovery input SHA256 differs")
            files[row["key"]] = {"bytes": count, "sha256": actual, "parts": parts,
                                 "original_path": str(path), "original_stat": list(identity)}
        for _, path, identity in prepared:
            if _identity(path) != identity:
                raise ValueError("Recovery input changed before final sealing")
        manifest = {"schema": "warm.self-contained-recovery-container.v1",
                    "chunk_bytes": chunk_bytes, "objects": objects, "files": files,
                    "external_base_required": False}
        pack.writestr("manifest.json", json.dumps(manifest, sort_keys=True).encode())
    budget.check()
    # Atomic no-overwrite publication; the temporary link is not a second copy.
    os.link(partial, output)
    partial.unlink()
    return manifest


def _manifest(pack: zipfile.ZipFile) -> dict:
    names = pack.namelist()
    if len(set(names)) != len(names) or names.count("manifest.json") != 1:
        raise ValueError("Duplicate or missing recovery manifest")
    if pack.getinfo("manifest.json").file_size > 256 * 1024**2:
        raise ValueError("Recovery manifest exceeds its memory bound")
    m = json.loads(pack.read("manifest.json"))
    if m.get("schema") != "warm.self-contained-recovery-container.v1" or m.get("external_base_required") is not False:
        raise ValueError("Unsupported or externally dependent recovery container")
    for key in m["files"]:
        _key(key)
    expected = {"manifest.json"} | {"objects/" + h for h in m["objects"]}
    if set(names) != expected or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in m["objects"]):
        raise ValueError("Unexpected or malformed recovery objects")
    if type(m.get("chunk_bytes")) is not int or not 4096 <= m["chunk_bytes"] <= 8 * 1024**2:
        raise ValueError("Invalid recovery object size bound")
    for h, size in m["objects"].items():
        if type(size) is not int or not 0 < size <= m["chunk_bytes"] or pack.getinfo("objects/" + h).file_size != size:
            raise ValueError("Recovery object size differs from its bounded identity")
    for item in m["files"].values():
        if (type(item["bytes"]) is not int or item["bytes"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                or sum(m["objects"].get(h, -1) for h in item["parts"]) != item["bytes"]):
            raise ValueError("Invalid recovery file recipe")
    referenced = {h for item in m["files"].values() for h in item["parts"]}
    if referenced != set(m["objects"]):
        raise ValueError("Recovery file recipes and objects differ")
    return m


def _chunks(pack, manifest, entry, budget):
    item = manifest["files"][entry]
    for h in item["parts"]:
        budget.check()
        b = pack.read("objects/" + h)
        if len(b) != manifest["objects"][h] or hashlib.sha256(b).hexdigest() != h:
            raise ValueError("Recovery object SHA256 differs")
        yield b


def verify_container(path: Path, budget: Budget) -> dict:
    budget.check()
    with zipfile.ZipFile(path) as pack:
        m = _manifest(pack)
        for entry, item in m["files"].items():
            h, n = hashlib.sha256(), 0
            for b in _chunks(pack, m, entry, budget):
                h.update(b)
                n += len(b)
            if n != item["bytes"] or h.hexdigest() != item["sha256"]:
                raise ValueError("Reconstructed recovery file SHA256 differs")
    return {"files": len(m["files"]), "objects": len(m["objects"]),
            "logical_bytes": sum(v["bytes"] for v in m["files"].values()),
            "unique_uncompressed_bytes": sum(m["objects"].values()),
            "archive_sha256": file_sha256(path, budget), "all_files_exact": True,
            "external_base_required": False}


def extract_entry(path: Path, entry: str, output: Path, budget: Budget) -> dict:
    budget.check()
    _key(entry)
    output = output.absolute()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists() or output.is_symlink():
        raise FileExistsError("Recovery extraction never overwrites an existing file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as pack:
        m = _manifest(pack)
        item = m["files"][entry]
        h, n = hashlib.sha256(), 0
        with partial.open("xb") as f:
            for b in _chunks(pack, m, entry, budget):
                f.write(b)
                h.update(b)
                n += len(b)
            f.flush()
            os.fsync(f.fileno())
        if n != item["bytes"] or h.hexdigest() != item["sha256"]:
            raise ValueError("Extracted recovery file SHA256 differs")
    os.link(partial, output)
    partial.unlink()
    return {"entry": entry, "bytes": n, "sha256": h.hexdigest(), "output": str(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["build", "verify", "extract"])
    p.add_argument("--container", type=Path, required=True)
    p.add_argument("--plan", type=Path)
    p.add_argument("--entry")
    p.add_argument("--output", type=Path)
    p.add_argument("--seconds", type=float, default=7200)
    a = p.parse_args()
    budget = Budget(a.seconds)
    if a.action == "build":
        if a.plan is None: p.error("build requires --plan")
        m = build_container(json.loads(a.plan.read_text()), a.container, budget)
        result = {"files": len(m["files"]), "external_base_required": False}
    elif a.action == "verify": result = verify_container(a.container, budget)
    else:
        if a.entry is None or a.output is None: p.error("extract requires --entry and --output")
        result = extract_entry(a.container, a.entry, a.output, budget)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
