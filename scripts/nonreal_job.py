#!/usr/bin/env python3
"""Logged, bounded ACP command runner. No Git commands or dependency installs.

Specs contain argument vectors (no shell expansion), expected input hashes, and
explicit qualification gates. A dry run validates exactly the same gates as an
actual run. Logs and run status survive ordinary errors and scheduler SIGTERM.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fastwam.research.evidence import append_record, canonical


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_identity(root: Path) -> dict:
    records = {}
    for directory in ("src", "scripts", "configs", "experiments", "tests"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".sh", ".json", ".yaml", ".yml"}:
                records[path.relative_to(root).as_posix()] = file_hash(path)
    return {"files": records, "sha256": hashlib.sha256(canonical(records)).hexdigest()}


def prepare(spec: dict, *, root: Path, python: str) -> list[str]:
    if spec.get("schema") != "warm.nonreal.job.v1":
        raise ValueError("unsupported job schema")
    seconds = spec.get("max_seconds")
    if type(seconds) is not int or not 1 <= seconds <= 23 * 3600:
        raise ValueError("max_seconds must be 1..82800; reserve time below ACP's 24h limit")
    if spec.get("readiness") != "ready":
        raise ValueError("job is not ready: " + str(spec.get("blocking_reasons")))
    argv = spec.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
        raise ValueError("argv must be a nonempty list of strings")
    # Replace only our two literal placeholders; preserve JSON/Python braces.
    command = [x.replace("{code}", str(root)).replace("{python}", python) for x in argv]
    if not command[0] or any("\x00" in x for x in command):
        raise ValueError("invalid argv")
    for record in spec.get("inputs", []):
        path = Path(record["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"required input missing: {path}")
        expected = record.get("sha256")
        if not expected or file_hash(path) != expected:
            raise ValueError(f"input identity mismatch: {path}")
    for gate in spec.get("qualification", []):
        payload = json.loads(Path(gate["path"]).read_text(encoding="utf-8"))
        if file_hash(Path(gate["path"])) != gate["sha256"]:
            raise ValueError("qualification file changed")
        if payload.get("status") != "qualified":
            raise ValueError("required qualification has not passed")
    return command


def run(spec: dict, command: list[str], output: Path, *, source: dict) -> int:
    output.mkdir(parents=True, exist_ok=False)
    (output / "job.json").write_bytes(canonical(spec))
    (output / "source_manifest.json").write_bytes(canonical(source))
    events = output / "events.jsonl"
    started = time.time()
    manifest = {"schema": "warm.nonreal.run.v1", "start_utc": datetime.now(timezone.utc).isoformat(),
                "argv": command, "cwd": str(ROOT), "python": sys.executable,
                "source_sha256": source["sha256"], "spec_sha256": hashlib.sha256(canonical(spec)).hexdigest(),
                "evidence_type": spec.get("evidence_type"), "claim": spec.get("claim"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "status": "running", "exit_code": None}
    (output / "run_manifest.json").write_bytes(canonical(manifest))
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled",
               DIFFSYNTH_SKIP_DOWNLOAD="true", TOKENIZERS_PARALLELISM="false")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    child = None
    reason = None
    previous_handlers = {}

    def terminate(signum, _frame):
        nonlocal reason
        reason = "scheduler_signal_" + str(signum)
        if child is not None and child.poll() is None:
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, terminate)
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, start_new_session=os.name == "posix")

        def copy_log():
            try:
                with (output / "console.log").open("wb", buffering=0) as log:
                    while True:
                        chunk = child.stdout.read1(65536)
                        if not chunk:
                            break
                        log.write(chunk)
                        try:
                            sys.stdout.buffer.write(chunk)
                            sys.stdout.buffer.flush()
                        except (BrokenPipeError, OSError):
                            # A disconnected UI must not interrupt durable logs.
                            pass
            except Exception as exc:
                logging_errors.append(f"{type(exc).__name__}: {exc}")
                terminate(signal.SIGTERM, None)
        logging_errors = []
        copier = threading.Thread(target=copy_log, daemon=True)
        copier.start()
        deadline = time.monotonic() + spec["max_seconds"]
        last_heartbeat = 0.0
        terminating_at = None
        while child.poll() is None:
            now = time.monotonic()
            if now - last_heartbeat >= 60:
                append_record(events, {"kind": "heartbeat", "elapsed_s": time.time() - started, "pid": child.pid})
                last_heartbeat = now
            if now >= deadline and reason is None:
                terminate(signal.SIGTERM, None)
                reason = "wall_time_limit"
            if reason is not None:
                terminating_at = terminating_at or now
                if now - terminating_at > 120:
                    if os.name == "posix":
                        os.killpg(child.pid, signal.SIGKILL)
                    else:
                        child.kill()
            time.sleep(0.2)
        copier.join(timeout=10)
        code = child.returncode
        if logging_errors or copier.is_alive():
            reason = "logging_failed: " + str(logging_errors)
            code = code or 4
        after = source_identity(ROOT)["sha256"]
        if after != source["sha256"]:
            reason = "source_changed_during_run"
            code = code or 3
        manifest.update(exit_code=code, status="complete" if code == 0 and reason is None else "failed",
                        reason=reason, elapsed_s=time.time() - started, source_sha256_after=after)
        return code if code else (3 if reason else 0)
    except BaseException as exc:
        manifest.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        if child is not None and child.poll() is None:
            terminate(signal.SIGTERM, None)
        raise
    finally:
        manifest["end_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "run_manifest.json").write_bytes(canonical(manifest))
        append_record(events, {"kind": "exit", **manifest})
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    try:
        args.output_root.resolve().relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("runtime outputs must be outside the source checkout")
    try:
        command = prepare(spec, root=ROOT, python=sys.executable)
    except Exception as exc:
        args.output_root.mkdir(parents=True, exist_ok=True)
        append_record(args.output_root / "preflight.jsonl", {
            "kind": "preflight_blocked", "utc": datetime.now(timezone.utc).isoformat(),
            "spec": str(args.spec.resolve()), "spec_sha256": hashlib.sha256(canonical(spec)).hexdigest(),
            "reason": f"{type(exc).__name__}: {exc}", "gpu_job_started": False,
        })
        print(f"PREFLIGHT_BLOCKED: {exc}", file=sys.stderr)
        return 2
    source = source_identity(ROOT)
    if args.dry_run:
        print(json.dumps({"status": "ready", "argv": command, "source_sha256": source["sha256"]}))
        return 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output_root / run_id
    print("nonreal_run_output=" + str(output), flush=True)
    return run(spec, command, output, source=source)


if __name__ == "__main__":
    raise SystemExit(main())
