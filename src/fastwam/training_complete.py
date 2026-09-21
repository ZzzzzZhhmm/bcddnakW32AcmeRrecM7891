"""Side-effect-free training-completion marker for ACP fail-closed checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

TRAINING_COMPLETE_FILENAME = "training_complete.json"
TRAINING_COMPLETE_SCHEMA = "warm.training-complete"


def training_complete_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / TRAINING_COMPLETE_FILENAME


def is_training_complete(output_dir: str | Path) -> bool:
    marker = training_complete_path(output_dir)
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("schema") == TRAINING_COMPLETE_SCHEMA


def latest_training_state(output_dir: str | Path) -> Path | None:
    """Return the highest ``checkpoints/state/step_*`` directory, if any."""

    state_root = Path(output_dir) / "checkpoints" / "state"
    if not state_root.is_dir():
        return None
    steps: list[tuple[int, Path]] = []
    for child in state_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("step_"):
            continue
        suffix = name[len("step_") :]
        if not suffix.isdigit():
            continue
        steps.append((int(suffix), child))
    if not steps:
        return None
    steps.sort(key=lambda item: item[0])
    return steps[-1][1]


def write_training_complete_marker(
    output_dir: str | Path,
    *,
    step: int,
    weights_path: str,
    state_path: str,
    reason: str,
) -> Path:
    """Atomically publish the only artifact ACP treats as a finished run."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = training_complete_path(output)
    payload = {
        "reason": str(reason),
        "schema": TRAINING_COMPLETE_SCHEMA,
        "state_path": str(state_path),
        "step": int(step),
        "version": 1,
        "weights_path": str(weights_path),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = output / f".{TRAINING_COMPLETE_FILENAME}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(text, encoding="utf-8")
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
