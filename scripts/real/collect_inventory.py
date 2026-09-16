"""Collect read-only deployment facts; never import a robot driver or send commands.

This is an inventory, not a motion-safety or model-inference qualification.
Run separately on the robot client and the intended inference machine.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def inspect_command(argv: list[str]) -> dict:
    executable = shutil.which(argv[0])
    if executable is None:
        return {"available": False, "reason": "executable not found"}
    try:
        result = subprocess.run(
            [executable, *argv[1:]],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
            shell=False,
        )
        return {
            "available": True,
            "returncode": result.returncode,
            "stdout": result.stdout[-30000:],
            "stderr": result.stderr[-4000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error": str(exc)}


def inspect_artifact(spec: str) -> dict:
    role, separator, raw_path = spec.partition("=")
    if not separator or not role.strip() or not raw_path.strip():
        raise ValueError("--artifact must have the form ROLE=PATH")
    path = Path(raw_path).expanduser().resolve()
    entry = {"role": role, "path": str(path), "exists": path.exists()}
    if path.is_file():
        entry.update(kind="file", size_bytes=path.stat().st_size)
    elif path.is_dir():
        entry.update(kind="directory")
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", action="append", default=[], metavar="ROLE=PATH")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new filename")
    try:
        artifacts = [inspect_artifact(item) for item in args.artifact]
    except ValueError as exc:
        parser.error(str(exc))

    packages = {}
    for package in (
        "torch", "torchvision", "transformers", "deepspeed", "numpy",
        "piper_sdk", "python-can", "opencv-python", "opencv-python-headless",
    ):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None

    commands = {
        "gpu": ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits"],
    }
    if platform.system() == "Linux":
        commands.update(
            can_interfaces=["ip", "-details", "-statistics", "link", "show", "type", "can"],
            video_devices=["v4l2-ctl", "--list-devices"],
            usb_devices=["lsusb"],
        )
    disk = shutil.disk_usage(Path.cwd())
    payload = {
        "schema": "warm.deployment.readonly-inventory.v1",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.system(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "cwd_free_disk_bytes": disk.free,
        "packages": packages,
        "read_only_commands": {key: inspect_command(argv) for key, argv in commands.items()},
        "artifacts": artifacts,
        "checkpoint_deserialized": False,
        "robot_driver_imported": False,
        "robot_commands_sent": False,
        "qualification": "inventory_only_not_permission_to_move",
        "still_requires_measurement": [
            "exact arm model and firmware; firmware-matched DH and URDF",
            "CAN-to-left/right-arm identity and actual fresh feedback",
            "camera serials, exposure, calibration, orientation, timestamps",
            "robot limits, tool/payload, collision scene, independent emergency stop",
            "checkpoint/config/statistics/bank compatibility and real inference latency",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Saved read-only inventory: {args.output.resolve()}")
    print("No robot driver was imported; no robot command was sent.")


if __name__ == "__main__":
    main()
