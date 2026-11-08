#!/usr/bin/env python3
"""Write process-local Vulkan/EGL configuration for extracted matching libraries.

No downloads, driver installation, system configuration, device changes or
global environment edits. Rendering must still pass the separate runtime check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def configure(driver_root: Path, output: Path, *, driver_version: str):
    driver_root = driver_root.resolve(strict=True)
    output = output.resolve()
    glx = driver_root / f"libGLX_nvidia.so.{driver_version}"
    egl = driver_root / f"libEGL_nvidia.so.{driver_version}"
    if not glx.is_file() or not egl.is_file():
        raise ValueError("extracted graphics libraries must match the live driver version")
    original = json.loads((driver_root / "nvidia_icd.json").read_text())
    original["ICD"]["library_path"] = str(glx)
    vendor = {"file_format_version": "1.0.0", "ICD": {"library_path": str(egl)}}
    output.mkdir(parents=True, exist_ok=True)
    payloads = {
        "vulkan_icd.json": json.dumps(original, indent=2) + "\n",
        "egl_vendor.json": json.dumps(vendor, indent=2) + "\n",
        "graphics_env.sh": "\n".join([
            "# Process-local libraries only; do not run a driver installer.",
            f"export LD_LIBRARY_PATH={shlex.quote(str(driver_root))}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}",
            f"export VK_ICD_FILENAMES={shlex.quote(str(output / 'vulkan_icd.json'))}",
            f"export __EGL_VENDOR_LIBRARY_FILENAMES={shlex.quote(str(output / 'egl_vendor.json'))}", ""]),
    }
    # Validate every existing file before creating any new configuration file.
    for name, text in payloads.items():
        path = output / name
        if path.exists() and path.read_text() != text:
            raise ValueError(f"refusing to overwrite a different configuration: {name}")
    for name, text in payloads.items():
        if not (output / name).exists():
            (output / name).write_text(text, encoding="utf-8", newline="\n")
    return {"driver_version": driver_version, "driver_root": str(driver_root),
            "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (glx, egl)},
            "environment_script": str(output / "graphics_env.sh"),
            "qualification": "not yet tested; run the unchanged simulator runtime check"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    versions = subprocess.check_output([
        "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True)
    versions = {v.strip() for v in versions.splitlines() if v.strip()}
    if len(versions) != 1:
        raise ValueError("one consistent live driver version required")
    report = configure(args.driver_root, args.output, driver_version=versions.pop())
    path = args.output / "graphics_manifest.json"
    if not path.exists():
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
