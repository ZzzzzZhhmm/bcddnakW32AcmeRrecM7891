#!/usr/bin/env python3
"""Install WARM's Open3D import guard into an evaluation venv overlay."""

from __future__ import annotations

import argparse
import shutil
import sysconfig
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    source = args.source.resolve()
    if not (source / "__init__.py").is_file():
        raise SystemExit(f"Open3D guard source is incomplete: {source}")

    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    destination = purelib / "open3d"
    # Only mutate the active evaluation overlay.  Never touch the inherited
    # WARM base environment selected through --system-site-packages.
    if destination.parent != purelib:
        raise SystemExit(f"unsafe Open3D guard destination: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    print(f"open3d_rgb_guard={destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
