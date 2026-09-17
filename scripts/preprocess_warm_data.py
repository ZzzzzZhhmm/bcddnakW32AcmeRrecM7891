#!/usr/bin/env python3
"""Select a benchmark/real adapter and prepare WARM artifacts without actuation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastwam.preprocessing.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
