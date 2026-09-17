#!/usr/bin/env python3
"""Piper-only convenience entrypoint; shares the benchmark preparation stages."""
import _bootstrap  # noqa: F401
from fastwam.preprocessing.cli import main

if __name__ == "__main__":
    raise SystemExit(main(required_adapter="piper_teleop"))
