"""Allow handoff CLIs without installing the CUDA training dependencies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
