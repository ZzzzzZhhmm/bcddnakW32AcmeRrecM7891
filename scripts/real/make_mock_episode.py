"""Generate conspicuously synthetic fixtures. No cameras, CAN, GPU or robot I/O."""
import argparse
import _bootstrap  # noqa: F401
from fastwam.real.synthetic import make_episode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    args = parser.parse_args()
    make_episode(args.output)
    print("Created SYNTHETIC fixture; not admissible as real demonstrations.")
