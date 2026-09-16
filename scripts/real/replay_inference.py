"""Offline recorded-observation inference. NEVER sends a robot command."""
import argparse
import _bootstrap  # noqa: F401
from fastwam.real.shadow import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefix", type=int, default=4)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--backend", help="trusted Python module:factory supplied with a release")
    parser.add_argument("--release", help="release descriptor supplied by the model team")
    args = parser.parse_args()
    result = replay(args.episode, args.output, mock=args.mock, backend_spec=args.backend,
                    release=args.release, prefix=args.prefix)
    print(f"Saved {result['calls']} offline predictions; qualification={result['qualification']}")


if __name__ == "__main__":
    main()
