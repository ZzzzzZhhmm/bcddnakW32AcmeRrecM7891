"""Check the raw handoff format; does not qualify control or train a model."""
import argparse
import json
import _bootstrap  # noqa: F401
from fastwam.real.episodes import audit_episode, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    report = audit_episode(args.episode, args.allow_synthetic)
    if args.report:
        write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
