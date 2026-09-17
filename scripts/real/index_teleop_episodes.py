#!/usr/bin/env python3
"""Index existing, explicitly split episode logs; never invent a data split."""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from fastwam.real.episodes import load_episode
from fastwam.preprocessing.contracts import PreparationError, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-outcome", choices=("success", "failure", "aborted"), action="append")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("--root must exist")
    outcomes = set(args.include_outcome or ["success"])
    episodes, excluded, sessions, ids = [], [], {}, set()
    for path in sorted(root.rglob("episode.json")):
        meta, _, _, outcome = load_episode(path.parent, allow_synthetic=args.allow_synthetic)
        episode_id = meta["episode_id"]
        if episode_id in ids:
            raise PreparationError(f"Duplicate raw episode id: {episode_id}")
        ids.add(episode_id)
        split = meta["split"]
        if sessions.setdefault(meta["session_id"], split) != split:
            raise PreparationError("One session crosses splits; fix the collection manifest before indexing")
        entry = {"id": episode_id, "path": path.parent.relative_to(root).as_posix(), "split": split}
        if outcome["status"] in outcomes:
            episodes.append(entry)
        else:
            excluded.append({**entry, "reason": "outcome=" + outcome["status"]})
    if not episodes:
        raise PreparationError("No matching complete episodes")
    write_json(args.output, {"schema": "warm.source-episodes.v1", "episodes": episodes,
                             "excluded": excluded, "split_policy": "preserve_raw_episode_metadata",
                             "synthetic_allowed": args.allow_synthetic})
    print(f"Indexed {len(episodes)} episodes; explicitly excluded {len(excluded)}. Saved {args.output}")


if __name__ == "__main__":
    main()
