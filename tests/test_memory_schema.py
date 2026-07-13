from __future__ import annotations

import json

import numpy as np
import pytest

from fastwam.memory import ArraySpec, EventBankManifest, EventId, ManifestError
from fastwam.memory.manifest import sha256_array, sha256_file, sha256_path_tree


def _digest() -> str:
    return "a" * 64


def test_event_id_is_strict_and_episode_key_omits_only_start_frame() -> None:
    first = EventId("libero", np.int64(2), np.int64(7), np.int64(11))
    second = EventId("libero", 2, 7, 29)

    assert first.to_dict() == {
        "dataset_id": "libero",
        "dataset_index": 2,
        "episode_index": 7,
        "start_frame": 11,
    }
    assert first.episode_key == second.episode_key == ("libero", 2, 7)
    assert EventId.from_dict(first.to_dict()) == first


@pytest.mark.parametrize(
    "args,exception",
    [
        (("", 0, 0, 0), ValueError),
        ((" libero", 0, 0, 0), ValueError),
        (("libero", True, 0, 0), TypeError),
        (("libero", -1, 0, 0), ValueError),
        (("libero", 0, 0, 1.5), TypeError),
    ],
)
def test_event_id_rejects_ambiguous_or_unsafe_values(args, exception) -> None:
    with pytest.raises(exception):
        EventId(*args)


def test_manifest_has_required_contract_and_round_trips_json(tmp_path) -> None:
    array = np.ascontiguousarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    manifest = EventBankManifest(
        action_normalizer={"type": "quantile", "stats_sha256": "1" * 64},
        encoder={"id": "dinov2", "revision": "pinned"},
        camera_layout={"views": ["external", "wrist"], "effect_view": "external"},
        provenance={"catalog_sha256": "2" * 64, "start_mode": "uniform"},
        arrays={"context_key": ArraySpec.from_array(array)},
        content_hashes={
            "events.npz": _digest(),
            "array:context_key": sha256_array(array),
        },
        num_events=2,
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert raw["schema"] == "warm.event-bank"
    assert raw["version"] == 2
    assert raw["action_normalizer"]["type"] == "quantile"
    assert raw["encoder"]["revision"] == "pinned"
    assert raw["camera_layout"]["effect_view"] == "external"
    assert "events.npz" in raw["content_hashes"]
    assert EventBankManifest.read(path).to_dict() == manifest.to_dict()


def test_manifest_rejects_missing_per_array_hash() -> None:
    with pytest.raises(ManifestError, match="array:context_key"):
        EventBankManifest(
            action_normalizer={},
            encoder={},
            camera_layout={},
            provenance={},
            arrays={"context_key": ArraySpec("<f4", (1, 2))},
            content_hashes={"events.npz": _digest()},
            num_events=1,
        )


def test_path_tree_hash_is_deterministic_and_binds_names_sizes_and_bytes(
    tmp_path,
) -> None:
    root = tmp_path / "checkpoint"
    (root / "nested").mkdir(parents=True)
    first = root / "a.bin"
    second = root / "nested" / "b.bin"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")

    digest, count = sha256_path_tree(root)
    assert count == 2
    assert sha256_path_tree(root) == (digest, count)
    # A single file keeps the long-standing file digest identity; directory
    # trees use their own domain-separated recipe.
    assert sha256_path_tree(first) == (sha256_file(first), 1)

    second.write_bytes(b"BETA")
    assert sha256_path_tree(root)[0] != digest


def test_path_tree_hash_rejects_missing_and_empty_trees(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_path_tree(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ManifestError, match="empty"):
        sha256_path_tree(empty)
