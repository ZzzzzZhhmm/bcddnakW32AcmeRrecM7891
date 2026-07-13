from __future__ import annotations

import json

import numpy as np
import pytest

from fastwam.memory import EventBank, EventBankError, EventId, IntegrityError, ManifestError


def _bank() -> EventBank:
    event_ids = [
        EventId("libero", 0, 4, 0),
        EventId("libero", 0, 4, 16),
        EventId("libero", 0, 5, 0),
        EventId("libero", 1, 4, 0),
    ]
    # Rows 0/1 are the same source episode; row 2 is the best legal neighbor.
    context_keys = np.array(
        [[1.0, 0.0], [0.99, 0.01], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32
    )
    actions_base = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    # Exercise canonical C-contiguous storage from a strided input view.
    actions = actions_base[:, :, ::-1]
    effects = np.arange(12, dtype=np.float32).reshape(4, 3)
    valid = np.array([[True, False], [True, True], [False, True], [True, True]])
    source_hashes = np.stack(
        [
            np.frombuffer(bytes.fromhex(value * 64), dtype=np.uint8)
            for value in ("1", "1", "2", "3")
        ]
    )
    feature_hashes = np.stack(
        [
            np.frombuffer(bytes.fromhex(value * 64), dtype=np.uint8)
            for value in ("a", "a", "b", "b")
        ]
    )
    return EventBank.from_arrays(
        event_ids,
        context_keys,
        model_space_action=actions,
        effect_tokens=effects,
        timing_valid=valid,
        source_episode_sha256=source_hashes,
        feature_episode_sha256=feature_hashes,
    )


def _metadata() -> dict[str, dict[str, object]]:
    return {
        "action_normalizer": {"type": "quantile", "revision": "libero-v1"},
        "encoder": {"id": "dinov2-vitl14", "revision": "deadbeef"},
        "camera_layout": {
            "views": ["agentview", "robot0_eye_in_hand"],
            "effect_view": "agentview",
        },
        "provenance": {
            "catalog_sha256": "a" * 64,
            "feature_collection_sha256": "b" * 64,
            "mining": {"action_horizon": 3, "start_mode": "uniform"},
        },
    }


def test_exact_cosine_search_excludes_the_entire_current_episode() -> None:
    bank = _bank()

    unfiltered = bank.search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)
    filtered = bank.search(
        np.array([1.0, 0.0], dtype=np.float32),
        top_k=3,
        exclude_episode=EventId("libero", 0, 4, 999),
    )

    assert [result.index for result in unfiltered[:2]] == [0, 1]
    assert [result.index for result in filtered] == [2, 3]
    assert all(result.event_id.episode_key != ("libero", 0, 4) for result in filtered)
    assert filtered[0].score == pytest.approx(0.8 / np.sqrt(0.8**2 + 0.2**2))


def test_event_bank_round_trip_preserves_ids_payloads_and_contract(tmp_path) -> None:
    bank = _bank()
    metadata = _metadata()
    manifest = bank.save(tmp_path, **metadata)
    restored = EventBank.load(
        tmp_path,
        expected_action_normalizer=metadata["action_normalizer"],
        expected_encoder=metadata["encoder"],
        expected_camera_layout=metadata["camera_layout"],
        expected_provenance=metadata["provenance"],
    )

    assert restored.event_ids == bank.event_ids
    np.testing.assert_array_equal(restored.context_keys, bank.context_keys)
    for name, expected in bank.payloads.items():
        np.testing.assert_array_equal(restored.payload(name), expected)
        assert restored.payload(name).flags.c_contiguous
        assert not restored.payload(name).flags.writeable
    assert restored.manifest == manifest
    assert manifest.content_hashes["events.npz"]
    assert set(manifest.arrays) == {
        "_event_dataset_id_utf8",
        "_event_dataset_id_offsets",
        "_event_dataset_index",
        "_event_episode_index",
        "_event_start_frame",
        "context_key",
        "effect_tokens",
        "feature_episode_sha256",
        "model_space_action",
        "source_episode_sha256",
        "timing_valid",
    }
    with pytest.raises(FileExistsError):
        bank.save(tmp_path, **metadata)


def test_load_rejects_payload_file_hash_tampering(tmp_path) -> None:
    _bank().save(tmp_path, **_metadata())
    with (tmp_path / "events.npz").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(IntegrityError, match="SHA-256 mismatch"):
        EventBank.load(tmp_path)


@pytest.mark.parametrize(
    "field,replacement,match",
    [
        ("dtype", "<f8", "dtype mismatch"),
        ("shape", [4, 3], "shape mismatch"),
    ],
)
def test_load_rejects_manifest_dtype_or_shape_mismatch(
    tmp_path, field: str, replacement, match: str
) -> None:
    _bank().save(tmp_path, **_metadata())
    manifest_path = tmp_path / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["arrays"]["context_key"][field] = replacement
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError, match=match):
        EventBank.load(tmp_path)


def test_search_rejects_invalid_query_and_duplicate_event_ids() -> None:
    bank = _bank()
    with pytest.raises(EventBankError, match="shape"):
        bank.search(np.ones(3, dtype=np.float32))
    with pytest.raises(EventBankError, match="non-zero"):
        bank.search(np.zeros(2, dtype=np.float32))

    event_id = EventId("libero", 0, 0, 0)
    with pytest.raises(EventBankError, match="unique"):
        EventBank(
            [event_id, event_id],
            np.ones((2, 1), dtype=np.float32),
        )


def test_search_can_exclude_duplicate_source_content_across_episode_ids() -> None:
    bank = _bank()
    results = bank.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=10,
        exclude_episode=("libero", 0, 4),
        exclude_source_episode_sha256="2" * 64,
    )

    assert [item.index for item in results] == [3]


def test_search_can_exclude_duplicate_feature_content_across_episode_ids() -> None:
    bank = _bank()
    results = bank.search(
        np.asarray([1.0, 0.0], dtype=np.float32),
        top_k=10,
        exclude_episode=("libero", 0, 5),
        exclude_feature_episode_sha256="b" * 64,
    )

    assert [item.index for item in results] == [0, 1]
