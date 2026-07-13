from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from fastwam.memory.bank_builder import EpisodeFeatures
from fastwam.memory.feature_cache import (
    FEATURE_CACHE_SCHEMA,
    CachedEpisodeFeatures,
    FeatureCacheError,
    FeatureCacheIntegrityError,
    FeatureCacheManifest,
    FeatureCacheMetadata,
    FeatureCacheMismatchError,
    LoadedFeatureCache,
    canonical_feature_content_hash,
    feature_cache_manifest_path,
    load_episode_feature_cache,
    save_episode_feature_cache,
)


def _episode(
    *,
    with_vae: bool = True,
    dataset_id: str = "libero-train",
    dataset_index: int = 2,
    episode_index: int = 17,
    source_episode_sha256: str = "1" * 64,
) -> EpisodeFeatures:
    steps = 6
    return EpisodeFeatures(
        dataset_id=dataset_id,
        dataset_index=dataset_index,
        episode_index=episode_index,
        task_index=4,
        source_episode_sha256=source_episode_sha256,
        model_actions=np.arange(steps * 7, dtype=np.float32).reshape(steps, 7),
        proprio=np.arange((steps + 1) * 8, dtype=np.float32).reshape(steps + 1, 8),
        gripper=np.asarray([0, 0, 1, 1, 0, 0, 0], dtype=np.float32),
        context_keys=(
            np.arange((steps + 1) * 5, dtype=np.float32).reshape(steps + 1, 5)
            + 1.0
        ),
        semantic_features=np.arange(
            (steps + 1) * 2 * 3, dtype=np.float32
        ).reshape(steps + 1, 2, 3),
        vae_features=(
            np.arange((steps + 1) * 2 * 2 * 2, dtype=np.float32).reshape(
                steps + 1, 2, 2, 2
            )
            if with_vae
            else None
        ),
    )


def _metadata(
    episode: EpisodeFeatures,
    *,
    encoder: str = "c",
    split: str = "train",
) -> FeatureCacheMetadata:
    return FeatureCacheMetadata.for_episode(
        episode,
        split=split,
        catalog_hash="a" * 64,
        normalizer_hash="b" * 64,
        encoder_hash=encoder * 64,
        camera_hash="d" * 64,
    )


def _rewrite_manifest(path: Path, transform) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    transform(value)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("with_vae", [False, True])
def test_round_trip_preserves_episode_arrays_and_provenance(
    tmp_path: Path, with_vae: bool
) -> None:
    episode = _episode(with_vae=with_vae)
    metadata = _metadata(episode)
    payload = tmp_path / "episode-17.npz"

    written = save_episode_feature_cache(payload, episode, metadata=metadata)
    loaded = load_episode_feature_cache(payload, expected_metadata=metadata)
    reread = FeatureCacheManifest.read(feature_cache_manifest_path(payload))

    assert isinstance(loaded, LoadedFeatureCache)
    assert isinstance(loaded, CachedEpisodeFeatures)
    assert written == reread
    assert reread.schema == FEATURE_CACHE_SCHEMA
    assert reread.metadata == metadata
    assert reread.metadata.split == "train"
    assert loaded.metadata == metadata
    assert loaded.manifest == reread
    assert loaded.payload_path == payload.resolve()
    assert loaded.episode_content_hash == reread.episode_content_hash
    assert loaded.features.source_episode_sha256 == episode.source_episode_sha256
    assert set(reread.arrays) == set(reread.array_hashes)
    assert len(reread.payload_hash) == 64
    assert len(reread.episode_content_hash) == 64
    for name in (
        "model_actions",
        "proprio",
        "gripper",
        "context_keys",
        "semantic_features",
    ):
        loaded_array = getattr(loaded.features, name)
        np.testing.assert_array_equal(loaded_array, getattr(episode, name))
        assert not loaded_array.flags.writeable
    if with_vae:
        np.testing.assert_array_equal(loaded.features.vae_features, episode.vae_features)
        assert not loaded.features.vae_features.flags.writeable
        assert "vae_features" in reread.arrays
    else:
        assert loaded.features.vae_features is None
        assert "vae_features" not in reread.arrays

    with pytest.raises(ValueError):
        loaded.features.model_actions[0, 0] = -1.0


def test_file_sha256_detects_payload_tampering(tmp_path: Path) -> None:
    episode = _episode()
    payload = tmp_path / "episode.npz"
    save_episode_feature_cache(payload, episode, metadata=_metadata(episode))

    with payload.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(FeatureCacheIntegrityError, match="SHA-256 mismatch"):
        load_episode_feature_cache(payload)


def test_per_array_sha256_detects_repacked_array_tampering(tmp_path: Path) -> None:
    episode = _episode()
    payload = tmp_path / "episode.npz"
    save_episode_feature_cache(payload, episode, metadata=_metadata(episode))
    manifest_path = feature_cache_manifest_path(payload)

    with np.load(payload, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["model_actions"][0, 0] += 1.0
    with payload.open("wb") as handle:
        np.savez(handle, **arrays)

    # Simulate a repack operation that refreshed only the whole-file digest.
    file_digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    _rewrite_manifest(manifest_path, lambda value: value.update(payload_hash=file_digest))

    with pytest.raises(FeatureCacheIntegrityError, match="array 'model_actions'"):
        load_episode_feature_cache(payload)


def test_expected_encoder_hash_mismatch_is_rejected_before_reuse(tmp_path: Path) -> None:
    episode = _episode()
    payload = tmp_path / "episode.npz"
    save_episode_feature_cache(payload, episode, metadata=_metadata(episode, encoder="c"))

    with pytest.raises(FeatureCacheMismatchError, match="requested provenance"):
        load_episode_feature_cache(
            payload,
            expected_metadata=_metadata(episode, encoder="e"),
        )


def test_manifest_and_embedded_metadata_mismatch_is_detected(tmp_path: Path) -> None:
    episode = _episode()
    payload = tmp_path / "episode.npz"
    save_episode_feature_cache(payload, episode, metadata=_metadata(episode))
    manifest_path = feature_cache_manifest_path(payload)

    def change_identity(value: dict) -> None:
        value["metadata"]["episode_index"] = 18

    _rewrite_manifest(manifest_path, change_identity)

    with pytest.raises(FeatureCacheMismatchError, match="embedded metadata"):
        load_episode_feature_cache(payload)


def test_save_revalidates_mutated_episode_features(tmp_path: Path) -> None:
    episode = _episode()
    metadata = _metadata(episode)
    episode.semantic_features.flags.writeable = True
    episode.semantic_features[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        save_episode_feature_cache(tmp_path / "episode.npz", episode, metadata=metadata)
    assert not (tmp_path / "episode.npz").exists()
    assert not feature_cache_manifest_path(tmp_path / "episode.npz").exists()


def test_metadata_identity_and_existing_cache_are_strict(tmp_path: Path) -> None:
    episode = _episode()
    wrong_identity = FeatureCacheMetadata(
        dataset_id=episode.dataset_id,
        dataset_index=episode.dataset_index,
        episode_index=episode.episode_index + 1,
        task_index=episode.task_index,
        split="train",
        source_episode_sha256=episode.source_episode_sha256,
        catalog_hash="a" * 64,
        normalizer_hash="b" * 64,
        encoder_hash="c" * 64,
        camera_hash="d" * 64,
    )
    with pytest.raises(FeatureCacheMismatchError, match="identity"):
        save_episode_feature_cache(
            tmp_path / "wrong.npz", episode, metadata=wrong_identity
        )

    payload = tmp_path / "episode.npz"
    save_episode_feature_cache(payload, episode, metadata=_metadata(episode))
    with pytest.raises(FileExistsError):
        save_episode_feature_cache(payload, episode, metadata=_metadata(episode))
    assert list(tmp_path.glob(".*.tmp")) == []


def test_object_arrays_cannot_be_loaded(tmp_path: Path) -> None:
    episode = _episode()
    payload = tmp_path / "episode.npz"
    save_episode_feature_cache(payload, episode, metadata=_metadata(episode))
    manifest_path = feature_cache_manifest_path(payload)

    with np.load(payload, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["model_actions"] = np.asarray([[object()]], dtype=object)
    with payload.open("wb") as handle:
        np.savez(handle, **arrays)
    file_digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    _rewrite_manifest(manifest_path, lambda value: value.update(payload_hash=file_digest))

    with pytest.raises(FeatureCacheError, match="cannot load NumPy payload"):
        load_episode_feature_cache(payload)


def test_hash_fields_require_canonical_sha256() -> None:
    episode = _episode()
    with pytest.raises(ValueError, match="lowercase 64-character"):
        FeatureCacheMetadata.for_episode(
            episode,
            split="train",
            catalog_hash="not-a-hash",
            normalizer_hash="b" * 64,
            encoder_hash="c" * 64,
            camera_hash="d" * 64,
        )


def test_source_episode_hash_must_match_episode_features(tmp_path: Path) -> None:
    episode = _episode(source_episode_sha256="1" * 64)
    metadata = FeatureCacheMetadata(
        dataset_id=episode.dataset_id,
        dataset_index=episode.dataset_index,
        episode_index=episode.episode_index,
        task_index=episode.task_index,
        split="train",
        source_episode_sha256="2" * 64,
        catalog_hash="a" * 64,
        normalizer_hash="b" * 64,
        encoder_hash="c" * 64,
        camera_hash="d" * 64,
    )

    with pytest.raises(FeatureCacheMismatchError, match="identity"):
        save_episode_feature_cache(tmp_path / "episode.npz", episode, metadata=metadata)


def test_metadata_rejects_unknown_split() -> None:
    with pytest.raises(ValueError, match="split must be train, dev, or test"):
        _metadata(_episode(), split="validation")


def test_canonical_content_hash_is_ordered_and_ignores_identity_metadata() -> None:
    data = {
        "model_actions": "1" * 64,
        "proprio": "2" * 64,
        "gripper": "3" * 64,
        "context_keys": "4" * 64,
        "semantic_features": "5" * 64,
        "_metadata_json_utf8": "6" * 64,
    }
    expected = canonical_feature_content_hash(data)
    reordered = dict(reversed(list(data.items())))
    reordered["_metadata_json_utf8"] = "7" * 64

    assert canonical_feature_content_hash(reordered) == expected
    reordered["model_actions"] = "8" * 64
    assert canonical_feature_content_hash(reordered) != expected


def test_episode_content_hash_supports_cross_identity_deduplication(
    tmp_path: Path,
) -> None:
    first = _episode(
        dataset_id="dataset-a",
        dataset_index=0,
        episode_index=1,
        source_episode_sha256="1" * 64,
    )
    second = _episode(
        dataset_id="dataset-b",
        dataset_index=9,
        episode_index=22,
        source_episode_sha256="2" * 64,
    )
    first_manifest = save_episode_feature_cache(
        tmp_path / "first.npz", first, metadata=_metadata(first)
    )
    second_manifest = save_episode_feature_cache(
        tmp_path / "second.npz", second, metadata=_metadata(second)
    )

    assert first_manifest.metadata != second_manifest.metadata
    assert first_manifest.episode_content_hash == second_manifest.episode_content_hash


def test_manifest_rejects_mismatched_episode_content_hash(tmp_path: Path) -> None:
    episode = _episode()
    payload = tmp_path / "episode.npz"
    save_episode_feature_cache(payload, episode, metadata=_metadata(episode))
    manifest_path = feature_cache_manifest_path(payload)
    _rewrite_manifest(
        manifest_path,
        lambda value: value.update(episode_content_hash="f" * 64),
    )

    with pytest.raises(FeatureCacheIntegrityError, match="episode_content_hash"):
        load_episode_feature_cache(payload)
