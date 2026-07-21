from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import numpy as np
import pytest


if os.environ.get("WARM_REQUIRE_TORCH_TESTS") == "1":
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - server contract
        raise RuntimeError(
            "WARM_REQUIRE_TORCH_TESTS=1 but PyTorch is unavailable"
        ) from error
else:
    torch = pytest.importorskip("torch")


from fastwam.datasets.warm_candidates import (  # noqa: E402
    RuntimeCandidateDatasetAdapter,
    WARM_CANDIDATE_MASK,
)
from fastwam.datasets.warm_retrospective import (  # noqa: E402
    RetrospectiveFeatureStore,
    RetrospectiveFeatureStoreError,
    RuntimeRetrospectiveDatasetAdapter,
    WARM_CANDIDATE_CONTEXT,
    WARM_CANDIDATE_EFFECT_DELTA,
    WARM_CANDIDATE_EFFECT_POST,
    WARM_CANDIDATE_EFFECT_PRE,
    WARM_CANDIDATE_GRIPPER,
    WARM_CANDIDATE_START_PROPRIO,
    WARM_CANDIDATE_SUPPORT,
    WARM_CANDIDATE_TIMING,
    WARM_CURRENT_CONTEXT,
    WARM_CURRENT_SEMANTIC,
    WARM_EPISODE_MASK,
    WARM_EPISODE_ACTION_MASK,
    WARM_EPISODE_ACTION_SUMMARIES,
    WARM_EPISODE_TOKENS,
    WARM_FUTURE_SEMANTIC,
    WARM_FUTURE_VALID,
    WARM_TARGET_EFFECT,
    _action_summary_vector,
    _causal_event_frames_by_query,
    collect_feature_payloads,
    collect_feature_payloads_from_list,
)
from fastwam.memory.candidate_cache import QueryId  # noqa: E402
from fastwam.memory.bank_builder import EpisodeFeatures  # noqa: E402
from fastwam.memory.feature_cache import (  # noqa: E402
    FeatureCacheMetadata,
    load_episode_feature_cache,
    save_episode_feature_cache,
)
from fastwam.memory.offline_pipeline import (  # noqa: E402
    FeatureCacheCollection,
    FeatureCollectionContract,
)
from fastwam.memory.runtime_candidates import RuntimeCandidateResolver  # noqa: E402
from fastwam.memory.online_episode_memory import (  # noqa: E402
    OnlineRetrospectiveEpisodeMemory,
)
from tests.test_runtime_candidate_dataset_adapter import (  # noqa: E402
    _BaseDataset,
    _catalog,
    _sample,
)
from tests.test_runtime_candidates import (  # noqa: E402
    QUERY_CORPUS_HASH,
    _action_contract,
    _write_artifacts,
)


def _episode(*, semantic_width: int = 2) -> EpisodeFeatures:
    steps = 8
    semantic = np.arange(
        (steps + 1) * 2 * semantic_width, dtype=np.float32
    ).reshape(steps + 1, 2, semantic_width)
    return EpisodeFeatures(
        dataset_id="libero",
        dataset_index=0,
        episode_index=0,
        task_index=0,
        source_episode_sha256="1" * 64,
        model_actions=np.arange(steps * 3, dtype=np.float32).reshape(steps, 3),
        proprio=np.arange((steps + 1) * 8, dtype=np.float32).reshape(
            steps + 1, 8
        ),
        gripper=np.asarray(
            [0.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        context_keys=np.arange(
            (steps + 1) * 2, dtype=np.float32
        ).reshape(steps + 1, 2),
        semantic_features=semantic,
        vae_features=np.arange(
            (steps + 1) * 4, dtype=np.float32
        ).reshape(steps + 1, 2, 2),
    )


def _loaded_feature(
    root: Path,
    *,
    catalog_hash: str,
    split: str = "train",
    semantic_width: int = 2,
):
    episode = _episode(semantic_width=semantic_width)
    metadata = FeatureCacheMetadata.for_episode(
        episode,
        split=split,
        catalog_hash=catalog_hash,
        normalizer_hash="2" * 64,
        encoder_hash="3" * 64,
        camera_hash="4" * 64,
    )
    path = root / "episode-0.npz"
    save_episode_feature_cache(path, episode, metadata=metadata)
    return path, load_episode_feature_cache(path)


def _collection(loaded, *, split: str, content_hash: str) -> FeatureCacheCollection:
    metadata = loaded.metadata
    return FeatureCacheCollection(
        records=(loaded,),
        contract=FeatureCollectionContract(
            catalog_hash=metadata.catalog_hash,
            normalizer_hash=metadata.normalizer_hash,
            encoder_hash=metadata.encoder_hash,
            camera_hash=metadata.camera_hash,
        ),
        content_hash=content_hash,
        split=split,
    )


def _adapter_fixture(
    tmp_path: Path,
    *,
    frame_index: int = 0,
    semantic_width: int = 2,
) -> RuntimeRetrospectiveDatasetAdapter:
    catalog = _catalog(split="train", length=8)
    bank, cache, _ = _write_artifacts(
        tmp_path / "artifacts", catalog_hash=catalog.content_sha256
    )
    resolver = RuntimeCandidateResolver.from_artifacts(
        bank,
        cache,
        expected_query_split="train",
        expected_query_corpus_sha256=QUERY_CORPUS_HASH,
        expected_action_space=_action_contract(),
    )
    _, loaded = _loaded_feature(
        tmp_path / "features",
        catalog_hash=catalog.content_sha256,
        semantic_width=semantic_width,
    )
    store = RetrospectiveFeatureStore(
        _collection(
            loaded,
            split="train",
            content_hash=resolver.query_corpus_sha256,
        ),
        expected_split="train",
        expected_collection_sha256=resolver.query_corpus_sha256,
        expected_catalog_sha256=resolver.query_catalog_sha256,
        action_horizon=resolver.action_horizon,
        recent_event_capacity=2,
    )
    candidate_dataset = RuntimeCandidateDatasetAdapter(
        _BaseDataset([_sample(frame_index)]), resolver, catalog
    )
    return RuntimeRetrospectiveDatasetAdapter(candidate_dataset, store)


def test_feature_payload_discovery_requires_verified_sidecars(tmp_path: Path) -> None:
    catalog = _catalog()
    payload, _ = _loaded_feature(
        tmp_path / "features", catalog_hash=catalog.content_sha256
    )
    (tmp_path / "features" / "unverified.npz").write_bytes(b"not-a-cache")

    assert collect_feature_payloads(tmp_path / "features") == (
        payload.resolve(),
    )
    with pytest.raises(FileNotFoundError, match="directory not found"):
        collect_feature_payloads(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(RetrospectiveFeatureStoreError, match="no episode"):
        collect_feature_payloads(tmp_path / "empty")


def test_feature_list_is_relative_ordered_and_fail_closed(tmp_path: Path) -> None:
    catalog = _catalog()
    first, _ = _loaded_feature(
        tmp_path / "features", catalog_hash=catalog.content_sha256
    )
    second = tmp_path / "features" / "episode-1.npz"
    second.write_bytes(first.read_bytes())
    second.with_suffix(".manifest.json").write_bytes(
        first.with_suffix(".manifest.json").read_bytes()
    )
    list_path = tmp_path / "features.list"
    list_path.write_text(
        "# immutable split list\nfeatures/episode-1.npz\n"
        f"{first.resolve()}\n",
        encoding="utf-8",
    )

    assert collect_feature_payloads_from_list(list_path) == (
        second.resolve(),
        first.resolve(),
    )

    list_path.write_text(
        "features/episode-0.npz\nfeatures/episode-0.npz\n",
        encoding="utf-8",
    )
    with pytest.raises(RetrospectiveFeatureStoreError, match="duplicate"):
        collect_feature_payloads_from_list(list_path)

    list_path.write_text("features/unverified.npz\n", encoding="utf-8")
    (tmp_path / "features" / "unverified.npz").write_bytes(b"not-a-cache")
    with pytest.raises(RetrospectiveFeatureStoreError, match="manifest sidecar"):
        collect_feature_payloads_from_list(list_path)


def test_store_fails_closed_on_split_corpus_and_catalog_mismatch(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    _, loaded = _loaded_feature(
        tmp_path / "features", catalog_hash=catalog.content_sha256
    )
    collection = _collection(
        loaded, split="train", content_hash=QUERY_CORPUS_HASH
    )
    common = {
        "expected_collection_sha256": QUERY_CORPUS_HASH,
        "expected_catalog_sha256": catalog.content_sha256,
        "action_horizon": 4,
    }

    with pytest.raises(RetrospectiveFeatureStoreError, match="feature split"):
        RetrospectiveFeatureStore(
            collection, expected_split="dev", **common
        )
    with pytest.raises(
        RetrospectiveFeatureStoreError, match="candidate query corpus"
    ):
        RetrospectiveFeatureStore(
            collection,
            expected_split="train",
            **{**common, "expected_collection_sha256": "f" * 64},
        )
    with pytest.raises(RetrospectiveFeatureStoreError, match="episode catalog"):
        RetrospectiveFeatureStore(
            collection,
            expected_split="train",
            **{**common, "expected_catalog_sha256": "e" * 64},
        )


def test_retrospective_adapter_has_fixed_shapes_and_zero_sentinel_rows(
    tmp_path: Path,
) -> None:
    adapter = _adapter_fixture(tmp_path)
    sample = adapter[0]

    assert sample[WARM_CANDIDATE_MASK].tolist() == [True, True, False]
    expected_shapes = {
        WARM_CANDIDATE_CONTEXT: (3, 2),
        WARM_CANDIDATE_EFFECT_PRE: (3, 2, 2),
        WARM_CANDIDATE_EFFECT_POST: (3, 2, 2),
        WARM_CANDIDATE_EFFECT_DELTA: (3, 2, 2),
        WARM_CANDIDATE_START_PROPRIO: (3, 8),
        WARM_CANDIDATE_GRIPPER: (3, 5),
        WARM_CANDIDATE_TIMING: (3, 4),
        WARM_CANDIDATE_SUPPORT: (3,),
        WARM_CURRENT_CONTEXT: (2,),
        WARM_CURRENT_SEMANTIC: (2, 2),
        WARM_FUTURE_SEMANTIC: (2, 2),
        WARM_TARGET_EFFECT: (2, 2),
        WARM_EPISODE_TOKENS: (6, 2),
        WARM_EPISODE_MASK: (6,),
        WARM_EPISODE_ACTION_SUMMARIES: (2, 13),
        WARM_EPISODE_ACTION_MASK: (2,),
    }
    for field, shape in expected_shapes.items():
        assert tuple(sample[field].shape) == shape, field
    for field in (
        WARM_CANDIDATE_CONTEXT,
        WARM_CANDIDATE_EFFECT_PRE,
        WARM_CANDIDATE_EFFECT_POST,
        WARM_CANDIDATE_EFFECT_DELTA,
        WARM_CANDIDATE_START_PROPRIO,
        WARM_CANDIDATE_GRIPPER,
        WARM_CANDIDATE_TIMING,
        WARM_CANDIDATE_SUPPORT,
    ):
        assert torch.count_nonzero(sample[field][2]).item() == 0, field
    assert sample[WARM_CANDIDATE_SUPPORT].tolist() == [1.0, 1.0, 0.0]
    assert sample[WARM_FUTURE_VALID].dtype == torch.bool
    assert bool(sample[WARM_FUTURE_VALID].item())
    assert sample[WARM_EPISODE_MASK].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert not bool(sample[WARM_EPISODE_ACTION_MASK].any().item())
    assert torch.count_nonzero(sample[WARM_EPISODE_ACTION_SUMMARIES]).item() == 0


def test_empty_candidate_row_and_invalid_future_are_exactly_zero(
    tmp_path: Path,
) -> None:
    # Frame five is an explicitly padded action tail.  It is absent from the
    # candidate cache and frame+H lies past the factual semantic episode.
    adapter = _adapter_fixture(tmp_path, frame_index=5)
    sample = adapter[0]

    assert not bool(sample[WARM_CANDIDATE_MASK].any().item())
    for field in (
        WARM_CANDIDATE_CONTEXT,
        WARM_CANDIDATE_EFFECT_PRE,
        WARM_CANDIDATE_EFFECT_POST,
        WARM_CANDIDATE_EFFECT_DELTA,
        WARM_CANDIDATE_START_PROPRIO,
        WARM_CANDIDATE_GRIPPER,
        WARM_CANDIDATE_TIMING,
        WARM_CANDIDATE_SUPPORT,
    ):
        assert torch.count_nonzero(sample[field]).item() == 0, field
    assert not bool(sample[WARM_FUTURE_VALID].item())
    assert torch.count_nonzero(sample[WARM_FUTURE_SEMANTIC]).item() == 0
    assert torch.count_nonzero(sample[WARM_TARGET_EFFECT]).item() == 0
    # At frame five with H=4, frame one is the latest factual observation
    # that an online replan could already have committed.  It is protected
    # even when it was not an event-write point.
    assert sample[WARM_EPISODE_MASK].tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert torch.equal(
        sample[WARM_EPISODE_TOKENS][2:4],
        torch.from_numpy(_episode().semantic_features[1]),
    )


def test_action_summary_repetition_uses_all_bounded_predecessors() -> None:
    first_actions = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    different_actions = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
    current_actions = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    _, first = _action_summary_vector(
        first_actions,
        action_horizon=4,
        gripper_indices=(2,),
        previous_signatures=(),
    )
    _, different = _action_summary_vector(
        different_actions,
        action_horizon=4,
        gripper_indices=(2,),
        previous_signatures=(first,),
    )
    current, _ = _action_summary_vector(
        current_actions,
        action_horizon=4,
        gripper_indices=(2,),
        previous_signatures=(first, different),
    )

    assert current[-2] == pytest.approx(1.0)
    assert current[-1] == pytest.approx(1.0)


def test_action_summary_keeps_model_width_but_uses_compact_signature() -> None:
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, :6] = np.arange(24, dtype=np.float32).reshape(4, 6)
    actions[-1, 6] = 1.0

    vector, signature = _action_summary_vector(
        actions,
        action_horizon=16,
        gripper_indices=(6,),
        previous_signatures=(),
    )

    assert vector.shape == (3 * 7 + 4,)
    assert signature.shape == (2 * 7 + 1,)
    np.testing.assert_allclose(vector[:7], signature[:7])
    np.testing.assert_allclose(vector[7:14], signature[7:14])
    assert vector[20] == signature[14] == pytest.approx(1.0)


def test_causal_event_replay_is_invariant_to_future_suffix() -> None:
    base = _episode()
    actions = np.array(base.model_actions, copy=True)
    proprio = np.array(base.proprio, copy=True)
    semantic = np.array(base.semantic_features, copy=True)
    vae = np.array(base.vae_features, copy=True)
    gripper = np.array(base.gripper, copy=True)
    actions[5:] *= -11.0
    proprio[6:] += 1000.0
    semantic[6:] -= 777.0
    vae[6:] += 333.0
    gripper[6:] = 1.0 - gripper[6:]
    altered = replace(
        base,
        model_actions=actions,
        proprio=proprio,
        semantic_features=semantic,
        vae_features=vae,
        gripper=gripper,
    )

    kwargs = {
        "action_horizon": 4,
        "action_chunk_size": 4,
        "action_dim": 3,
        "semantic_dim": 2,
        "gripper_indices": (2,),
        "recent_event_capacity": 2,
    }
    original_replay = _causal_event_frames_by_query(base, **kwargs)
    altered_replay = _causal_event_frames_by_query(altered, **kwargs)

    assert original_replay[5] == altered_replay[5]


def test_offline_payload_matches_online_episode_state_machine(tmp_path: Path) -> None:
    catalog = _catalog()
    _, loaded = _loaded_feature(
        tmp_path / "features", catalog_hash=catalog.content_sha256
    )
    store = RetrospectiveFeatureStore(
        _collection(loaded, split="train", content_hash=QUERY_CORPUS_HASH),
        expected_split="train",
        expected_collection_sha256=QUERY_CORPUS_HASH,
        expected_catalog_sha256=catalog.content_sha256,
        action_horizon=4,
        recent_event_capacity=2,
        action_summary_capacity=2,
        action_summary_chunk_size=4,
        gripper_indices=(2,),
    )
    query = QueryId("libero", 0, 0, 5)
    payload = store.query_payload(query)
    features = loaded.features

    online = OnlineRetrospectiveEpisodeMemory(
        action_dim=3,
        action_horizon=4,
        semantic_dim=2,
        gripper_indices=(2,),
        recent_event_capacity=2,
    )
    online.begin_episode(1)
    online.record_factual_observation(
        frame_index=0,
        factual_payload={
            "world_tokens": features.semantic_features[0],
            "vae_latent": features.vae_features[0],
            "proprio": features.proprio[0],
        },
        executed_actions_since_previous=None,
    )
    online.record_factual_observation(
        frame_index=1,
        factual_payload={
            "world_tokens": features.semantic_features[1],
            "vae_latent": features.vae_features[1],
            "proprio": features.proprio[1],
        },
        executed_actions_since_previous=features.model_actions[0:1],
    )
    history = online.history_inputs(
        executed_actions_since_previous=features.model_actions[1:5]
    )
    assert history is not None

    np.testing.assert_array_equal(
        payload[WARM_EPISODE_TOKENS][payload[WARM_EPISODE_MASK]],
        history.episode_tokens,
    )
    np.testing.assert_allclose(
        payload[WARM_EPISODE_ACTION_SUMMARIES][
            payload[WARM_EPISODE_ACTION_MASK]
        ],
        history.episode_action_summaries,
        rtol=0.0,
        atol=1.0e-6,
    )


def test_adapter_rejects_feature_and_bank_semantic_shape_mismatch(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        RetrospectiveFeatureStoreError,
        match="semantic shape does not match",
    ):
        _adapter_fixture(tmp_path, semantic_width=3)
