from __future__ import annotations

import importlib

import numpy as np
import pytest

from fastwam.memory.feature_precompute import (
    DinoFactualFeatures,
    FastWAMProcessorAdapter,
    FeaturePrecomputeError,
    ProcessedActionState,
    assemble_episode_features,
    build_m1_context_keys,
    extract_dino_factual_features,
    factual_gripper_from_state,
    pool_dino_patch_grid_2x2,
)


def _assert_immutable_float32(value: np.ndarray) -> None:
    assert value.dtype == np.dtype(np.float32)
    assert value.flags.c_contiguous
    assert not value.flags.writeable
    assert np.isfinite(value).all()


def test_module_import_does_not_bind_torch_or_hydra() -> None:
    module = importlib.import_module("fastwam.memory.feature_precompute")
    assert "torch" not in module.__dict__
    assert "hydra" not in module.__dict__


def test_factual_gripper_uses_absolute_last_two_state_dimensions() -> None:
    state = np.asarray(
        [[10.0, 20.0, -0.2, 0.3], [11.0, 21.0, -0.4, -0.1]],
        dtype=np.float64,
    )

    gripper = factual_gripper_from_state(state)

    np.testing.assert_allclose(gripper, np.asarray([0.5, 0.5], dtype=np.float32))
    _assert_immutable_float32(gripper)


def test_context_key_has_equal_visual_and_catalog_task_energy() -> None:
    cls = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)

    keys = build_m1_context_keys(
        cls,
        task_index=2,
        catalog_task_count=4,
    )

    expected_scale = np.sqrt(0.5)
    np.testing.assert_allclose(
        keys[:, :2],
        np.asarray([[0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
        * expected_scale,
        atol=1e-7,
    )
    np.testing.assert_allclose(np.linalg.norm(keys[:, :2], axis=1), expected_scale)
    np.testing.assert_allclose(np.linalg.norm(keys[:, 2:], axis=1), expected_scale)
    np.testing.assert_allclose(
        keys[:, 2:][:, 2], np.ones(2, dtype=np.float32) * expected_scale
    )
    np.testing.assert_allclose(np.linalg.norm(keys, axis=1), 1.0, atol=1e-7)
    _assert_immutable_float32(keys)


def test_visual_only_context_is_normalized_cls_without_task_dimensions() -> None:
    cls = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)

    keys = build_m1_context_keys(
        cls,
        task_index=1,
        catalog_task_count=3,
        visual_only=True,
    )

    assert keys.shape == (2, 2)
    np.testing.assert_allclose(keys, [[0.6, 0.8], [0.0, 1.0]])
    _assert_immutable_float32(keys)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"task_index": 3, "catalog_task_count": 3}, "outside"),
        ({"task_index": 0, "catalog_task_count": 0}, "positive"),
    ],
)
def test_context_key_rejects_invalid_catalog_task_contract(kwargs, match: str) -> None:
    with pytest.raises(FeaturePrecomputeError, match=match):
        build_m1_context_keys(np.ones((2, 3), dtype=np.float32), **kwargs)


def test_context_key_rejects_zero_or_nonfinite_visual_rows() -> None:
    with pytest.raises(FeaturePrecomputeError, match="non-zero"):
        build_m1_context_keys(
            np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            task_index=0,
            catalog_task_count=1,
        )
    with pytest.raises(FeaturePrecomputeError, match="finite"):
        build_m1_context_keys(
            np.asarray([[np.nan, 1.0]], dtype=np.float32),
            task_index=0,
            catalog_task_count=1,
        )


def test_patch_grid_adaptive_pool_returns_row_major_four_tokens() -> None:
    grid = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)

    pooled = pool_dino_patch_grid_2x2(grid)

    assert pooled.shape == (1, 4, 1)
    np.testing.assert_allclose(
        pooled[0, :, 0],
        np.asarray([2.5, 4.5, 10.5, 12.5], dtype=np.float32),
    )
    _assert_immutable_float32(pooled)


def test_flattened_odd_patch_grid_matches_adaptive_pool_regions() -> None:
    grid = np.arange(15, dtype=np.float32).reshape(1, 3, 5, 1)

    pooled = pool_dino_patch_grid_2x2(
        grid.reshape(1, 15, 1), patch_grid_size=(3, 5)
    )

    expected = np.asarray(
        [
            grid[:, 0:2, 0:3, :].mean(),
            grid[:, 0:2, 2:5, :].mean(),
            grid[:, 1:3, 0:3, :].mean(),
            grid[:, 1:3, 2:5, :].mean(),
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(pooled[0, :, 0], expected)

    with pytest.raises(FeaturePrecomputeError, match="positive integers"):
        pool_dino_patch_grid_2x2(
            grid.reshape(1, 15, 1), patch_grid_size=(3.0, 5)
        )


def test_extract_dino_features_skips_register_tokens_and_validates_grid() -> None:
    # CLS + 2 register tokens + a 2x2 patch grid.
    hidden = np.arange(2 * 7 * 3, dtype=np.float32).reshape(2, 7, 3)

    factual = extract_dino_factual_features(
        hidden,
        patch_grid_size=(2, 2),
        register_token_count=2,
    )

    assert isinstance(factual, DinoFactualFeatures)
    np.testing.assert_array_equal(factual.cls, hidden[:, 0])
    np.testing.assert_array_equal(factual.spatial, hidden[:, 3:].reshape(2, 4, 3))
    _assert_immutable_float32(factual.cls)
    _assert_immutable_float32(factual.spatial)

    with pytest.raises(FeaturePrecomputeError, match="token count"):
        extract_dino_factual_features(
            hidden[:, :-1],
            patch_grid_size=(2, 2),
            register_token_count=2,
        )


class _RecordingStage:
    def __init__(self, name: str, calls: list[str], transform) -> None:
        self._name = name
        self._calls = calls
        self._transform = transform

    def forward(self, batch):
        self._calls.append(self._name)
        return self._transform(batch)


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.normalizer = _RecordingStage(
            "normalizer.forward",
            self.calls,
            self._normalize,
        )
        self.action_state_merger = _RecordingStage(
            "merger.forward",
            self.calls,
            self._merge,
        )

    def action_state_transform(self, batch):
        self.calls.append("action_state_transform")
        assert batch["action"]["default"].shape[0] == 3
        assert batch["state"]["default"].shape[0] == 4
        batch["action"]["default"] += 1.0
        batch["state"]["default"] += 2.0
        return batch

    @staticmethod
    def _normalize(batch):
        batch["action"]["default"] *= 2.0
        batch["state"]["default"] *= 3.0
        return batch

    @staticmethod
    def _merge(batch):
        batch["action"] = batch["action"]["default"]
        batch["state"] = batch["state"]["default"]
        return batch


def test_processor_adapter_uses_exact_chain_and_drops_terminal_raw_action() -> None:
    processor = _FakeProcessor()
    adapter = FastWAMProcessorAdapter(processor, tensor_backend="numpy")
    raw_action = np.arange(4 * 2, dtype=np.float64).reshape(4, 2)
    raw_state = np.asarray(
        [
            [0.0, 1.0, -0.1, 0.2],
            [1.0, 2.0, -0.2, 0.3],
            [2.0, 3.0, 0.4, -0.1],
            [3.0, 4.0, -0.5, -0.5],
        ],
        dtype=np.float64,
    )

    output = adapter.process(raw_action, raw_state)

    assert isinstance(output, ProcessedActionState)
    assert processor.calls == [
        "action_state_transform",
        "normalizer.forward",
        "merger.forward",
    ]
    np.testing.assert_allclose(output.model_actions, (raw_action[:-1] + 1.0) * 2.0)
    np.testing.assert_allclose(output.proprio, (raw_state + 2.0) * 3.0)
    # Factual gripper comes from the untouched raw observations.
    np.testing.assert_allclose(output.gripper, [0.3, 0.5, 0.5, 1.0])
    for array in (output.model_actions, output.proprio, output.gripper):
        _assert_immutable_float32(array)
    # The adapter owns copies and never mutates caller data.
    np.testing.assert_array_equal(raw_action, np.arange(8).reshape(4, 2))


def test_processor_adapter_rejects_non_aligned_per_frame_raw_data() -> None:
    adapter = FastWAMProcessorAdapter(_FakeProcessor(), tensor_backend="numpy")
    with pytest.raises(FeaturePrecomputeError, match="both have N rows"):
        adapter.process(
            np.ones((3, 2), dtype=np.float32),
            np.ones((4, 4), dtype=np.float32),
        )


def test_assembler_builds_strict_immutable_episode_features() -> None:
    observations = 4
    actions = np.arange(3 * 2, dtype=np.float64).reshape(3, 2)
    proprio = np.arange(4 * 4, dtype=np.float64).reshape(4, 4)
    cls = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        dtype=np.float64,
    )
    semantics = np.arange(observations * 4 * 2, dtype=np.float64).reshape(
        observations, 4, 2
    )
    raw_state = np.asarray(
        [[0, 0, -0.1, 0.2], [0, 0, -0.2, 0.3], [0, 0, 0.4, -0.1], [0, 0, 0, 0]],
        dtype=np.float64,
    )

    episode = assemble_episode_features(
        dataset_id="libero",
        dataset_index=0,
        episode_index=7,
        task_index=1,
        catalog_task_count=3,
        source_episode_sha256="a" * 64,
        model_actions=actions,
        proprio=proprio,
        dino_cls=cls,
        semantic_features=semantics,
        raw_state_for_gripper=raw_state,
    )

    assert episode.model_actions.shape == (observations - 1, 2)
    assert episode.proprio.shape == (observations, 4)
    assert episode.semantic_features.shape == (observations, 4, 2)
    assert episode.context_keys.shape == (observations, 5)
    np.testing.assert_allclose(episode.gripper, [0.3, 0.5, 0.5, 0.0])
    for array in (
        episode.model_actions,
        episode.proprio,
        episode.semantic_features,
        episode.context_keys,
        episode.gripper,
    ):
        _assert_immutable_float32(array)


def test_assembler_rejects_any_temporal_contract_mismatch() -> None:
    base = {
        "dataset_id": "libero",
        "dataset_index": 0,
        "episode_index": 7,
        "task_index": 0,
        "catalog_task_count": 1,
        "source_episode_sha256": "a" * 64,
        "model_actions": np.ones((3, 2), dtype=np.float32),
        "proprio": np.ones((4, 4), dtype=np.float32),
        "dino_cls": np.ones((4, 2), dtype=np.float32),
        "semantic_features": np.ones((4, 4, 2), dtype=np.float32),
        "gripper": np.ones(4, dtype=np.float32),
    }
    with pytest.raises(FeaturePrecomputeError, match="N observations and N-1"):
        assemble_episode_features(
            **{**base, "model_actions": np.ones((4, 2), dtype=np.float32)}
        )
    with pytest.raises(FeaturePrecomputeError, match="semantic_features must have"):
        assemble_episode_features(
            **{
                **base,
                "semantic_features": np.ones((3, 4, 2), dtype=np.float32),
            }
        )
    with pytest.raises(FeaturePrecomputeError, match="exactly one"):
        assemble_episode_features(
            **base,
            raw_state_for_gripper=np.ones((4, 4), dtype=np.float32),
        )


def test_processed_action_state_rejects_nonfinite_output() -> None:
    with pytest.raises(FeaturePrecomputeError, match="finite"):
        ProcessedActionState(
            model_actions=np.asarray([[np.nan]], dtype=np.float32),
            proprio=np.ones((2, 2), dtype=np.float32),
            gripper=np.ones(2, dtype=np.float32),
        )
