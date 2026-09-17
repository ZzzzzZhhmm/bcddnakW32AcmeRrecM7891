"""Check the actual FastWAM processor and the narrow Piper training extension."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hydra")

from hydra.utils import instantiate
from omegaconf import OmegaConf

from fastwam.datasets.warm_candidates import RuntimeCandidateDatasetAdapter, RuntimeCandidateDatasetContractError
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.memory.action_contract import ActionSpaceContract
from fastwam.memory.feature_precompute import FastWAMProcessorAdapter
from fastwam.memory.server_feature_encoders import FastWAMImageAdapter
from fastwam.preprocessing.contracts import file_sha256
from fastwam.preprocessing.prepare import prepare
from tests.test_unified_preprocessing import config, manifest, real_episode, robotwin_episode


@pytest.mark.parametrize("kind", ["piper", "robotwin"])
def test_generated_processor_passes_formal_contract_and_normalizes(tmp_path, kind):
    cfg = config(tmp_path, kind)
    entries = ([real_episode(cfg, "a", offset=1), real_episode(cfg, "b", offset=20)] if kind == "piper" else
               [robotwin_episode(cfg, "a"), robotwin_episode(cfg, "b", offset=1)])
    manifest(cfg, entries)
    prepared = prepare(cfg)
    data = OmegaConf.load(prepared / "data_config.yaml")
    processor = instantiate(data.train.processor)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(prepared / "dataset_stats.json")))
    processor.eval()
    profile = cfg["profile"]
    action_space = ActionSpaceContract(action_dim=profile["action_dim"],
        arm_dims=tuple(i for i in range(profile["action_dim"]) if i not in profile["gripper_action_indices"]),
        gripper_dims=tuple(profile["gripper_action_indices"]), gripper_threshold=0.,
        normalization_mode="global:" + profile["normalization"],
        normalization_stats_sha256=file_sha256(prepared / "dataset_stats.json"),
        control_mode=profile["control_mode"], embodiment=profile["embodiment"])
    resolver = SimpleNamespace(action_space=action_space, action_horizon=profile["action_horizon"])
    RuntimeCandidateDatasetAdapter._validate_processor_action_space(processor, resolver)
    n = 5
    actions = np.zeros((n, profile["action_dim"]), np.float32)
    states = np.zeros((n, profile["state_dim"]), np.float32)
    adapter = FastWAMProcessorAdapter(processor)
    processed = adapter.process(actions, states, gripper_indices=profile["gripper_state_indices"])
    # Cached model-space vectors must equal the actual training processor, including clamp.
    batch = {"action": {"default": torch.from_numpy(actions[:-1].copy())},
             "state": {"default": torch.from_numpy(states.copy())}}
    expected = processor.action_state_merger.forward(processor.normalizer.forward(processor.action_state_transform(batch)))
    np.testing.assert_allclose(processed.model_actions, expected["action"].numpy(), atol=1e-6)
    np.testing.assert_allclose(processed.proprio, expected["state"].numpy(), atol=1e-6)
    image_adapter = FastWAMImageAdapter(processor, profile["camera_keys"],
        "horizontal" if kind == "piper" else "robotwin", benchmark_profile="libero" if kind == "piper" else "robotwin")
    images = image_adapter.prepare({key: np.zeros((n, 3, 8, 8), dtype=np.float32) for key in profile["camera_keys"]})
    assert images.vae_frames.shape == ((n, 3, 224, 448) if kind == "piper" else (n, 3, 384, 320))
    assert images.camera_frames[profile["semantic_camera"]].shape == (n, 3, 224, 224)
    if kind == "piper":
        processor.proprio_output_dim = 8
        with pytest.raises(RuntimeCandidateDatasetContractError, match="proprio"):
            RuntimeCandidateDatasetAdapter._validate_processor_action_space(processor, resolver)


def test_piper_artifacts_load_into_existing_causal_recollection(tmp_path):
    from fastwam.preprocessing.features import precompute
    from fastwam.preprocessing.memory import build_memory
    from fastwam.memory.runtime_candidates import RuntimeCandidateResolver
    from fastwam.memory.offline_pipeline import load_feature_cache_collection
    from fastwam.memory.candidate_cache import QueryId
    from fastwam.datasets.warm_retrospective import RetrospectiveFeatureStore
    from tests.test_unified_preprocessing import FixtureEncoder
    cfg = config(tmp_path)
    manifest(cfg, [real_episode(cfg, "a", offset=1), real_episode(cfg, "b", offset=20)])
    prepare(cfg)
    features = precompute(cfg, _test_encoder=FixtureEncoder())
    memory = build_memory(cfg)
    resolver = RuntimeCandidateResolver.from_artifacts(memory / "event_bank", memory / "train_candidates",
                                                      expected_query_split="train")
    collection = load_feature_cache_collection(features / path for path in
                         (features / "train_features.list").read_text().splitlines())
    store = RetrospectiveFeatureStore(collection, expected_split="train",
             expected_collection_sha256=resolver.query_corpus_sha256,
             expected_catalog_sha256=resolver.query_catalog_sha256, action_horizon=4, gripper_indices=(6,))
    record = collection.records[0].features
    def query(frame):
        return store.query_payload(QueryId(record.dataset_id, record.dataset_index, record.episode_index, frame))
    initial = query(0)
    assert not initial["warm_episode_mask"].any()
    assert initial["warm_future_valid"]
    final = query(len(record.proprio) - 1)
    assert not final["warm_future_valid"]
    assert not final["warm_future_semantic"].any()
