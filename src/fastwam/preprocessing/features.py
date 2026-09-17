"""Factual encoders and feature caches shared by all explicit dataset adapters."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from fastwam.datasets.lerobot.full_episode_reader import read_full_lerobot_episode
from fastwam.memory.action_contract import ActionSpaceContract
from fastwam.memory.feature_cache import FeatureCacheMetadata, save_episode_feature_cache
from fastwam.memory.feature_precompute import assemble_episode_features
from fastwam.memory.offline_pipeline import load_feature_cache_collection, validate_feature_collection_against_catalog
from .config import stage_directory
from .contracts import PreparationError, file_sha256, read_json, write_json
from .prepare import load_prepared


def decode_video(path: Path, timestamps, tolerance: float) -> np.ndarray:
    """Exact full sequential PyAV decode without a Torch/LeRobot dependency."""
    import av
    frames, actual_times = [], []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:
                raise PreparationError("Video frame lacks a factual presentation timestamp")
            actual_times.append(float(frame.pts * frame.time_base))
            frames.append(frame.to_ndarray(format="rgb24"))
    if len(frames) != len(timestamps) or not np.allclose(actual_times, timestamps, atol=tolerance, rtol=0):
        raise PreparationError("Video timestamps/frame count disagree with the audited table")
    return np.stack(frames).transpose(0, 3, 1, 2).astype(np.float32) / 255


def checkpoint_tree(path: Path) -> str:
    if not path.is_dir():
        raise PreparationError(f"Local DINO snapshot not found: {path}")
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise PreparationError("Empty encoder checkpoint directory")
    digest = sha256()
    for file in files:
        digest.update(file.relative_to(path).as_posix().encode())
        digest.update(file_sha256(file).encode())
    return digest.hexdigest()


class FactualEncoder:
    """Uses the exact FastWAM processor and existing frozen DINO/Wan encoders."""
    test_only = False

    def __init__(self, cfg: dict, prepared: Path, task_vocabulary: list[str]):
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from fastwam.memory.feature_precompute import FastWAMProcessorAdapter
        from fastwam.memory.server_feature_encoders import FastWAMImageAdapter, DinoV2FactualEncoder

        self.cfg, self.profile = cfg["encoders"], cfg["profile"]
        self.dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[self.cfg["dtype"]]
        checkpoint = Path(self.cfg["dino_checkpoint"])
        before = checkpoint_tree(checkpoint)
        data = OmegaConf.load(prepared / "data_config.yaml")
        processor = instantiate(data.train.processor)
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(prepared / "dataset_stats.json")))
        processor.eval()
        self.processor = FastWAMProcessorAdapter(processor)
        robotwin = self.profile["image_layout"] == "robotwin_3cam"
        self.images = FastWAMImageAdapter(processor, self.profile["camera_keys"],
                                         "robotwin" if robotwin else "horizontal",
                                         benchmark_profile="robotwin" if robotwin else "libero")
        self.dino = DinoV2FactualEncoder.from_pretrained(checkpoint, model_id=self.cfg["dino_model_id"],
                      revision=self.cfg["dino_revision"], device=self.cfg["device"], torch_dtype=self.dtype)
        if self.dino.hidden_size != 768:
            raise PreparationError("Current WARM retrospective semantic heads require DINO hidden_size=768")
        if checkpoint_tree(checkpoint) != before:
            raise PreparationError("DINO snapshot changed while loading")
        self.vae = None
        vae_hash = None
        if self.cfg["include_vae"]:
            from fastwam.models.wan22.helpers.loader import load_wan22_vae_only
            vae_path = Path(self.cfg["vae_checkpoint"])
            vae_hash = file_sha256(vae_path)
            self.vae = load_wan22_vae_only(device=self.cfg["device"], torch_dtype=self.dtype, vae_path=vae_path).vae
            if file_sha256(vae_path) != vae_hash:
                raise PreparationError("VAE snapshot changed while loading")
        self.contract = {"schema": "warm.feature-encoder", "version": 1,
                         "official_complete": not cfg.get("test_only", False) and not cfg["source"].get("allow_synthetic", False),
                         "implementation": "warm.preprocessing.factual.v1",
                         "data_config_sha256": file_sha256(prepared / "data_config.yaml"),
                         "normalization": {"stats_sha256": file_sha256(prepared / "dataset_stats.json")},
                         "compute": dict(self.cfg), "output_dtype": "float32",
                         "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda},
                         "dino": {"checkpoint_tree_sha256": before, "model_id": self.dino.model_id,
                                  "revision": self.dino.revision, "hidden_size": self.dino.hidden_size,
                                  "image_mean": list(self.dino.image_mean), "image_std": list(self.dino.image_std),
                                  "semantic_pool": "adaptive_avg_pool_2x2_row_major", "resize_in_encoder": False},
                         "context": {"mode": "visual-task", "task_vocabulary": task_vocabulary},
                         "vae": {"enabled": self.vae is not None, "checkpoint_sha256": vae_hash,
                                 "per_frame_singleton_time": self.vae is not None, "spatial_pool": [4, 8]},
                         "factual_gripper": {"indices": self.profile["gripper_state_indices"], "reduction": "abs_sum"}}

    def encode(self, full) -> dict[str, Any]:
        profile = self.profile
        processed = self.processor.process(full.actions, full.states, gripper_indices=profile["gripper_state_indices"])
        cameras = {key: full.images["observation.images." + key] for key in profile["camera_keys"]}
        prepared = self.images.prepare(cameras)
        dino = self.dino.encode(prepared.camera_frames[profile["semantic_camera"]], batch_size=self.cfg["dino_batch_size"])
        vae = None
        if self.vae is not None:
            from fastwam.memory.feature_precompute_vae import encode_wan22_factual_frames
            vae = encode_wan22_factual_frames(self.vae, prepared.vae_frames, device=self.cfg["device"],
                                              batch_size=self.cfg["vae_batch_size"], torch_dtype=self.dtype)
        return {"model_actions": processed.model_actions, "proprio": processed.proprio,
                "gripper": processed.gripper, "dino_cls": dino.cls, "semantic_features": dino.spatial, "vae_features": vae}


def precompute(cfg: dict, *, _test_encoder=None) -> Path:
    prepared, marker, catalog, audit = load_prepared(cfg)
    profile = cfg["profile"]
    # Test episodes do not need offline memory features for deployment. Keeping
    # them out also prevents accidental initialization from an evaluation future.
    records = [record for record in catalog.episodes if record.split in {"train", "dev"}]
    vocabulary = sorted({record.primary_task for record in records})
    if _test_encoder is not None and (not cfg.get("test_only") or not _test_encoder.test_only):
        raise PreparationError("An injected test encoder requires an explicitly test-only configuration")
    output = Path(cfg["output"]) / "features"
    with stage_directory(output) as stage:
        encoder = _test_encoder if _test_encoder is not None else FactualEncoder(cfg, prepared, vocabulary)
        grippers = tuple(profile["gripper_action_indices"])
        normalizer = ActionSpaceContract(action_dim=profile["action_dim"],
                       arm_dims=tuple(i for i in range(profile["action_dim"]) if i not in grippers),
                       gripper_dims=grippers, gripper_threshold=0.0,
                       normalization_mode="global:" + profile["normalization"],
                       normalization_stats_sha256=file_sha256(prepared / "dataset_stats.json"),
                       control_mode=profile["control_mode"], embodiment=profile["embodiment"]).to_dict()
        camera_contract = {"schema": "warm.adapter-camera.v1", "adapter": cfg["adapter"],
                           "camera_keys": profile["camera_keys"], "semantic_camera": profile["semantic_camera"],
                           "image_layout": profile["image_layout"], "control_frame": cfg["source"].get("control_frame"),
                           "tcp_frame": cfg["source"].get("tcp_frame"), "calibration_id": cfg["source"].get("calibration_id"),
                           "fps": profile["fps"], "decoder": "pyav_exact_sequential_rgb24"}
        contract_values = {"normalizer": normalizer, "encoder": encoder.contract, "camera": camera_contract}
        hashes = {"catalog_hash": catalog.content_sha256}
        for name, contract in contract_values.items():
            write_json(stage / "contracts" / (name + ".json"), contract)
            hashes[name + "_hash"] = file_sha256(stage / "contracts" / (name + ".json"))
        paths = {split: [] for split in ("train", "dev")}
        for ordinal, record in enumerate(records, 1):
            proof = audit.proof_index[(record.dataset_id, record.dataset_index, record.episode_index)]
            full = read_full_lerobot_episode(record, dataset_root=marker["roots"][record.dataset_index],
                       audit_proof=proof, camera_keys=["observation.images." + key for key in profile["camera_keys"]],
                       video_decoder=decode_video)
            episode = assemble_episode_features(dataset_id=record.dataset_id, dataset_index=record.dataset_index,
                       episode_index=record.episode_index, task_index=vocabulary.index(record.primary_task),
                       catalog_task_count=len(vocabulary), source_episode_sha256=full.source_episode_sha256,
                       **encoder.encode(full))
            relative = f"dataset_{record.dataset_index:03d}/episode_{record.episode_index:06d}.npz"
            save_episode_feature_cache(stage / relative, episode,
                  metadata=FeatureCacheMetadata.for_episode(episode, split=record.split, **hashes))
            paths[record.split].append(relative)
            print(f"encoded {ordinal}/{len(records)}: {record.dataset_id}/{record.episode_index}", flush=True)
        collection_hashes = {}
        for split, files in paths.items():
            if not files:
                continue
            collection = load_feature_cache_collection(stage / path for path in files)
            validate_feature_collection_against_catalog(collection, catalog, audit, expected_split=split)
            (stage / (split + "_features.list")).write_text("\n".join(files) + "\n", encoding="utf-8")
            collection_hashes[split] = collection.content_hash
        write_json(stage / "COMPLETE.json", {"schema": "warm.precomputed.v1", "prepared_sha256": file_sha256(prepared / "COMPLETE.json"),
                   "contracts": hashes, "collections": collection_hashes, "task_vocabulary": vocabulary,
                   "test_only": bool(cfg.get("test_only") or encoder.test_only or marker["synthetic"]),
                   "recollection": "causal prefixes of factual features; reset to empty at every online episode",
                   "test_episode_features_created": False})
    return output
