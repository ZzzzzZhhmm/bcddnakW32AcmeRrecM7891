"""Contract-bound DINO truth reader without loading a WARM policy or video model."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .resumable_pairs import file_sha


def bind_reader_files(runtime, contract):
    """Check exact original processor/encoder/camera/statistics before loading."""
    fields = {
        "warm_encoder_contract_path": "encoder_contract_sha256",
        "warm_camera_contract_path": "camera_contract_sha256",
        "warm_m1_data_config_path": "m1_data_config_sha256",
        "dataset_stats_path": "normalization_stats_sha256",
    }
    result = {}
    for key, digest_key in fields.items():
        path = Path(runtime[key]).resolve(strict=True)
        digest = file_sha(path)
        if digest != contract[digest_key]:
            raise ValueError("frozen effect reader input differs: " + key)
        result[key] = dict(path=str(path), sha256=digest)
    return result


class FrozenFactualEffectReader:
    def __init__(self, image_adapter, dino, processor_keys, semantic_camera, *, batch_size, receipt):
        self.image_adapter = image_adapter
        self.dino = dino
        self.processor_keys = tuple(processor_keys)
        self.semantic_camera = semantic_camera
        self.batch_size = batch_size
        self.receipt = receipt

    def encode(self, observation):
        # Reuse both unchanged official-observation mapping and M1 quantization.
        from experiments.rmbench.warm_policy.policy_core import factual_cameras
        from fastwam.memory.online_retrieval import FrozenDinoOnlineRetriever
        raw, _ = FrozenDinoOnlineRetriever._prepare_raw_cameras(
            SimpleNamespace(_processor_camera_keys=self.processor_keys), factual_cameras(observation))
        prepared = self.image_adapter.prepare(raw)
        features = self.dino.encode(prepared.camera_frames[self.semantic_camera], batch_size=self.batch_size)
        value = np.asarray(features.spatial[0], dtype=np.float32)
        if value.shape != (4, 768) or not np.isfinite(value).all():
            raise ValueError("original four-token finite DINO truth required")
        return value.copy()

    @classmethod
    def from_source_runtime(cls, runtime, contract):
        binding = bind_reader_files(runtime, contract)
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from fastwam.memory.manifest import sha256_canonical_json, sha256_path_tree
        from fastwam.memory.online_retrieval import validate_online_encoder_contract, validate_online_camera_contract, _validate_dino_instance
        from fastwam.memory.processor_contract import load_m1_data_config, validate_processor_instance
        from fastwam.memory.runtime_fingerprint import current_encoder_runtime
        from fastwam.memory.server_feature_encoders import DinoV2FactualEncoder, FastWAMImageAdapter
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        encoder = json.loads(Path(runtime["warm_encoder_contract_path"]).read_text())
        camera = json.loads(Path(runtime["warm_camera_contract_path"]).read_text())
        _, _, dtype, device = validate_online_encoder_contract(encoder)
        _, processor_keys, semantic_camera, concat_mode = validate_online_camera_contract(camera, benchmark_profile="robotwin")
        if device != runtime["warm_dino_device"] or sha256_canonical_json(current_encoder_runtime(device)) != contract["encoder_runtime_sha256"]:
            raise ValueError("effect reader runtime differs from the original online contract")
        recipe, digest = load_m1_data_config(runtime["warm_m1_data_config_path"], profile="robotwin")
        if digest != contract["m1_data_config_sha256"]:
            raise ValueError("M1 recipe changed")
        config = OmegaConf.create({"data": OmegaConf.load(runtime["warm_m1_data_config_path"])})
        OmegaConf.resolve(config)
        processor = instantiate(config.data.train.processor).eval()
        processor.set_normalizer_from_stats(load_dataset_stats_from_json(runtime["dataset_stats_path"]))
        validate_processor_instance(processor, recipe)
        dino_path = Path(runtime["warm_dino_checkpoint_path"]).resolve(strict=True)
        before, count = sha256_path_tree(dino_path)
        if before != contract["dino_checkpoint_tree_sha256"] or count != contract["dino_checkpoint_file_count"]:
            raise ValueError("frozen DINO checkpoint tree differs")
        dc = encoder["dino"]
        dino = DinoV2FactualEncoder.from_pretrained(dino_path,model_id=dc["model_id"],revision=dc["revision"],
            device=device,torch_dtype={"float32":torch.float32,"float16":torch.float16,"bfloat16":torch.bfloat16}[dtype],
            expected_image_size=tuple(dc["image_size"]),register_token_count=dc["register_token_count"])
        _validate_dino_instance(dino,dc,expected_compute_dtype=dtype,expected_compute_device=device)
        after, after_count = sha256_path_tree(dino_path)
        if (before,count) != (after,after_count) or binding != bind_reader_files(runtime,contract):
            raise ValueError("effect reader artifacts changed during loading")
        adapter = FastWAMImageAdapter(processor,processor_keys,concat_mode,benchmark_profile="robotwin")
        receipt = dict(inputs=binding,dino_tree_sha256=before,dino_file_count=count,compute_dtype=dtype,compute_device=device,
            original_observation_mapping=True,original_m1_preprocessing=True,frozen_encoder=True,policy_loaded=False,
            original_saved_retrieval_tensor_available=False,exact_saved_retrieval_tensor_match_claimed=False)
        return cls(adapter,dino,processor_keys,semantic_camera,batch_size=runtime["warm_dino_batch_size"],receipt=receipt)
