#!/usr/bin/env python3
"""Publish pilot_20hz_joint without recomputing VAE or DINO features.

Action rows become absolute joint_target_rad (6) plus command gripper_width_m.
Proprio stays the existing TCP state. Image caches are copied and only
model_actions are renormalized. The TCP tree at pilot_20hz is not modified.

delta_action_mask stays true for the first six channels. Those channels are
absolute joint targets, but proprio is still TCP, so recollection must not
subtract start_proprio from the action (absolute_target mode). The mask only
zeros padded steps, which the action loss already ignores.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fastwam.datasets.lerobot.audit import (
    audit_lerobot_catalog,
    load_audit_report,
    write_audit_report,
)
from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.memory.action_contract import ActionSpaceContract
from fastwam.memory.bank_builder import EpisodeFeatures
from fastwam.memory.feature_cache import (
    FeatureCacheMetadata,
    load_episode_feature_cache,
    save_episode_feature_cache,
)
from fastwam.memory.feature_precompute import FastWAMProcessorAdapter
from fastwam.memory.offline_pipeline import (
    load_feature_cache_collection,
    validate_feature_collection_against_catalog,
)
from fastwam.preprocessing.contracts import file_sha256, write_json
from fastwam.preprocessing.memory import build_memory
from fastwam.preprocessing.prepare import Moments, training_data_config
from fastwam.real.preprocessing.piper import base_rotation_delta

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "real/piper/processed/pilot_20hz"
DST = REPO / "real/piper/processed/pilot_20hz_joint"
RAW = REPO / "tmp/WARM_real/pilot_20hz"
CONTROL_MODE = "absolute_joint_target_rad_plus_absolute_gripper_width_m"
ACTION_PROVENANCE = "accepted_joint_target_rad_plus_command_gripper_width_m"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def episode_actions(relative: str) -> tuple[np.ndarray, np.ndarray]:
    root = RAW / relative
    observations = read_jsonl(root / "observations.jsonl")
    commands = read_jsonl(root / "commands.jsonl")
    tcp_rows, joint_rows = [], []
    for obs, command in zip(observations[:-1], commands, strict=True):
        robot = obs["robot"]
        xyz = np.asarray(robot["tcp_position_m"], dtype=np.float64)
        quat = robot["tcp_quaternion_xyzw"]
        joints = np.asarray(command["joint_target_rad"], dtype=np.float64)
        grip = float(command["gripper_width_m"])
        if joints.shape != (6,) or not np.isfinite(joints).all() or not np.isfinite(grip):
            raise RuntimeError(f"invalid joint command in {relative}")
        tcp_rows.append([
            *(np.asarray(command["tcp_position_m"], dtype=np.float64) - xyz),
            *base_rotation_delta(command["tcp_quaternion_xyzw"], quat),
            grip,
        ])
        joint_rows.append(np.r_[joints, grip])
    return np.asarray(tcp_rows, dtype=np.float32), np.asarray(joint_rows, dtype=np.float32)


def table_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["action", "observation.state"])
    action = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    return action, state


def write_action_column(source: Path, dest: Path, actions: np.ndarray) -> None:
    table = pq.read_table(source)
    values = pa.array(np.ascontiguousarray(actions, dtype=np.float32).reshape(-1))
    column = pa.FixedSizeListArray.from_arrays(values, 7)
    if column.type != table.schema.field("action").type:
        raise RuntimeError(f"action type changed for {source.name}")
    index = table.schema.get_field_index("action")
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.set_column(index, "action", column), dest)


def link_tree(source: Path, dest: Path) -> None:
    for directory, _, names in os.walk(source):
        relative = Path(directory).relative_to(source)
        target = dest / relative
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            os.link(Path(directory) / name, target / name)


def vector_stats(array: np.ndarray) -> dict:
    return {
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "count": [int(array.shape[0])],
    }


def make_processor(data_config: Path, stats_path: Path) -> FastWAMProcessorAdapter:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    data = OmegaConf.load(data_config)
    processor = instantiate(data.train.processor)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
    processor.eval()
    return FastWAMProcessorAdapter(processor)


def replay_matches(adapter: FastWAMProcessorAdapter, actions: np.ndarray, states: np.ndarray, cached) -> None:
    processed = adapter.process(
        {"default": actions}, {"default": states}, gripper_indices=(6,)
    )
    if not np.array_equal(processed.model_actions, cached.features.model_actions):
        delta = np.max(np.abs(processed.model_actions - cached.features.model_actions))
        raise RuntimeError(f"TCP model_actions replay mismatch, max abs {delta}")
    if not np.array_equal(processed.proprio, cached.features.proprio):
        delta = np.max(np.abs(processed.proprio - cached.features.proprio))
        raise RuntimeError(f"proprio replay mismatch, max abs {delta}")


def main() -> None:
    if DST.exists():
        raise FileExistsError(f"refusing to overwrite {DST}")
    if not (SRC / "memory" / "COMPLETE.json").is_file():
        raise FileNotFoundError("source pilot_20hz memory is incomplete")

    prepared_src = SRC / "prepared"
    catalog = EpisodeCatalog.load(prepared_src / "catalog.json")
    manifest = {row["id"]: row for row in read_json(prepared_src / "source_manifest.json")["episodes"]}
    conversion = read_json(prepared_src / "dataset" / "meta" / "conversion.json")
    by_index = {int(row["episode_index"]): row for row in conversion["episodes"]}
    if set(by_index) != {record.episode_index for record in catalog.episodes}:
        raise RuntimeError("conversion episodes do not match the catalog")

    joint_by_index: dict[int, np.ndarray] = {}
    print("checking TCP alignment and reading joint targets", flush=True)
    tcp_adapter = make_processor(prepared_src / "data_config.yaml", prepared_src / "dataset_stats.json")
    for record in catalog.episodes:
        row = by_index[record.episode_index]
        source = manifest[row["source_id"]]
        if source["split"] != record.split or source["path"] != row["source_id"].replace("__", "/", 1):
            raise RuntimeError(f"source mapping mismatch at episode {record.episode_index}")
        tcp, joints = episode_actions(source["path"])
        parquet = prepared_src / "dataset" / record.data_relpath
        stored_action, stored_state = table_arrays(parquet)
        if tcp.shape != stored_action.shape or not np.array_equal(tcp, stored_action):
            raise RuntimeError(f"TCP replay does not match parquet episode {record.episode_index}")
        if joints.shape != stored_action.shape or not np.array_equal(joints[:, 6], stored_action[:, 6]):
            raise RuntimeError(f"gripper width changed at episode {record.episode_index}")
        cached = load_episode_feature_cache(
            SRC / "features" / f"dataset_{record.dataset_index:03d}" / f"episode_{record.episode_index:06d}.npz"
        )
        replay_matches(tcp_adapter, stored_action, stored_state, cached)
        joint_by_index[record.episode_index] = joints
        row["annotations"]["action_provenance"] = ACTION_PROVENANCE
        row["annotations"]["rotation_composition"] = (
            "proprio remains tcp xyz+rotvec+width; action is absolute joint_target_rad plus gripper_width_m"
        )
    print(f"aligned {len(joint_by_index)} episodes; TCP features replay exactly", flush=True)

    old_stats = read_json(prepared_src / "dataset_stats.json")
    moments = Moments()
    for record in catalog.episodes:
        if record.split == "train":
            moments.add(joint_by_index[record.episode_index])
    if moments.count != int(old_stats["num_transition"]) or sum(record.split == "train" for record in catalog.episodes) != 20:
        raise RuntimeError(f"unexpected train rows: {moments.count}")
    action_min = moments.minimum.copy()
    action_max = moments.maximum.copy()
    state_min = np.asarray(old_stats["state"]["default"]["global_min"], dtype=np.float64)
    state_max = np.asarray(old_stats["state"]["default"]["global_max"], dtype=np.float64)
    low = min(float(action_min[6]), float(state_min[6]))
    high = max(float(action_max[6]), float(state_max[6]))
    state_changed = low != float(state_min[6]) or high != float(state_max[6])
    action_min[6], action_max[6] = low, high
    state_block = old_stats["state"]
    if state_changed:
        state_min = state_min.copy()
        state_max = state_max.copy()
        state_min[6], state_max[6] = low, high
        state_block = {"default": {"global_min": state_min.tolist(), "global_max": state_max.tolist()}}
    stats = {
        "action": {"default": {"global_min": action_min.tolist(), "global_max": action_max.tolist()}},
        "num_episodes": 20,
        "num_transition": int(moments.count),
        "state": state_block,
    }
    print("joint action min", stats["action"]["default"]["global_min"], flush=True)
    print("joint action max", stats["action"]["default"]["global_max"], flush=True)
    print("state gripper range changed", state_changed, flush=True)

    dataset = DST / "prepared" / "dataset"
    link_tree(prepared_src / "dataset" / "videos", dataset / "videos")
    for record in catalog.episodes:
        source = prepared_src / "dataset" / record.data_relpath
        write_action_column(source, dataset / record.data_relpath, joint_by_index[record.episode_index])
    meta_src = prepared_src / "dataset" / "meta"
    meta = dataset / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "warm_episode_catalog.json"):
        shutil.copy2(meta_src / name, meta / name)
    stats_rows = []
    for line in (meta_src / "episodes_stats.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["stats"]["action"] = vector_stats(joint_by_index[int(row["episode_index"])])
        stats_rows.append(row)
    with (meta / "episodes_stats.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in stats_rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    conversion["alignment"] = (
        "N command-bearing rows; action is absolute joint_target_rad plus gripper_width_m; "
        "memory uses N-1 factual transitions; no padding"
    )
    conversion["action_format"] = CONTROL_MODE
    write_json(meta / "conversion.json", conversion)
    shutil.copy2(prepared_src / "catalog.json", DST / "prepared" / "catalog.json")
    shutil.copy2(prepared_src / "source_manifest.json", DST / "prepared" / "source_manifest.json")

    prepared = DST / "prepared"
    catalog = EpisodeCatalog.load(prepared / "catalog.json")
    cameras = ["observation.images." + name for name in ("external", "wrist")]
    report = audit_lerobot_catalog(
        catalog, [dataset], hash_episode_tables=True, camera_keys=cameras
    )
    write_audit_report(report, prepared / "audit.json")
    audit = load_audit_report(prepared / "audit.json")
    write_json(prepared / "dataset_stats.json", stats)

    profile = read_json(prepared_src / "config.json")["profile"]
    profile = dict(profile)
    profile["control_mode"] = CONTROL_MODE
    data_config = training_data_config(profile, [str(dataset)], prepared, catalog)
    for split_cfg in data_config.values():
        split_cfg["dataset_dirs"] = [str(dataset)]
        split_cfg["episode_catalog_path"] = str(prepared / "catalog.json")
        split_cfg["pretrained_norm_stats"] = str(prepared / "dataset_stats.json")
    import yaml
    (prepared / "data_config.yaml").write_text(
        yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8"
    )

    cfg = read_json(prepared_src / "config.json")
    cfg["output"] = str(DST)
    cfg["dataset_id"] = "piper_pilot_20hz"
    cfg["profile"] = profile
    write_json(prepared / "config.json", cfg)
    cfg = read_json(prepared / "config.json")
    train_sources = []
    for record in catalog.episodes:
        if record.split != "train":
            continue
        proof = audit.proof_index[(record.dataset_id, record.dataset_index, record.episode_index)]
        train_sources.append({
            "dataset_index": record.dataset_index,
            "episode_index": record.episode_index,
            "source_episode_sha256": proof.source_episode_sha256,
        })
    write_json(prepared / "train_stats_manifest.json", {
        "schema": "warm.adapter-train-stats.v1",
        "split": "train",
        "catalog_sha256": catalog.content_sha256,
        "audit_report_sha256": audit.report_sha256,
        "stats_sha256": file_sha256(prepared / "dataset_stats.json"),
        "mode": profile["normalization"],
        "std_ddof": 0,
        "rows": "all command-bearing train rows",
        "episodes": train_sources,
        "shared_gripper_action_state_train_range": True,
        "action_format": CONTROL_MODE,
    })
    tracked = [
        "config.json", "catalog.json", "audit.json", "dataset_stats.json",
        "train_stats_manifest.json", "data_config.yaml", "source_manifest.json",
        "dataset/meta/conversion.json",
    ]
    write_json(prepared / "COMPLETE.json", {
        "schema": "warm.prepared.v1",
        "adapter": cfg["adapter"],
        "dataset_id": cfg["dataset_id"],
        "roots": [str(dataset)],
        "profile": profile,
        "catalog_sha256": catalog.content_sha256,
        "audit_report_sha256": audit.report_sha256,
        "synthetic": False,
        "action_format": CONTROL_MODE,
        "files": {name: file_sha256(prepared / name) for name in tracked},
    })

    features = DST / "features"
    contracts = features / "contracts"
    contracts.mkdir(parents=True)
    shutil.copy2(SRC / "features" / "contracts" / "camera.json", contracts / "camera.json")
    encoder = read_json(SRC / "features" / "contracts" / "encoder.json")
    encoder["data_config_sha256"] = file_sha256(prepared / "data_config.yaml")
    encoder["normalization"]["stats_sha256"] = file_sha256(prepared / "dataset_stats.json")
    write_json(contracts / "encoder.json", encoder)
    normalizer = ActionSpaceContract(
        action_dim=7,
        arm_dims=(0, 1, 2, 3, 4, 5),
        gripper_dims=(6,),
        gripper_threshold=0.0,
        normalization_mode="global:" + profile["normalization"],
        normalization_stats_sha256=file_sha256(prepared / "dataset_stats.json"),
        control_mode=CONTROL_MODE,
        embodiment=profile["embodiment"],
    ).to_dict()
    write_json(contracts / "normalizer.json", normalizer)
    hashes = {
        "catalog_hash": catalog.content_sha256,
        "normalizer_hash": file_sha256(contracts / "normalizer.json"),
        "encoder_hash": file_sha256(contracts / "encoder.json"),
        "camera_hash": file_sha256(contracts / "camera.json"),
    }
    joint_adapter = make_processor(prepared / "data_config.yaml", prepared / "dataset_stats.json")
    paths = {"train": [], "dev": []}
    print("rewriting model_actions", flush=True)
    for record in catalog.episodes:
        if record.split not in paths:
            continue
        relative = f"dataset_{record.dataset_index:03d}/episode_{record.episode_index:06d}.npz"
        cached = load_episode_feature_cache(SRC / "features" / relative)
        parquet_action, parquet_state = table_arrays(dataset / record.data_relpath)
        processed = joint_adapter.process(
            {"default": parquet_action}, {"default": parquet_state}, gripper_indices=(6,)
        )
        if state_changed:
            proprio = processed.proprio
        else:
            if not np.array_equal(processed.proprio, cached.features.proprio):
                delta = float(np.max(np.abs(processed.proprio - cached.features.proprio)))
                raise RuntimeError(f"proprio changed without a stats change, max abs {delta}")
            proprio = cached.features.proprio
        if not np.array_equal(processed.gripper, cached.features.gripper):
            raise RuntimeError(f"gripper feature changed at episode {record.episode_index}")
        proof = audit.proof_index[(record.dataset_id, record.dataset_index, record.episode_index)]
        episode = EpisodeFeatures(
            dataset_id=cached.features.dataset_id,
            dataset_index=cached.features.dataset_index,
            episode_index=cached.features.episode_index,
            task_index=cached.features.task_index,
            source_episode_sha256=proof.source_episode_sha256,
            model_actions=processed.model_actions,
            proprio=proprio,
            gripper=cached.features.gripper,
            context_keys=cached.features.context_keys,
            semantic_features=cached.features.semantic_features,
            vae_features=cached.features.vae_features,
        )
        save_episode_feature_cache(
            features / relative,
            episode,
            metadata=FeatureCacheMetadata.for_episode(episode, split=record.split, **hashes),
        )
        paths[record.split].append(relative)
        print(f"features {record.episode_index}", flush=True)

    old_complete = read_json(SRC / "features" / "COMPLETE.json")
    collection_hashes = {}
    for split, files in paths.items():
        (features / f"{split}_features.list").write_text("\n".join(files) + "\n", encoding="utf-8")
        collection = load_feature_cache_collection(features / path for path in files)
        validate_feature_collection_against_catalog(collection, catalog, audit, expected_split=split)
        collection_hashes[split] = collection.content_hash
    write_json(features / "COMPLETE.json", {
        "schema": "warm.precomputed.v1",
        "prepared_sha256": file_sha256(prepared / "COMPLETE.json"),
        "contracts": hashes,
        "collections": collection_hashes,
        "task_vocabulary": old_complete["task_vocabulary"],
        "test_only": False,
        "recollection": old_complete["recollection"],
        "test_episode_features_created": False,
        "image_features": "copied from pilot_20hz; model_actions renormalized to joint targets",
    })
    print("building memory", flush=True)
    build_memory(cfg)
    memory = read_json(DST / "memory" / "COMPLETE.json")
    print("memory events", memory["events"], "caches", memory["caches"], flush=True)


if __name__ == "__main__":
    main()
