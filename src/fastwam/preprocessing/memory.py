"""Compose the existing verified WARM event-bank and candidate-cache builders."""
from __future__ import annotations

from pathlib import Path

from fastwam.memory.candidate_cache import CandidateCache, canonical_event_bank_content_hash
from fastwam.memory.event_bank import EventBank, MANIFEST_FILENAME
from fastwam.memory.event_mining import EventMiningConfig
from fastwam.memory.offline_pipeline import (
    load_feature_cache_collection, validate_feature_collection_against_catalog,
    build_event_bank_from_collection, build_candidate_cache_from_collection,
)
from .config import stage_directory
from .contracts import PreparationError, file_sha256, read_json, write_json
from .prepare import load_prepared


def build_memory(cfg: dict) -> Path:
    prepared, prepared_marker, catalog, audit = load_prepared(cfg)
    features = Path(cfg["output"]) / "features"
    marker = read_json(features / "COMPLETE.json")
    if marker["prepared_sha256"] != file_sha256(prepared / "COMPLETE.json"):
        raise PreparationError("Feature stage was built from a different prepared dataset")
    contracts = {}
    for name in ("normalizer", "encoder", "camera"):
        path = features / "contracts" / (name + ".json")
        digest = file_sha256(path)
        if marker["contracts"][name + "_hash"] != digest:
            raise PreparationError("Feature contract changed after publication")
        contracts[name] = {"file_sha256": digest, "contract": read_json(path)}
    collections, bindings = {}, {}
    for split, digest in marker["collections"].items():
        if split not in {"train", "dev"}:
            raise PreparationError("Only train/dev feature collections may enter offline memory preparation")
        lines = (features / (split + "_features.list")).read_text(encoding="utf-8").splitlines()
        collection = load_feature_cache_collection(features / line for line in lines)
        if collection.content_hash != digest:
            raise PreparationError("Feature collection changed after publication")
        collections[split] = collection
        bindings[split] = validate_feature_collection_against_catalog(collection, catalog, audit, expected_split=split)
        if any(record.features.vae_features is None for record in collection.records):
            raise PreparationError("Full WARM recollection requires factual VAE features; rerun a new version with include_vae=true")
    if len(collections["train"].records) < 2:
        raise PreparationError("Use at least two distinct train episodes for same-episode-excluded retrieval")
    profile = cfg["profile"]
    mining = EventMiningConfig(action_horizon=profile["action_horizon"], **cfg.get("memory", {}).get("mining", {}))
    output = Path(cfg["output"]) / "memory"
    with stage_directory(output) as stage:
        bank, summary, provenance = build_event_bank_from_collection(collections["train"], mining_config=mining,
                     start_mode=cfg.get("memory", {}).get("start_mode", "hybrid"), data_binding=bindings["train"])
        provenance["preprocessing"] = {"adapter": cfg["adapter"], "prepared_sha256": marker["prepared_sha256"],
                                        "test_only": marker["test_only"]}
        bank.save(stage / "event_bank", action_normalizer=contracts["normalizer"], encoder=contracts["encoder"],
                  camera_layout=contracts["camera"], provenance=provenance)
        bank = EventBank.load(stage / "event_bank")
        bank_hash = file_sha256(stage / "event_bank" / MANIFEST_FILENAME)
        content_hash = canonical_event_bank_content_hash(bank.manifest.content_hashes)
        counts = {}
        for split, collection in collections.items():
            cache = build_candidate_cache_from_collection(bank, collection, action_horizon=profile["action_horizon"],
                      query_stride=1, top_k=cfg.get("memory", {}).get("top_k", 32), query_split=split,
                      include_partial_action_queries=True)
            recipe = {"implementation": "exact_cosine_v1", "action_horizon": profile["action_horizon"],
                      "query_stride": 1, "top_k": cfg.get("memory", {}).get("top_k", 32), "query_split": split,
                      "query_frame_policy": "all_factual_states_v1", "query_data_binding": bindings[split].to_dict(),
                      "episode_exclusion": ["global_episode_identity", "source_episode_sha256", "feature_episode_sha256"]}
            cache.save(stage / (split + "_candidates"), event_bank_manifest_hash=bank_hash,
                       event_bank_content_hash=content_hash, query_corpus_hash=collection.content_hash,
                       query_key_encoder=dict(bank.manifest.encoder), build_recipe=recipe)
            restored = CandidateCache.load(stage / (split + "_candidates"))
            restored.validate_against_event_bank(bank)
            counts[split] = {"queries": len(restored), "candidates": restored.num_candidates,
                             "empty_queries": sum(not row for row in restored.candidates)}
        write_json(stage / "training_bindings.json", {"schema": "warm.training-bindings.v1",
                   "data_config": str(prepared / "data_config.yaml"), "dataset_stats": str(prepared / "dataset_stats.json"),
                   "catalog": str(prepared / "catalog.json"), "audit": str(prepared / "audit.json"),
                   "event_bank": str(output / "event_bank"), "action_horizon": profile["action_horizon"],
                   "action_dim": profile["action_dim"], "proprio_dim": profile["state_dim"],
                   "recollection_action_mode": "delta" if any(profile["delta_action_mask"]) else "absolute_target",
                   "gripper_indices": profile["gripper_action_indices"],
                   "splits": {split: {"features": str(features / (split + "_features.list")),
                                       "candidates": str(output / (split + "_candidates")),
                                       "query_corpus_sha256": collection.content_hash}
                              for split, collection in collections.items()},
                   "checkpoint_compatibility": "UNVERIFIED: match action/state heads, normalization, camera layout and runtime adapters",
                   "online_recollection_initialization": "empty; factual observations and executed actions only",
                   "test_only": marker["test_only"]})
        import yaml
        data_config = yaml.safe_load((prepared / "data_config.yaml").read_text(encoding="utf-8"))
        data_config["warm_candidates"] = {}
        for split, collection in collections.items():
            data_config["warm_candidates"]["train" if split == "train" else "val"] = {
                "bank_directory": str(output / "event_bank"),
                "candidate_directory": str(output / (split + "_candidates")),
                "catalog_path": str(prepared / "catalog.json"),
                "normalization_stats_path": str(prepared / "dataset_stats.json"),
                "audit_report_path": str(prepared / "audit.json"),
                "expected_query_corpus_sha256": collection.content_hash,
                "retrospective_feature_list": str(features / (split + "_features.list")),
                "retrospective_feature_directory": None,
                "retrospective_recent_event_capacity": 6,
                "retrospective_action_summary_capacity": 2,
                "retrospective_action_summary_chunk_size": min(10, profile["action_horizon"]),
                "retrospective_gripper_indices": profile["gripper_action_indices"],
            }
        (stage / "warm_data_config.yaml").write_text(yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8")
        write_json(stage / "COMPLETE.json", {"schema": "warm.memory-ready.v1", "events": summary.num_events,
                   "source_episodes": summary.num_source_episodes, "bank_manifest_sha256": bank_hash,
                   "bank_content_sha256": content_hash, "caches": counts, "test_only": marker["test_only"],
                   "train_only_repertoire": True, "test_episode_future_used": False})
    return output
