"""Dataset adaptation, split audit and train-only statistics; no GPU needed."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np

from fastwam.datasets.lerobot.audit import audit_lerobot_catalog, load_audit_report, write_audit_report
from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog, episode_indices_by_dataset
from .config import stage_directory
from .contracts import PreparationError, file_sha256, read_json, read_source_manifest, write_json
from .lerobot_writer import convert_episodes


def read_table(record, root: Path, proof, profile: dict) -> dict[str, np.ndarray]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    payload = (root / record.data_relpath).read_bytes()
    if sha256(payload).hexdigest() != proof.table_sha256:
        raise PreparationError("Episode table differs from the audited snapshot")
    table = pq.read_table(pa.BufferReader(payload))
    arrays = {key: np.asarray(table[key].to_pylist()) for key in
              ("action", "observation.state", "timestamp", "frame_index", "episode_index", "task_index")}
    n = record.length
    for key, dim in (("action", profile["action_dim"]), ("observation.state", profile["state_dim"])):
        if arrays[key].shape != (n, dim) or not np.isfinite(arrays[key]).all():
            raise PreparationError(f"{key} shape/values disagree with profile")
    for key in ("frame_index", "episode_index", "task_index"):
        if arrays[key].shape != (n,) or arrays[key].dtype.kind not in "iu" or np.any(arrays[key] < 0):
            raise PreparationError(f"{key} must contain nonnegative integer scalars")
    if not np.array_equal(arrays["frame_index"], np.arange(n)):
        raise PreparationError("Noncontiguous frame_index")
    if not np.all(arrays["episode_index"] == record.episode_index):
        raise PreparationError("episode_index differs from catalog")
    if (arrays["timestamp"].shape != (n,) or not np.allclose(arrays["timestamp"], np.arange(n) / record.fps, atol=1e-4, rtol=0)):
        raise PreparationError("Timestamp does not match source fps")
    if arrays["task_index"].shape != (n,) or len(np.unique(arrays["task_index"])) != 1:
        raise PreparationError("Episode task_index must be constant")
    return arrays


class Moments:
    def __init__(self):
        self.count = 0

    def add(self, rows: np.ndarray):
        rows = rows.astype(np.float64)
        count, mean = len(rows), rows.mean(axis=0)
        m2 = ((rows - mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.mean, self.m2 = mean, m2
            self.minimum, self.maximum = rows.min(axis=0), rows.max(axis=0)
        else:
            delta = mean - self.mean
            total = self.count + count
            self.m2 += m2 + delta ** 2 * self.count * count / total
            self.mean += delta * count / total
            self.minimum = np.minimum(self.minimum, rows.min(axis=0))
            self.maximum = np.maximum(self.maximum, rows.max(axis=0))
        self.count += count

    def result(self, mode: str):
        if not self.count:
            raise PreparationError("No training rows for statistics")
        if mode == "min/max":
            return {"global_min": self.minimum.tolist(), "global_max": self.maximum.tolist()}
        return {"global_mean": self.mean.tolist(), "global_std": np.sqrt(np.maximum(0, self.m2 / self.count)).tolist()}


def training_data_config(profile: dict, roots: list[str], prepared: Path, catalog: EpisodeCatalog) -> dict:
    """Ordinary RobotVideoDataset config; WARM wrapper is added by the training recipe."""
    camera_size = [240, 320] if profile["image_layout"] == "robotwin_3cam" else [224, 224]
    infos = [read_json(Path(root) / "meta/info.json") for root in roots]
    images = []
    for camera in profile["camera_keys"]:
        shapes = [info["features"]["observation.images." + camera]["shape"] for info in infos]
        if any(shape != shapes[0] for shape in shapes):
            raise PreparationError("Multiple dataset roots must agree on raw camera shapes")
        images.append({"key": camera, "raw_shape": shapes[0], "shape": [3, *camera_size]})
    shape = {"images": images, "action": [{"key": "default", "raw_shape": profile["action_dim"], "shape": profile["action_dim"]}],
             "state": [{"key": "default", "raw_shape": profile["state_dim"], "shape": profile["state_dim"]}]}
    transforms = [{"_target_": "fastwam.datasets.lerobot.transforms.image.ToTensor"},
                  {"_target_": "torchvision.transforms.Resize", "size": camera_size}]
    processor = {"_target_": "fastwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor",
                 "shape_meta": shape, "num_obs_steps": profile["action_horizon"] + 1,
                 "num_output_cameras": len(images), "action_output_dim": profile["action_dim"],
                 "proprio_output_dim": profile["state_dim"],
                 "delta_action_dim_mask": {"default": profile["delta_action_mask"]} if any(profile["delta_action_mask"]) else None,
                 "action_state_transforms": None, "use_stepwise_action_norm": False,
                 "norm_default_mode": profile["normalization"], "norm_exception_mode": None,
                 "action_state_merger": {"_target_": "fastwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign"},
                 "train_transforms": transforms, "val_transforms": transforms}
    common = {"_target_": "fastwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset",
              "dataset_dirs": roots, "shape_meta": shape, "num_frames": profile["action_horizon"] + 1,
              "action_video_freq_ratio": profile["action_video_freq_ratio"], "global_sample_stride": 1,
              "video_size": [384, 320] if profile["image_layout"] == "robotwin_3cam" else [224, 224 * len(images)],
              "concat_multi_camera": "robotwin" if profile["image_layout"] == "robotwin_3cam" else "horizontal",
              "val_set_proportion": 0.0, "strict_sample_loading": True, "skip_padding_as_possible": False,
              "episode_catalog_path": str(prepared / "catalog.json"),
              "pretrained_norm_stats": str(prepared / "dataset_stats.json"),
              "processor": processor, "context_len": 128}
    result = {}
    for split in ("train", "dev"):
        if any(record.split == split for record in catalog.episodes):
            result["train" if split == "train" else "val"] = {
                **deepcopy(common), "is_training_set": split == "train", "episode_split": split}
    return result


def prepare(cfg: dict) -> Path:
    output = Path(cfg["output"]) / "prepared"
    profile, source = cfg["profile"], cfg["source"]
    with stage_directory(output) as stage:
        write_json(stage / "config.json", cfg)
        if cfg["adapter"] == "libero_lerobot":
            roots = [Path(p) for p in source["roots"]]
            catalog = EpisodeCatalog.load(source["catalog"])
            if (profile["action_dim"], profile["state_dim"]) != (7, 8):
                raise PreparationError("The existing LIBERO input profile is action=7, state=8")
        elif cfg["adapter"] == "rmbench":
            from fastwam.datasets.rmbench.converter import RMBenchConversionConfig, convert_rmbench_dataset
            if (profile["action_dim"], profile["state_dim"], profile["fps"]) != (14, 14, 15):
                raise PreparationError("Official RMBench requires 14/14 dimensions at 15 Hz")
            options = dict(source["conversion"])
            # Preserve the existing official contract. A new adapter is not an
            # escape hatch for a shortened/faulty official benchmark release.
            if set(options) - {"source_revision", "data_revision", "rmbench_code_revision", "split_seed", "workers"}:
                raise PreparationError("RMBench only exposes revisions, split_seed and workers")
            convert_rmbench_dataset(RMBenchConversionConfig(source_root=Path(source["root"]),
                                     output_root=stage / "dataset", dataset_id=cfg["dataset_id"], **options))
            roots = [stage / "dataset"]
            catalog = EpisodeCatalog.load(roots[0] / "meta/warm_episode_catalog.json")
        else:
            if cfg["adapter"] == "robotwin2":
                from fastwam.datasets.robotwin2.source import RoboTwin2Adapter
                adapter = RoboTwin2Adapter(source, profile)
            else:
                from fastwam.real.preprocessing.piper import PiperTeleopAdapter
                adapter = PiperTeleopAdapter(source, profile)
            entries = read_source_manifest(Path(source["manifest"]))
            write_json(stage / "source_manifest.json", read_json(Path(source["manifest"])))
            catalog = convert_episodes(adapter, entries, stage / "dataset", profile, cfg["dataset_id"])
            roots = [stage / "dataset"]
        if not catalog.episodes or any(e.split not in {"train", "dev", "test"} for e in catalog.episodes):
            raise PreparationError("Use a complete preassigned catalog; no implicit random split")
        if any(e.fps != profile["fps"] for e in catalog.episodes):
            raise PreparationError("Catalog fps disagrees with profile")
        train = [e for e in catalog.episodes if e.split == "train"]
        if not train:
            raise PreparationError("At least one train episode is required")
        for split in ("train", "dev"):
            if any(e.split == split for e in catalog.episodes):
                episode_indices_by_dataset(catalog, split=split)
        if any(e.length <= profile["action_horizon"] for e in train):
            raise PreparationError("Every train episode must contain at least horizon+1 published observations")
        catalog.save(stage / "catalog.json")
        cameras = ["observation.images." + name for name in profile["camera_keys"]]
        report = audit_lerobot_catalog(catalog, roots, hash_episode_tables=True, camera_keys=cameras)
        write_audit_report(report, stage / "audit.json")
        if report["summary"]["cross_split_duplicate_count"]:
            raise PreparationError("Cross-split duplicate source content found; inspect failed-stage audit.json")
        audit = load_audit_report(stage / "audit.json")
        moments = {key: Moments() for key in ("action", "state")}
        train_sources = []
        for record in catalog.episodes:
            proof = audit.proof_index[(record.dataset_id, record.dataset_index, record.episode_index)]
            arrays = read_table(record, roots[record.dataset_index], proof, profile)
            if record.split == "train":
                moments["action"].add(arrays["action"])
                moments["state"].add(arrays["observation.state"])
                train_sources.append({"dataset_index": record.dataset_index, "episode_index": record.episode_index,
                                      "source_episode_sha256": proof.source_episode_sha256})
        shared_gripper_range = cfg["adapter"] == "piper_teleop"
        if shared_gripper_range:
            # The retrospective timing adapter compares the factual start
            # gripper with commanded grippers. They represent the same width
            # in meters and must use the same affine map, even with tracking
            # error. Fit the union using train data only, never nominal limits.
            low = min(moments["action"].minimum[6], moments["state"].minimum[6])
            high = max(moments["action"].maximum[6], moments["state"].maximum[6])
            for value in moments.values():
                value.minimum[6], value.maximum[6] = low, high
        stats = {key: {"default": value.result(profile["normalization"])} for key, value in moments.items()}
        stats.update(num_episodes=len(train), num_transition=moments["action"].count)
        write_json(stage / "dataset_stats.json", stats)
        write_json(stage / "train_stats_manifest.json", {"schema": "warm.adapter-train-stats.v1", "split": "train",
                   "catalog_sha256": catalog.content_sha256, "audit_report_sha256": audit.report_sha256,
                   "stats_sha256": file_sha256(stage / "dataset_stats.json"), "mode": profile["normalization"],
                   "std_ddof": 0, "rows": "all command-bearing train rows", "episodes": train_sources,
                   "shared_gripper_action_state_train_range": shared_gripper_range})
        future_roots = [str(output / p.relative_to(stage)) if p.is_relative_to(stage) else str(p) for p in roots]
        # Resolve image metadata before moving staging. Serialize final paths only.
        data_config = training_data_config(profile, [str(p) for p in roots], output, catalog)
        for dataset in data_config.values():
            dataset["dataset_dirs"] = future_roots
        import yaml
        (stage / "data_config.yaml").write_text(yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8")
        tracked_files = ["config.json", "catalog.json", "audit.json", "dataset_stats.json", "train_stats_manifest.json", "data_config.yaml"]
        tracked_files += [name for name in ("source_manifest.json", "dataset/meta/conversion.json", "dataset/meta/rmbench_conversion_manifest.json")
                          if (stage / name).is_file()]
        manifest = {"schema": "warm.prepared.v1", "adapter": cfg["adapter"], "dataset_id": cfg["dataset_id"],
                    "roots": future_roots, "profile": profile,
                    "catalog_sha256": catalog.content_sha256, "audit_report_sha256": audit.report_sha256,
                    "synthetic": bool(source.get("allow_synthetic", False)),
                    "files": {name: file_sha256(stage / name) for name in tracked_files}}
        write_json(stage / "COMPLETE.json", manifest)
    return output


def load_prepared(cfg: dict):
    root = Path(cfg["output"]) / "prepared"
    marker = read_json(root / "COMPLETE.json")
    if read_json(root / "config.json") != cfg:
        raise PreparationError("Configuration changed; choose a new output version and rerun prepare")
    for name, digest in marker["files"].items():
        if file_sha256(root / name) != digest:
            raise PreparationError(f"Prepared artifact changed: {name}")
    return root, marker, EpisodeCatalog.load(root / "catalog.json"), load_audit_report(root / "audit.json")
