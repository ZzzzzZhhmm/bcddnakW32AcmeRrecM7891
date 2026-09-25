"""Dataset bridge for immutable WARM candidate-cache rows.

The event bank and candidate cache are built on raw, stride-one frame
indices.  This adapter keeps that identity contract intact at training time:
it derives an exact :class:`~fastwam.memory.candidate_cache.QueryId` from the
sample provenance and the immutable episode catalog, resolves the precomputed
row, and attaches fixed-width CPU tensors that PyTorch's default collator can
stack without special handling.

Missing cache rows are fail-closed for every factual, non-padded sample.  A
missing row may become an explicit all-masked null row only when the sample's
padding masks and catalog length both prove that it is a replicated episode
tail for which no fixed-horizon query exists.
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from fastwam.datasets.lerobot.episode_catalog import EpisodeCatalog, EpisodeRecord
from fastwam.datasets.lerobot.audit import load_audit_report
from fastwam.memory.candidate_cache import QueryId
from fastwam.memory.manifest import sha256_file
from fastwam.memory.oracle_metrics import ActionDistanceConfig, action_distance
from fastwam.memory.runtime_candidates import RuntimeCandidateResolver
from fastwam.real.preprocessing.contract import (
    PIPER_CONTROL_MODE, PIPER_EMBODIMENT, PIPER_IMAGE_SIGNATURE,
    PIPER_JOINT_CONTROL_MODE,
)


WARM_CANDIDATE_MU = "warm_candidate_mu"
WARM_CANDIDATE_MASK = "warm_candidate_mask"
WARM_CANDIDATE_SCORE = "warm_candidate_score"
WARM_CANDIDATE_EVENT_INDEX = "warm_candidate_event_index"
WARM_ORACLE_CANDIDATE_INDEX = "warm_oracle_candidate_index"
WARM_QUERY_SPLIT = "warm_query_split"
WARM_CANDIDATE_FIELDS = (
    WARM_CANDIDATE_MU,
    WARM_CANDIDATE_MASK,
    WARM_CANDIDATE_SCORE,
    WARM_CANDIDATE_EVENT_INDEX,
    WARM_ORACLE_CANDIDATE_INDEX,
    WARM_QUERY_SPLIT,
)

_PADDING_FIELDS = (
    "action_is_pad",
    "image_is_pad",
    "proprio_is_pad",
)


class RuntimeCandidateDatasetError(ValueError):
    """Base class for a dataset/candidate runtime-contract violation."""


class RuntimeCandidateDatasetContractError(RuntimeCandidateDatasetError):
    """Raised when artifacts, catalog, or sample shapes do not agree."""


class RuntimeCandidateMissingRowError(KeyError):
    """Raised when a factual sample has no exact candidate-cache row."""


def _nonnegative_scalar_int(value: object, *, field: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeCandidateDatasetContractError(
                f"{field} must be an integer scalar"
            )
        value = value.item()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise RuntimeCandidateDatasetContractError(
                f"{field} must be an integer scalar"
            )
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RuntimeCandidateDatasetContractError(
            f"{field} must be an integer scalar"
        )
    result = int(value)
    if result < 0:
        raise RuntimeCandidateDatasetContractError(
            f"{field} must be non-negative"
        )
    return result


def _positive_scalar_int(value: object, *, field: str) -> int:
    result = _nonnegative_scalar_int(value, field=field)
    if result <= 0:
        raise RuntimeCandidateDatasetContractError(f"{field} must be positive")
    return result


def _bool_padding_vector(sample: Mapping[str, Any], field: str) -> torch.Tensor:
    if field not in sample:
        raise RuntimeCandidateDatasetContractError(
            f"sample is missing explicit padding field {field!r}"
        )
    value = sample[field]
    if not isinstance(value, torch.Tensor):
        raise RuntimeCandidateDatasetContractError(
            f"sample {field} must be a torch.Tensor"
        )
    if value.device.type != "cpu" or value.dtype != torch.bool or value.ndim != 1:
        raise RuntimeCandidateDatasetContractError(
            f"sample {field} must be a CPU bool tensor with shape [T]"
        )
    if value.numel() <= 0:
        raise RuntimeCandidateDatasetContractError(
            f"sample {field} must be non-empty"
        )
    result = value.detach().clone().contiguous()
    true_positions = torch.nonzero(result, as_tuple=False).flatten()
    if true_positions.numel():
        first = int(true_positions[0].item())
        if not bool(torch.all(result[first:]).item()):
            raise RuntimeCandidateDatasetContractError(
                f"sample {field} padding must be a contiguous suffix"
            )
    return result


class RuntimeCandidateDatasetAdapter(torch.utils.data.Dataset):
    """Attach one verified, fixed-K WARM candidate row to every sample.

    The wrapped dataset is expected to be ``RobotVideoDataset`` (or an object
    exposing the same provenance, padding, and ``lerobot_dataset`` contract).
    Attribute access delegates to the wrapped object, while
    ``.lerobot_dataset`` retains the exact underlying object expected by the
    existing Trainer validation path.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        resolver: RuntimeCandidateResolver,
        catalog: EpisodeCatalog | str | Path,
        normalization_stats_path: str | Path | None = None,
        audit_report_path: str | Path | None = None,
    ) -> None:
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise TypeError("dataset must be a torch Dataset")
        if not isinstance(resolver, RuntimeCandidateResolver):
            raise TypeError("resolver must be RuntimeCandidateResolver")
        if isinstance(catalog, (str, Path)):
            catalog = EpisodeCatalog.load(catalog)
        if not isinstance(catalog, EpisodeCatalog):
            raise TypeError("catalog must be EpisodeCatalog or a catalog path")
        if catalog.content_sha256 != resolver.query_catalog_sha256:
            raise RuntimeCandidateDatasetContractError(
                "episode catalog does not match the candidate-cache data binding"
            )
        if resolver.query_stride != 1:
            raise RuntimeCandidateDatasetContractError(
                "runtime dataset candidates require query_stride=1"
            )
        if (normalization_stats_path is None) != (audit_report_path is None):
            raise RuntimeCandidateDatasetContractError(
                "normalization_stats_path and audit_report_path must be provided "
                "together, or both omitted for an explicit synthetic fixture"
            )

        lerobot_dataset = getattr(dataset, "lerobot_dataset", dataset)
        if not hasattr(lerobot_dataset, "global_sample_stride"):
            raise RuntimeCandidateDatasetContractError(
                "wrapped dataset does not expose global_sample_stride"
            )
        stride = _positive_scalar_int(
            getattr(lerobot_dataset, "global_sample_stride"),
            field="global_sample_stride",
        )
        if stride != 1:
            raise RuntimeCandidateDatasetContractError(
                "WARM QueryId alignment requires global_sample_stride=1"
            )
        if not hasattr(lerobot_dataset, "action_size"):
            raise RuntimeCandidateDatasetContractError(
                "wrapped dataset does not expose action_size"
            )
        action_size = _positive_scalar_int(
            getattr(lerobot_dataset, "action_size"), field="action_size"
        )
        if action_size != resolver.action_horizon:
            raise RuntimeCandidateDatasetContractError(
                "wrapped dataset action_size does not match candidate action horizon"
            )
        strict_sample_loading = getattr(
            dataset,
            "strict_sample_loading",
            getattr(lerobot_dataset, "strict_sample_loading", None),
        )
        if strict_sample_loading is not None and not bool(strict_sample_loading):
            raise RuntimeCandidateDatasetContractError(
                "WARM training requires strict_sample_loading=true so failed "
                "reads cannot substitute a different sample"
            )
        skip_padding = getattr(
            dataset,
            "skip_padding_as_possible",
            getattr(lerobot_dataset, "skip_padding_as_possible", None),
        )
        if skip_padding is not None and bool(skip_padding):
            raise RuntimeCandidateDatasetContractError(
                "WARM training requires skip_padding_as_possible=false so "
                "dataset indices remain stable"
            )

        descriptors = tuple(
            sorted(catalog.datasets, key=lambda item: item.dataset_index)
        )
        if tuple(item.dataset_index for item in descriptors) != tuple(
            range(len(descriptors))
        ):
            raise RuntimeCandidateDatasetContractError(
                "catalog dataset_index values must be contiguous from zero"
            )
        dataset_ids = {
            descriptor.dataset_index: descriptor.dataset_id
            for descriptor in descriptors
        }
        records = {
            (record.dataset_index, record.episode_index): record
            for record in catalog.episodes
        }

        self._validate_selected_episodes(
            lerobot_dataset,
            catalog,
            query_split=resolver.query_split,
            task_allowlist=getattr(
                lerobot_dataset, "episode_task_allowlist", None
            ),
            require_introspection=normalization_stats_path is not None,
        )
        if normalization_stats_path is not None:
            control_mode = resolver.action_space.control_mode
            expected_video = {
                "libero_delta_eef_axis_angle_plus_gripper": (
                    (224, 448),
                    "horizontal",
                ),
                "robotwin_bimanual_qpos_plus_grippers": (
                    (384, 320),
                    "robotwin",
                ),
                PIPER_CONTROL_MODE: ((224, 448), "horizontal"),
                PIPER_JOINT_CONTROL_MODE: ((224, 448), "horizontal"),
            }.get(control_mode)
            if expected_video is None:
                raise RuntimeCandidateDatasetContractError(
                    f"unsupported formal WARM control mode {control_mode!r}"
                )
            if (
                tuple(getattr(dataset, "video_size", ())) != expected_video[0]
                or getattr(dataset, "concat_multi_camera", None)
                != expected_video[1]
                or int(getattr(dataset, "num_frames", -1))
                != resolver.action_horizon + 1
            ):
                raise RuntimeCandidateDatasetContractError(
                    "formal WARM dataset video layout does not match the "
                    f"{control_mode!r} action contract"
                )
            stats_path = Path(normalization_stats_path).expanduser().resolve()
            if not stats_path.is_file():
                raise RuntimeCandidateDatasetContractError(
                    f"normalization stats file does not exist: {stats_path}"
                )
            try:
                stats_sha256 = sha256_file(stats_path)
            except OSError as exc:
                raise RuntimeCandidateDatasetContractError(
                    f"cannot hash normalization stats file: {stats_path}"
                ) from exc
            if stats_sha256 != resolver.action_space.normalization_stats_sha256:
                raise RuntimeCandidateDatasetContractError(
                    "normalization stats file does not match the WARM action space"
                )
            configured_stats = getattr(dataset, "pretrained_norm_stats_path", None)
            configured_hash = getattr(
                dataset, "pretrained_norm_stats_sha256", None
            )
            if (
                configured_stats is None
                or Path(configured_stats).expanduser().resolve() != stats_path
                or configured_hash != stats_sha256
            ):
                raise RuntimeCandidateDatasetContractError(
                    "wrapped dataset did not load the exact contract-bound "
                    "normalization stats snapshot"
                )
            self._validate_processor_action_space(
                getattr(lerobot_dataset, "processor", None), resolver
            )
        if audit_report_path is not None:
            audit_path = Path(audit_report_path).expanduser().resolve()
            if not audit_path.is_file():
                raise RuntimeCandidateDatasetContractError(
                    f"audit report file does not exist: {audit_path}"
                )
            try:
                audit = load_audit_report(audit_path)
            except (OSError, ValueError) as exc:
                raise RuntimeCandidateDatasetContractError(
                    f"cannot validate WARM audit report: {audit_path}"
                ) from exc
            if audit.report_sha256 != resolver.query_audit_sha256:
                raise RuntimeCandidateDatasetContractError(
                    "audit report does not match the candidate-cache data binding"
                )
            if audit.catalog_sha256 != catalog.content_sha256:
                raise RuntimeCandidateDatasetContractError(
                    "audit report does not match the episode catalog"
                )
            if not audit.episode_tables_hashed:
                raise RuntimeCandidateDatasetContractError(
                    "formal WARM training requires a complete episode-source audit"
                )
        self._dataset = dataset
        self._resolver = resolver
        self._catalog = catalog
        self._dataset_ids = dataset_ids
        self._episode_records = records
        action_space = resolver.action_space
        try:
            self._oracle_distance_config = ActionDistanceConfig(
                action_dim=action_space.action_dim,
                arm_dims=action_space.arm_dims,
                gripper_dims=action_space.gripper_dims,
                gripper_threshold=action_space.gripper_threshold,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeCandidateDatasetContractError(
                "WARM action contract cannot define the M1 oracle distance"
            ) from exc
        self.global_sample_stride = stride
        self._sampling_query_records_cache: tuple[
            tuple[QueryId, str], ...
        ] | None = None
        # Preserve the exact object path used by FastWAMTrainer:
        # ``dataset.lerobot_dataset.processor``.
        self.lerobot_dataset = lerobot_dataset

    @staticmethod
    def _validate_processor_action_space(
        processor: object,
        resolver: RuntimeCandidateResolver,
    ) -> None:
        if processor is None:
            raise RuntimeCandidateDatasetContractError(
                "wrapped dataset has no configured action processor"
            )
        contract = resolver.action_space
        processor_type = f"{type(processor).__module__}.{type(processor).__name__}"
        if processor_type != (
            "fastwam.datasets.lerobot.processors.fastwam_processor."
            "FastWAMProcessor"
        ):
            raise RuntimeCandidateDatasetContractError(
                "formal WARM training requires the exact FastWAMProcessor"
            )
        is_libero = contract.control_mode == (
            "libero_delta_eef_axis_angle_plus_gripper"
        )
        is_robotwin = contract.control_mode == (
            "robotwin_bimanual_qpos_plus_grippers"
        )
        is_piper = contract.control_mode in {
            PIPER_CONTROL_MODE,
            PIPER_JOINT_CONTROL_MODE,
        }
        if not (is_libero or is_robotwin or is_piper):
            raise RuntimeCandidateDatasetContractError(
                f"unsupported WARM processor control mode {contract.control_mode!r}"
            )
        if is_piper and contract.embodiment != PIPER_EMBODIMENT:
            raise RuntimeCandidateDatasetContractError(
                "Piper preprocessing requires the single active 6-DoF arm embodiment"
            )
        if is_libero or is_piper:
            expected_action_dim = 7
            expected_arm_dims = tuple(range(6))
            expected_gripper_dims = (6,)
            expected_proprio_dim = 7 if is_piper else 8
            expected_cameras = 2
            expected_norm_mode = "global:min/max"
        else:
            expected_action_dim = 14
            expected_gripper_dims = (6, 13)
            expected_arm_dims = tuple(
                index for index in range(14) if index not in expected_gripper_dims
            )
            expected_proprio_dim = 14
            expected_cameras = 3
            expected_norm_mode = "global:z-score"
        if (
            contract.action_dim != expected_action_dim
            or contract.arm_dims != expected_arm_dims
            or contract.gripper_dims != expected_gripper_dims
        ):
            raise RuntimeCandidateDatasetContractError(
                "processor action-space dimensions disagree with the WARM contract"
            )
        if int(getattr(processor, "action_output_dim", -1)) != contract.action_dim:
            raise RuntimeCandidateDatasetContractError(
                "processor action_output_dim does not match the WARM action contract"
            )
        if int(getattr(processor, "proprio_output_dim", -1)) != expected_proprio_dim:
            raise RuntimeCandidateDatasetContractError(
                "processor proprio dimension does not match the WARM action contract"
            )
        if (
            int(getattr(processor, "num_obs_steps", -1))
            != resolver.action_horizon + 1
            or int(getattr(processor, "num_output_cameras", -1))
            != expected_cameras
        ):
            raise RuntimeCandidateDatasetContractError(
                "processor observation horizon/camera count does not match "
                "the WARM action contract"
            )
        use_stepwise_norm = bool(
            getattr(processor, "use_stepwise_action_norm", True)
        )
        if use_stepwise_norm:
            raise RuntimeCandidateDatasetContractError(
                "WARM model-space actions require stepwise normalization disabled"
            )
        processor_norm_mode = str(
            getattr(processor, "norm_default_mode", "")
        )
        canonical_norm_mode = (
            f"{'stepwise' if use_stepwise_norm else 'global'}:"
            f"{processor_norm_mode}"
        )
        if (
            canonical_norm_mode != contract.normalization_mode
            or canonical_norm_mode != expected_norm_mode
        ):
            raise RuntimeCandidateDatasetContractError(
                "processor normalization mode does not match the WARM action contract"
            )
        if getattr(processor, "norm_exception_mode", None) not in (None, {}):
            raise RuntimeCandidateDatasetContractError(
                "processor normalization exceptions are outside the WARM action contract"
            )
        if getattr(processor, "action_state_transforms", None) is not None:
            raise RuntimeCandidateDatasetContractError(
                "processor action transforms are outside the WARM action contract"
            )
        try:
            normalizer = processor.normalizer
        except (AttributeError, ValueError) as exc:
            raise RuntimeCandidateDatasetContractError(
                "processor did not initialize its contract-bound normalizer"
            ) from exc
        if normalizer is None:
            raise RuntimeCandidateDatasetContractError(
                "processor did not initialize its contract-bound normalizer"
            )
        normalizer_type = (
            f"{type(normalizer).__module__}.{type(normalizer).__name__}"
        )
        if normalizer_type != (
            "fastwam.datasets.lerobot.utils.normalizer.LinearNormalizer"
        ):
            raise RuntimeCandidateDatasetContractError(
                "processor must use FastWAM's exact LinearNormalizer"
            )
        shape_meta = getattr(processor, "shape_meta", None)
        if not isinstance(shape_meta, Mapping):
            raise RuntimeCandidateDatasetContractError(
                "processor shape_meta must be a mapping"
            )
        action_meta = shape_meta.get("action")
        state_meta = shape_meta.get("state")
        image_meta = shape_meta.get("images")
        if any(
            value is None or isinstance(value, (str, bytes, Mapping))
            for value in (action_meta, state_meta, image_meta)
        ):
            raise RuntimeCandidateDatasetContractError(
                "processor shape_meta image/action/state entries must be sequences"
            )
        try:
            action_meta = tuple(action_meta)
            state_meta = tuple(state_meta)
            image_meta = tuple(image_meta)
            action_signature = tuple(
                (
                    str(item["key"]),
                    int(item["raw_shape"]),
                    int(item["shape"]),
                )
                for item in action_meta
            )
            state_signature = tuple(
                (
                    str(item["key"]),
                    int(item["raw_shape"]),
                    int(item["shape"]),
                )
                for item in state_meta
            )
            image_signature = tuple(
                (
                    str(item["key"]),
                    tuple(int(value) for value in item["shape"]),
                )
                for item in image_meta
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeCandidateDatasetContractError(
                "processor action/state metadata is incomplete"
            ) from exc
        if action_signature != (
            ("default", expected_action_dim, expected_action_dim),
        ):
            raise RuntimeCandidateDatasetContractError(
                "processor must preserve the contract-bound default action field"
            )
        if state_signature != (
            ("default", expected_proprio_dim, expected_proprio_dim),
        ):
            raise RuntimeCandidateDatasetContractError(
                "processor must preserve the contract-bound default state field"
            )
        expected_images = (
            (
                ("image", (3, 224, 224)),
                ("wrist_image", (3, 224, 224)),
            )
            if is_libero
            else (
                ("cam_high", (3, 240, 320)),
                ("cam_left_wrist", (3, 240, 320)),
                ("cam_right_wrist", (3, 240, 320)),
            )
        )
        if is_piper:
            expected_images = PIPER_IMAGE_SIGNATURE
        if image_signature != expected_images:
            raise RuntimeCandidateDatasetContractError(
                "processor camera metadata differs from the benchmark profile"
            )

        merger = getattr(processor, "action_state_merger", None)
        merger_type = (
            ""
            if merger is None
            else f"{type(merger).__module__}.{type(merger).__name__}"
        )
        if merger_type != (
            "fastwam.datasets.lerobot.transforms.action_state_merger."
            "ConcatLeftAlign"
        ):
            raise RuntimeCandidateDatasetContractError(
                "processor must use the exact ConcatLeftAlign action/state merger"
            )
        if (
            getattr(merger, "action_target_dim", None) is not None
            or getattr(merger, "state_target_dim", None) is not None
        ):
            raise RuntimeCandidateDatasetContractError(
                "processor must not pad action or state dimensions"
            )
        delta_masks = getattr(processor, "delta_action_dim_mask", None)
        if is_robotwin:
            if delta_masks is not None:
                raise RuntimeCandidateDatasetContractError(
                    "native RoboTwin qpos must not use a delta-action mask"
                )
            return
        if not isinstance(delta_masks, Mapping):
            raise RuntimeCandidateDatasetContractError(
                "processor must expose the Cartesian delta-action dimension mask"
            )
        action_keys = [
            item.get("key")
            for item in action_meta
            if isinstance(item, Mapping)
        ]
        if (
            len(action_keys) != len(action_meta)
            or len(set(action_keys)) != len(action_keys)
            or set(delta_masks) != set(action_keys)
        ):
            raise RuntimeCandidateDatasetContractError(
                "processor action metadata and delta-action masks have different keys"
            )
        flattened: list[bool] = []
        for item in action_meta:
            if not isinstance(item, Mapping):
                raise RuntimeCandidateDatasetContractError(
                    "processor action metadata entries must be mappings"
                )
            key = item.get("key")
            mask = delta_masks.get(key)
            shape = item.get("shape")
            if (
                not isinstance(mask, torch.Tensor)
                or mask.dtype != torch.bool
                or mask.ndim != 1
                or isinstance(shape, bool)
                or not isinstance(shape, Integral)
                or int(shape) != mask.numel()
            ):
                raise RuntimeCandidateDatasetContractError(
                    "processor delta-action mask is incomplete or invalid"
                )
            flattened.extend(bool(value) for value in mask.tolist())
        if len(flattened) != contract.action_dim:
            raise RuntimeCandidateDatasetContractError(
                "processor delta-action mask dimension does not match action space"
            )
        if any(not flattened[index] for index in contract.arm_dims) or any(
            flattened[index] for index in contract.gripper_dims
        ):
            raise RuntimeCandidateDatasetContractError(
                "processor arm/gripper delta semantics do not match the WARM action contract"
            )

    @staticmethod
    def _validate_selected_episodes(
        lerobot_dataset: object,
        catalog: EpisodeCatalog,
        *,
        query_split: str,
        task_allowlist: tuple[str, ...] | None,
        require_introspection: bool,
    ) -> None:
        """When introspection is available, prove the wrapped split exactly."""

        descriptors = tuple(
            sorted(catalog.datasets, key=lambda item: item.dataset_index)
        )
        multi_dataset = getattr(lerobot_dataset, "multi_dataset", None)
        children = getattr(multi_dataset, "_datasets", None)
        dataset_roots = getattr(lerobot_dataset, "dataset_dirs", None)
        if require_introspection and (children is None or dataset_roots is None):
            raise RuntimeCandidateDatasetContractError(
                "formal WARM training requires dataset child/root introspection"
            )
        if children is not None:
            if len(children) != len(descriptors):
                raise RuntimeCandidateDatasetContractError(
                    "wrapped multi-dataset count does not match the episode catalog"
                )
            for descriptor, child in zip(descriptors, children, strict=True):
                meta = getattr(child, "meta", None)
                total_episodes = getattr(meta, "total_episodes", None)
                if (
                    total_episodes is not None
                    and int(total_episodes) != descriptor.total_episodes
                ):
                    raise RuntimeCandidateDatasetContractError(
                        "wrapped dataset episode totals do not match the catalog"
                    )
                expected = tuple(
                    sorted(
                        record.episode_index
                        for record in catalog.episodes
                        if record.dataset_index == descriptor.dataset_index
                        and record.split == query_split
                        and (
                            task_allowlist is None
                            or record.primary_task in task_allowlist
                        )
                    )
                )
                selected = getattr(child, "episodes", None)
                actual = (
                    tuple(range(descriptor.total_episodes))
                    if selected is None
                    else tuple(sorted(int(value) for value in selected))
                )
                if actual != expected:
                    raise RuntimeCandidateDatasetContractError(
                        "wrapped dataset episode selection does not exactly match "
                        f"catalog split {query_split!r} for {descriptor.dataset_id!r}"
                    )
        if dataset_roots is not None:
            if isinstance(dataset_roots, (str, bytes)):
                raise RuntimeCandidateDatasetContractError(
                    "wrapped dataset roots must be an ordered sequence"
                )
            if len(dataset_roots) != len(descriptors):
                raise RuntimeCandidateDatasetContractError(
                    "wrapped dataset root count does not match the episode catalog"
                )
            for descriptor, root_value in zip(
                descriptors, dataset_roots, strict=True
            ):
                root = Path(root_value).expanduser().resolve()
                info = root / "meta" / "info.json"
                episodes = root / "meta" / "episodes.jsonl"
                try:
                    matches = (
                        info.is_file()
                        and episodes.is_file()
                        and sha256_file(info) == descriptor.info_sha256
                        and sha256_file(episodes) == descriptor.episodes_sha256
                    )
                except OSError as exc:
                    raise RuntimeCandidateDatasetContractError(
                        f"cannot hash wrapped dataset metadata under {root}"
                    ) from exc
                if not matches:
                    raise RuntimeCandidateDatasetContractError(
                        "wrapped dataset root/order metadata does not match "
                        f"catalog dataset {descriptor.dataset_id!r}"
                    )
        if children is not None and dataset_roots is not None:
            for child, root_value in zip(children, dataset_roots, strict=True):
                child_root = getattr(child, "root", None)
                if (
                    child_root is not None
                    and Path(child_root).expanduser().resolve()
                    != Path(root_value).expanduser().resolve()
                ):
                    raise RuntimeCandidateDatasetContractError(
                        "wrapped multi-dataset child order does not match dataset_dirs"
                    )

    @property
    def wrapped_dataset(self) -> torch.utils.data.Dataset:
        return self._dataset

    @property
    def resolver(self) -> RuntimeCandidateResolver:
        return self._resolver

    @property
    def catalog(self) -> EpisodeCatalog:
        return self._catalog

    def __len__(self) -> int:
        return len(self._dataset)

    def sampling_query_records(self) -> tuple[tuple[QueryId, str], ...]:
        """Return index-aligned query/task metadata without decoding videos."""

        cached = self._sampling_query_records_cache
        if cached is not None:
            return cached
        multi_dataset = getattr(self.lerobot_dataset, "multi_dataset", None)
        children = getattr(multi_dataset, "_datasets", None)
        if children is None:
            raise RuntimeCandidateDatasetContractError(
                "balanced sampling requires LeRobot child introspection"
            )
        rows: list[tuple[QueryId, str]] = []
        for dataset_index, child in enumerate(children):
            selected = getattr(child, "episodes", None)
            if selected is None:
                selected = range(int(child.meta.total_episodes))
            selected = tuple(int(value) for value in selected)
            starts = tuple(
                int(value) for value in child.episode_data_index["from"].tolist()
            )
            stops = tuple(
                int(value) for value in child.episode_data_index["to"].tolist()
            )
            if not (len(selected) == len(starts) == len(stops)):
                raise RuntimeCandidateDatasetContractError(
                    "selected episodes and LeRobot episode ranges disagree"
                )
            for episode_index, start, stop in zip(
                selected, starts, stops, strict=True
            ):
                record = self._episode_records.get((dataset_index, episode_index))
                if record is None or stop - start != record.length:
                    raise RuntimeCandidateDatasetContractError(
                        "catalog episode length disagrees with LeRobot sampling range"
                    )
                for frame_index in range(record.length):
                    rows.append(
                        (
                            QueryId(
                                record.dataset_id,
                                record.dataset_index,
                                record.episode_index,
                                frame_index,
                            ),
                            record.primary_task,
                        )
                    )
        if len(rows) != len(self):
            raise RuntimeCandidateDatasetContractError(
                "balanced-sampling metadata does not align with dataset length"
            )
        result = tuple(rows)
        self._sampling_query_records_cache = result
        return result

    def __getattr__(self, name: str) -> Any:
        # Called only after ordinary attribute lookup fails.  Use __dict__ to
        # avoid recursion during unpickling/partially-completed construction.
        dataset = self.__dict__.get("_dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def _episode_record(self, sample: Mapping[str, Any]) -> tuple[EpisodeRecord, int]:
        for field in ("dataset_index", "episode_index", "frame_index"):
            if field not in sample:
                raise RuntimeCandidateDatasetContractError(
                    f"sample is missing provenance field {field!r}"
                )
        dataset_index = _nonnegative_scalar_int(
            sample["dataset_index"], field="dataset_index"
        )
        episode_index = _nonnegative_scalar_int(
            sample["episode_index"], field="episode_index"
        )
        frame_index = _nonnegative_scalar_int(
            sample["frame_index"], field="frame_index"
        )
        record = self._episode_records.get((dataset_index, episode_index))
        if record is None:
            raise RuntimeCandidateDatasetContractError(
                "sample episode is absent from the immutable catalog"
            )
        dataset_id = self._dataset_ids.get(dataset_index)
        if dataset_id is None or dataset_id != record.dataset_id:
            raise RuntimeCandidateDatasetContractError(
                "catalog dataset_index-to-dataset_id mapping is inconsistent"
            )
        if record.split != self._resolver.query_split:
            raise RuntimeCandidateDatasetContractError(
                "sample episode split does not match the candidate cache"
            )
        if frame_index >= record.length:
            raise RuntimeCandidateDatasetContractError(
                "sample frame_index lies outside its catalog episode"
            )
        return record, frame_index

    def _validate_action(self, sample: Mapping[str, Any]) -> torch.Tensor:
        action = sample.get("action")
        expected_shape = (
            self._resolver.action_horizon,
            self._resolver.action_space.action_dim,
        )
        if not isinstance(action, torch.Tensor):
            raise RuntimeCandidateDatasetContractError(
                "sample action must be a torch.Tensor"
            )
        if action.device.type != "cpu" or tuple(action.shape) != expected_shape:
            raise RuntimeCandidateDatasetContractError(
                "sample action must be a CPU tensor with shape "
                f"{expected_shape}, got {tuple(action.shape)} on {action.device}"
            )
        if not action.is_floating_point() or not bool(torch.isfinite(action).all().item()):
            raise RuntimeCandidateDatasetContractError(
                "sample action must contain finite floating-point values"
            )
        return action

    def _is_explicit_padded_tail(
        self,
        sample: Mapping[str, Any],
        *,
        record: EpisodeRecord,
        frame_index: int,
    ) -> bool:
        masks = {
            field: _bool_padding_vector(sample, field) for field in _PADDING_FIELDS
        }
        horizon = self._resolver.action_horizon
        action_padding = masks["action_is_pad"]
        if action_padding.numel() != horizon:
            raise RuntimeCandidateDatasetContractError(
                "action_is_pad length does not match the candidate action horizon"
            )
        expected_action_padding = (
            torch.arange(horizon, dtype=torch.int64) + frame_index >= record.length
        )
        if not torch.equal(action_padding, expected_action_padding):
            raise RuntimeCandidateDatasetContractError(
                "action_is_pad disagrees with catalog episode boundaries"
            )

        # Event features contain N states and N-1 transition-grounded actions.
        # The masks still have to prove the exact catalog tail.  Whether that
        # tail may omit a retrieval row is decided separately by the immutable
        # query-frame policy; V5 RMBench covers every one of the N states.
        expected_tail = frame_index + horizon >= record.length
        observed_tail = any(bool(mask.any().item()) for mask in masks.values())
        if observed_tail != expected_tail:
            raise RuntimeCandidateDatasetContractError(
                "sample padding masks disagree with the catalog fixed-horizon tail"
            )
        return expected_tail

    def _allows_missing_candidate_row(
        self,
        sample: Mapping[str, Any],
        *,
        padded_tail: bool,
    ) -> bool:
        """Return whether this exact sample is outside the cache query domain."""

        policy = self._resolver.build_recipe.get(
            "query_frame_policy", "full_horizon_v1"
        )
        if policy == "full_horizon_v1":
            return padded_tail
        if policy == "all_factual_states_v1":
            # The formal RMBench cache covers every catalog observation,
            # including the final state.  Any missing row is corruption.
            return False
        raise RuntimeCandidateDatasetContractError(
            f"unsupported candidate query_frame_policy {policy!r}"
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("dataset index must be an integer")
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        base_sample = self._dataset[index]
        if not isinstance(base_sample, Mapping):
            raise RuntimeCandidateDatasetContractError(
                "wrapped dataset must return a mapping"
            )
        sample = dict(base_sample)
        collisions = sorted(set(sample).intersection(WARM_CANDIDATE_FIELDS))
        if collisions:
            raise RuntimeCandidateDatasetContractError(
                f"wrapped sample already contains WARM candidate fields: {collisions!r}"
            )

        record, frame_index = self._episode_record(sample)
        action = self._validate_action(sample)
        padded_tail = self._is_explicit_padded_tail(
            sample,
            record=record,
            frame_index=frame_index,
        )
        allow_missing = self._allows_missing_candidate_row(
            sample, padded_tail=padded_tail
        )
        query_id = QueryId(
            record.dataset_id,
            record.dataset_index,
            record.episode_index,
            frame_index,
        )
        try:
            resolved = self._resolver.resolve(
                query_id,
                allow_missing=allow_missing,
            )
        except KeyError as exc:
            raise RuntimeCandidateMissingRowError(
                "candidate cache has no exact row for non-padded or supervised "
                "partial-tail sample "
                f"{query_id!r}"
            ) from exc
        if allow_missing and resolved.valid_count:
            raise RuntimeCandidateDatasetContractError(
                "query outside the cache domain unexpectedly resolves to factual "
                "candidate events"
            )

        candidate_mu = self._resolver.gather_model_actions(resolved)
        expected_mu_shape = (
            self._resolver.fixed_k,
            self._resolver.action_horizon,
            self._resolver.action_space.action_dim,
        )
        if candidate_mu.shape != expected_mu_shape or not np.all(
            np.isfinite(candidate_mu)
        ):
            raise RuntimeCandidateDatasetContractError(
                "resolved candidate actions violate the fixed model-space shape"
            )

        # Resolver arrays are immutable.  Copy before torch.from_numpy so the
        # returned tensors own writable worker-local storage without exposing
        # or mutating the resolver snapshot.
        sample[WARM_CANDIDATE_MU] = torch.from_numpy(
            np.array(candidate_mu, dtype=np.float32, copy=True, order="C")
        )
        sample[WARM_CANDIDATE_MASK] = torch.from_numpy(
            np.array(resolved.mask, dtype=np.bool_, copy=True, order="C")
        )
        sample[WARM_CANDIDATE_SCORE] = torch.from_numpy(
            np.array(resolved.cosine_scores, dtype=np.float32, copy=True, order="C")
        )
        sample[WARM_CANDIDATE_EVENT_INDEX] = torch.from_numpy(
            np.array(resolved.bank_rows, dtype=np.int64, copy=True, order="C")
        )
        valid_slots = np.flatnonzero(resolved.mask)
        if valid_slots.size == 0:
            oracle_index = -1
        else:
            action_padding = _bool_padding_vector(sample, "action_is_pad")
            if bool(action_padding.any().item()):
                # Partial-action tails still resolve factual candidates for
                # retrieval/source/gate supervision, but full-horizon oracle
                # distance is undefined when future teacher steps are padded.
                oracle_index = -1
            else:
                target = action.detach().to(dtype=torch.float32).numpy()
                distances = np.asarray(
                    [
                        action_distance(
                            candidate_mu[int(slot)],
                            target,
                            self._oracle_distance_config,
                        ).total
                        for slot in valid_slots
                    ],
                    dtype=np.float64,
                )
                oracle_index = int(valid_slots[int(np.argmin(distances))])
        sample[WARM_ORACLE_CANDIDATE_INDEX] = torch.tensor(
            oracle_index, dtype=torch.int64
        )
        # Trainer reuses ``training_loss`` for held-out loss evaluation.  Keep
        # the independently bound cache split explicit in every collated batch
        # so the model cannot silently interpret a dev row under train identity.
        sample[WARM_QUERY_SPLIT] = self._resolver.query_split
        return sample


__all__ = [
    "RuntimeCandidateDatasetAdapter",
    "RuntimeCandidateDatasetContractError",
    "RuntimeCandidateDatasetError",
    "RuntimeCandidateMissingRowError",
    "WARM_CANDIDATE_EVENT_INDEX",
    "WARM_CANDIDATE_FIELDS",
    "WARM_CANDIDATE_MASK",
    "WARM_CANDIDATE_MU",
    "WARM_CANDIDATE_SCORE",
    "WARM_ORACLE_CANDIDATE_INDEX",
    "WARM_QUERY_SPLIT",
]
