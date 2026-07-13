"""Server-facing, import-safe primitives for WARM M1 feature precomputation.

This module intentionally contains no module-level Torch, Hydra, or DINO
imports.  The array contracts can therefore be validated on a CPU-only local
machine, while :class:`FastWAMProcessorAdapter` can instantiate and execute the
real FastWAM processor on a training server.

The temporal contract is deliberately strict: an episode with ``N`` factual
observations has exactly ``N - 1`` executable actions.  The terminal action in
the LeRobot-style per-frame record is never treated as factual control.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from typing import Any, Mapping, Sequence

import numpy as np

from .bank_builder import EpisodeFeatures


class FeaturePrecomputeError(ValueError):
    """Raised when raw or encoded episode data violates the M1 contract."""


def _readonly_float32(
    name: str,
    value: Any,
    *,
    rank: int | None = None,
    min_rank: int | None = None,
) -> np.ndarray:
    """Return a finite, C-contiguous, read-only float32 copy."""

    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise FeaturePrecomputeError(f"{name} must be a numeric array") from exc
    if rank is not None and array.ndim != rank:
        raise FeaturePrecomputeError(
            f"{name} must be rank {rank}, got shape {array.shape}"
        )
    if min_rank is not None and array.ndim < min_rank:
        raise FeaturePrecomputeError(
            f"{name} must have rank >= {min_rank}, got shape {array.shape}"
        )
    if array.ndim == 0 or any(int(size) <= 0 for size in array.shape):
        raise FeaturePrecomputeError(f"{name} must be non-empty, got {array.shape}")
    if not np.isfinite(array).all():
        raise FeaturePrecomputeError(f"{name} must contain only finite values")
    result = np.array(array, dtype=np.float32, copy=True, order="C")
    result.flags.writeable = False
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise FeaturePrecomputeError(f"{name} must be positive")
    return result


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise FeaturePrecomputeError(f"{name} must be non-negative")
    return result


def _patch_grid_shape(value: Any) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FeaturePrecomputeError(
            "patch_grid_size must contain two positive integers"
        )
    if len(value) != 2:
        raise FeaturePrecomputeError(
            "patch_grid_size must contain two positive integers"
        )
    dimensions: list[int] = []
    for item in value:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, np.integer)
        ):
            raise FeaturePrecomputeError(
                "patch_grid_size must contain two positive integers"
            )
        dimension = int(item)
        if dimension <= 0:
            raise FeaturePrecomputeError(
                "patch_grid_size must contain two positive integers"
            )
        dimensions.append(dimension)
    return dimensions[0], dimensions[1]


def _l2_normalize_rows(name: str, value: np.ndarray) -> np.ndarray:
    array = _readonly_float32(name, value, rank=2)
    norms = np.linalg.norm(array.astype(np.float64, copy=False), axis=1)
    if np.any(norms <= 0.0):
        first = int(np.flatnonzero(norms <= 0.0)[0])
        raise FeaturePrecomputeError(
            f"{name} rows must have non-zero L2 norm; first invalid row={first}"
        )
    normalized = array.astype(np.float64, copy=False) / norms[:, None]
    return _readonly_float32(f"normalized {name}", normalized, rank=2)


def factual_gripper_from_state(
    raw_state: Any, *, gripper_indices: Sequence[int] | None = None
) -> np.ndarray:
    """Compute a factual scalar gripper signal from explicit state channels.

    LIBERO/FastWAM proprioception stores the two gripper fingers in the last
    two state dimensions and remains the default.  Native RoboTwin qpos uses
    indices ``(6, 13)``; its profile must pass those indices explicitly.
    This signal is observation metadata, not an executable command.
    """

    state = _readonly_float32("raw_state", raw_state, rank=2)
    if gripper_indices is None:
        if state.shape[1] < 2:
            raise FeaturePrecomputeError(
                "raw_state needs at least two dimensions for the default factual "
                "gripper signal"
            )
        indices = (state.shape[1] - 2, state.shape[1] - 1)
    else:
        indices = tuple(gripper_indices)
        if (
            not indices
            or any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or int(index) < 0
                or int(index) >= state.shape[1]
                for index in indices
            )
            or len(set(int(index) for index in indices)) != len(indices)
        ):
            raise FeaturePrecomputeError(
                "gripper_indices must be unique valid state dimensions"
            )
        indices = tuple(int(index) for index in indices)
    gripper = np.abs(state[:, indices].astype(np.float64, copy=False)).sum(axis=1)
    return _readonly_float32("factual_gripper", gripper, rank=1)


def build_m1_context_keys(
    dino_cls: Any,
    *,
    task_index: int,
    catalog_task_count: int,
    visual_only: bool = False,
) -> np.ndarray:
    """Build the fixed M1 retrieval key for every factual observation.

    The visual component is a row-wise normalized DINO CLS vector.  In the
    task-conditioned contract, a catalog-global one-hot task vector is
    concatenated with equal L2 energy: both visual and task components have
    norm ``sqrt(1/2)``.  ``visual_only`` is the explicit ablation and returns
    the unit-normalized CLS vector without changing its dimension.
    """

    visual = _l2_normalize_rows("dino_cls", dino_cls)
    task = _nonnegative_int("task_index", task_index)
    task_count = _positive_int("catalog_task_count", catalog_task_count)
    if task >= task_count:
        raise FeaturePrecomputeError(
            f"task_index {task} is outside catalog-global range [0, {task_count})"
        )
    if not isinstance(visual_only, (bool, np.bool_)):
        raise TypeError("visual_only must be a boolean")
    if bool(visual_only):
        return visual

    task_one_hot = np.zeros((visual.shape[0], task_count), dtype=np.float32)
    task_one_hot[:, task] = 1.0
    equal_energy_scale = np.float32(np.sqrt(0.5))
    keys = np.concatenate(
        (visual * equal_energy_scale, task_one_hot * equal_energy_scale), axis=1
    )
    return _readonly_float32("context_keys", keys, rank=2)


def _adaptive_bounds(input_size: int, output_index: int) -> tuple[int, int]:
    # Matches the regions used by adaptive average pooling.  Odd grids may
    # overlap at their center, which is intentional and matches Torch.
    start = (output_index * input_size) // 2
    stop = ((output_index + 1) * input_size + 1) // 2
    return start, stop


def pool_dino_patch_grid_2x2(
    patch_tokens: Any,
    *,
    patch_grid_size: Sequence[int] | None = None,
) -> np.ndarray:
    """Adaptive-average-pool factual DINO patch tokens to four spatial tokens.

    Args:
        patch_tokens: Either ``[N, grid_h, grid_w, D]`` or flattened
            ``[N, grid_h * grid_w, D]`` patch tokens.  CLS must already be
            removed.
        patch_grid_size: Required for flattened input; ignored only when the
            spatial grid is explicit.

    Returns:
        Read-only float32 array with shape ``[N, 4, D]`` in row-major 2x2
        order.  These are factual observation features, never predicted future
        tokens.
    """

    patches = _readonly_float32("patch_tokens", patch_tokens, min_rank=3)
    if patches.ndim == 4:
        if patch_grid_size is not None:
            supplied = _patch_grid_shape(patch_grid_size)
            if supplied != tuple(patches.shape[1:3]):
                raise FeaturePrecomputeError(
                    "patch_grid_size does not match explicit patch grid: "
                    f"{supplied} != {tuple(patches.shape[1:3])}"
                )
        grid = patches
    elif patches.ndim == 3:
        if patch_grid_size is None:
            raise FeaturePrecomputeError(
                "patch_grid_size is required for flattened DINO patch tokens"
            )
        grid_shape = _patch_grid_shape(patch_grid_size)
        if grid_shape[0] * grid_shape[1] != patches.shape[1]:
            raise FeaturePrecomputeError(
                "flattened patch count does not match patch_grid_size: "
                f"{patches.shape[1]} != {grid_shape[0]} * {grid_shape[1]}"
            )
        grid = patches.reshape(
            patches.shape[0], grid_shape[0], grid_shape[1], patches.shape[2]
        )
    else:
        raise FeaturePrecomputeError(
            "patch_tokens must have shape [N,P,D] or [N,H,W,D]"
        )

    height, width = int(grid.shape[1]), int(grid.shape[2])
    pooled: list[np.ndarray] = []
    for output_y in range(2):
        y_start, y_stop = _adaptive_bounds(height, output_y)
        for output_x in range(2):
            x_start, x_stop = _adaptive_bounds(width, output_x)
            region = grid[:, y_start:y_stop, x_start:x_stop, :]
            pooled.append(region.astype(np.float64, copy=False).mean(axis=(1, 2)))
    result = np.stack(pooled, axis=1)
    return _readonly_float32("pooled_dino_patch_tokens", result, rank=3)


@dataclass(frozen=True, slots=True)
class DinoFactualFeatures:
    """CLS and 2x2 spatial tokens extracted from factual DINO output."""

    cls: np.ndarray
    spatial: np.ndarray

    def __post_init__(self) -> None:
        cls = _readonly_float32("DINO CLS", self.cls, rank=2)
        spatial = _readonly_float32("DINO spatial tokens", self.spatial, rank=3)
        if spatial.shape[0] != cls.shape[0] or spatial.shape[1] != 4:
            raise FeaturePrecomputeError(
                "DINO spatial tokens must have shape [N,4,D] and share N with CLS"
            )
        if spatial.shape[2] != cls.shape[1]:
            raise FeaturePrecomputeError(
                "DINO CLS and spatial tokens must share their feature dimension"
            )
        object.__setattr__(self, "cls", cls)
        object.__setattr__(self, "spatial", spatial)


def extract_dino_factual_features(
    last_hidden_state: Any,
    *,
    patch_grid_size: Sequence[int],
    register_token_count: int = 0,
) -> DinoFactualFeatures:
    """Split a DINO hidden sequence and pool its factual patch grid.

    ``last_hidden_state`` follows the common ``[N, tokens, D]`` convention:
    CLS first, optional register tokens next, and spatial patches last.  This
    pure NumPy boundary keeps the GPU model wrapper replaceable and testable.
    """

    hidden = _readonly_float32("last_hidden_state", last_hidden_state, rank=3)
    register_count = _nonnegative_int("register_token_count", register_token_count)
    grid_shape = _patch_grid_shape(patch_grid_size)
    patch_count = grid_shape[0] * grid_shape[1]
    expected_tokens = 1 + register_count + patch_count
    if hidden.shape[1] != expected_tokens:
        raise FeaturePrecomputeError(
            "DINO token count does not match CLS/register/grid contract: "
            f"{hidden.shape[1]} != {expected_tokens}"
        )
    cls = hidden[:, 0, :]
    patches = hidden[:, 1 + register_count :, :]
    return DinoFactualFeatures(
        cls=cls,
        spatial=pool_dino_patch_grid_2x2(
            patches, patch_grid_size=grid_shape
        ),
    )


def _numpy_leaf_mapping(name: str, value: Any) -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        if not value:
            raise FeaturePrecomputeError(f"{name} mapping must not be empty")
        result: dict[str, np.ndarray] = {}
        for key, leaf in value.items():
            if not isinstance(key, str) or not key:
                raise FeaturePrecomputeError(
                    f"{name} keys must be non-empty strings"
                )
            result[key] = _readonly_float32(f"{name}[{key!r}]", leaf, rank=2)
        return result
    return {"default": _readonly_float32(name, value, rank=2)}


def _shared_time_length(name: str, values: Mapping[str, np.ndarray]) -> int:
    lengths = {int(value.shape[0]) for value in values.values()}
    if len(lengths) != 1:
        raise FeaturePrecomputeError(
            f"all {name} fields must share a time length, got {sorted(lengths)}"
        )
    return next(iter(lengths))


def _to_backend_array(value: np.ndarray, backend: str) -> Any:
    if backend == "numpy":
        return np.array(value, dtype=np.float32, copy=True, order="C")
    if backend != "torch":
        raise FeaturePrecomputeError(
            "tensor_backend must be either 'torch' or 'numpy'"
        )
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - depends on deployment env
        raise RuntimeError(
            "Torch is required only when tensor_backend='torch'; use the NumPy "
            "backend for local contract tests"
        ) from exc
    return torch.as_tensor(np.array(value, copy=True), dtype=torch.float32)


def _processor_output_to_numpy(name: str, value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return _readonly_float32(name, value, rank=2)
    # Torch-like duck typing avoids importing Torch merely to identify a tensor.
    try:
        detached = value.detach()
        cpu_value = detached.cpu()
        array = cpu_value.numpy()
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise FeaturePrecomputeError(
            f"processor output {name} must be a NumPy array or CPU-convertible tensor"
        ) from exc
    return _readonly_float32(name, array, rank=2)


@dataclass(frozen=True, slots=True)
class ProcessedActionState:
    """Exact FastWAM model-space actions and factual proprioception."""

    model_actions: np.ndarray
    proprio: np.ndarray
    gripper: np.ndarray

    def __post_init__(self) -> None:
        actions = _readonly_float32("model_actions", self.model_actions, rank=2)
        proprio = _readonly_float32("proprio", self.proprio, rank=2)
        gripper = _readonly_float32("gripper", self.gripper, rank=1)
        if proprio.shape[0] < 2:
            raise FeaturePrecomputeError("an episode needs at least two observations")
        if actions.shape[0] != proprio.shape[0] - 1:
            raise FeaturePrecomputeError(
                "model_actions must have N-1 rows for N factual proprio states"
            )
        if gripper.shape[0] != proprio.shape[0]:
            raise FeaturePrecomputeError(
                "gripper must have one value for every factual proprio state"
            )
        object.__setattr__(self, "model_actions", actions)
        object.__setattr__(self, "proprio", proprio)
        object.__setattr__(self, "gripper", gripper)


class FastWAMProcessorAdapter:
    """Run the exact FastWAM action/state preprocessing chain for one episode.

    The adapter deliberately calls only:

    ``action_state_transform -> normalizer.forward -> action_state_merger.forward``

    It does not invoke image augmentation, tokenization, padding, or training
    preprocessing.  Raw per-frame actions must have the same ``N`` rows as raw
    states; the terminal action is removed before the processor is called.
    """

    def __init__(self, processor: Any, *, tensor_backend: str = "torch") -> None:
        for attribute in (
            "action_state_transform",
            "normalizer",
            "action_state_merger",
        ):
            try:
                inspect.getattr_static(processor, attribute)
            except AttributeError as exc:
                raise TypeError(
                    f"processor is missing required attribute {attribute!r}"
                ) from exc
        if not callable(getattr(processor, "action_state_transform", None)):
            raise TypeError("processor.action_state_transform must be callable")
        if tensor_backend not in {"torch", "numpy"}:
            raise FeaturePrecomputeError(
                "tensor_backend must be either 'torch' or 'numpy'"
            )
        self._processor = processor
        self._tensor_backend = tensor_backend

    @classmethod
    def from_hydra_config(
        cls,
        processor_config: Any,
        *,
        dataset_stats: Mapping[str, Any] | None = None,
        tensor_backend: str = "torch",
    ) -> "FastWAMProcessorAdapter":
        """Instantiate the FastWAM processor without importing Hydra eagerly."""

        try:
            hydra_utils = importlib.import_module("hydra.utils")
        except ImportError as exc:  # pragma: no cover - server dependency path
            raise RuntimeError(
                "Hydra is required only to instantiate a processor from config"
            ) from exc
        processor = hydra_utils.instantiate(processor_config)
        if dataset_stats is not None:
            if not hasattr(processor, "set_normalizer_from_stats"):
                raise TypeError(
                    "instantiated processor cannot receive FastWAM dataset stats"
                )
            processor.set_normalizer_from_stats(dataset_stats)
        # The adapter itself performs no stochastic operation, but eval mode is
        # the safe contract when processor transforms expose train/eval state.
        if hasattr(processor, "eval"):
            processor.eval()
        return cls(processor, tensor_backend=tensor_backend)

    def process(
        self,
        raw_action: Mapping[str, Any] | Any,
        raw_state: Mapping[str, Any] | Any,
        *,
        gripper_state_key: str | None = None,
        gripper_indices: Sequence[int] | None = None,
    ) -> ProcessedActionState:
        action_fields = _numpy_leaf_mapping("raw_action", raw_action)
        state_fields = _numpy_leaf_mapping("raw_state", raw_state)
        observation_count = _shared_time_length("raw_state", state_fields)
        action_count = _shared_time_length("raw_action", action_fields)
        if observation_count < 2:
            raise FeaturePrecomputeError("an episode needs at least two observations")
        if action_count != observation_count:
            raise FeaturePrecomputeError(
                "raw per-frame action and state must both have N rows before "
                f"dropping the terminal action; got {action_count} and "
                f"{observation_count}"
            )

        if gripper_state_key is None:
            if "default" in state_fields:
                gripper_source = state_fields["default"]
            elif len(state_fields) == 1:
                gripper_source = next(iter(state_fields.values()))
            else:
                raise FeaturePrecomputeError(
                    "gripper_state_key is required when raw_state has multiple fields"
                )
        else:
            if gripper_state_key not in state_fields:
                raise FeaturePrecomputeError(
                    f"gripper_state_key {gripper_state_key!r} is not in raw_state"
                )
            gripper_source = state_fields[gripper_state_key]
        factual_gripper = factual_gripper_from_state(
            gripper_source, gripper_indices=gripper_indices
        )

        batch = {
            "action": {
                key: _to_backend_array(value[:-1], self._tensor_backend)
                for key, value in action_fields.items()
            },
            "state": {
                key: _to_backend_array(value, self._tensor_backend)
                for key, value in state_fields.items()
            },
        }
        try:
            transformed = self._processor.action_state_transform(batch)
            normalized = self._processor.normalizer.forward(transformed)
            merged = self._processor.action_state_merger.forward(normalized)
        except (AssertionError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise FeaturePrecomputeError(
                "FastWAM action/state processor rejected the complete episode"
            ) from exc
        if not isinstance(merged, Mapping):
            raise FeaturePrecomputeError("FastWAM processor must return a mapping")
        if "action" not in merged or "state" not in merged:
            raise FeaturePrecomputeError(
                "FastWAM merger output must contain 'action' and 'state'"
            )
        actions = _processor_output_to_numpy("model_actions", merged["action"])
        proprio = _processor_output_to_numpy("proprio", merged["state"])
        return ProcessedActionState(
            model_actions=actions,
            proprio=proprio,
            gripper=factual_gripper,
        )


def assemble_episode_features(
    *,
    dataset_id: str,
    dataset_index: int,
    episode_index: int,
    task_index: int,
    catalog_task_count: int,
    source_episode_sha256: str,
    model_actions: Any,
    proprio: Any,
    dino_cls: Any,
    semantic_features: Any,
    gripper: Any | None = None,
    raw_state_for_gripper: Any | None = None,
    vae_features: Any | None = None,
    visual_only_context: bool = False,
) -> EpisodeFeatures:
    """Assemble one immutable episode using the strict factual M1 contract.

    This function is pure NumPy.  It accepts already-processed FastWAM actions
    and proprioception plus factual encoder outputs.  Exactly one of ``gripper``
    and ``raw_state_for_gripper`` must be provided; the latter uses the default
    ``abs(last2).sum`` observation signal.
    """

    actions = _readonly_float32("model_actions", model_actions, rank=2)
    states = _readonly_float32("proprio", proprio, rank=2)
    cls = _readonly_float32("dino_cls", dino_cls, rank=2)
    semantics = _readonly_float32(
        "semantic_features", semantic_features, min_rank=2
    )
    if states.shape[0] < 2:
        raise FeaturePrecomputeError("an episode needs at least two observations")
    observation_count = int(states.shape[0])
    if actions.shape[0] != observation_count - 1:
        raise FeaturePrecomputeError(
            "strict episode contract requires N observations and N-1 actions; "
            f"got N={observation_count}, actions={actions.shape[0]}"
        )
    for name, value in (("dino_cls", cls), ("semantic_features", semantics)):
        if value.shape[0] != observation_count:
            raise FeaturePrecomputeError(
                f"{name} must have N={observation_count} factual rows, "
                f"got {value.shape[0]}"
            )

    if (gripper is None) == (raw_state_for_gripper is None):
        raise FeaturePrecomputeError(
            "provide exactly one of gripper or raw_state_for_gripper"
        )
    if gripper is None:
        factual_gripper = factual_gripper_from_state(raw_state_for_gripper)
    else:
        factual_gripper = _readonly_float32("gripper", gripper, rank=1)
    if factual_gripper.shape[0] != observation_count:
        raise FeaturePrecomputeError(
            f"gripper must have N={observation_count} factual values"
        )

    vae = None
    if vae_features is not None:
        vae = _readonly_float32("vae_features", vae_features, min_rank=2)
        if vae.shape[0] != observation_count:
            raise FeaturePrecomputeError(
                f"vae_features must have N={observation_count} factual rows"
            )

    keys = build_m1_context_keys(
        cls,
        task_index=task_index,
        catalog_task_count=catalog_task_count,
        visual_only=visual_only_context,
    )
    # EpisodeFeatures re-validates and takes immutable copies.  Keeping this
    # final boundary centralizes event-bank compatibility checks.
    return EpisodeFeatures(
        dataset_id=dataset_id,
        dataset_index=dataset_index,
        episode_index=episode_index,
        task_index=task_index,
        source_episode_sha256=source_episode_sha256,
        model_actions=actions,
        proprio=states,
        gripper=factual_gripper,
        context_keys=keys,
        semantic_features=semantics,
        vae_features=vae,
    )


__all__ = [
    "DinoFactualFeatures",
    "FastWAMProcessorAdapter",
    "FeaturePrecomputeError",
    "ProcessedActionState",
    "assemble_episode_features",
    "build_m1_context_keys",
    "extract_dino_factual_features",
    "factual_gripper_from_state",
    "pool_dino_patch_grid_2x2",
]
