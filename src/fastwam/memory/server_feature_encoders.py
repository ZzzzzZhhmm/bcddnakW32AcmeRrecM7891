"""Server-only image and DINO adapters for factual WARM feature caches.

The module is deliberately import-safe on a CPU-only development machine:
neither :mod:`torch` nor :mod:`transformers` is imported at module import
time.  The production backends are loaded only when an adapter actually
needs them, while tests may inject small duck-typed backends.

Two invariants are especially important here:

* decoded LeRobot frames follow the *same* uint8 round-trip and
  ``processor.val_transforms`` path as FastWAM; and
* DINO receives those already-resized ``[0, 1]`` tensors and applies only the
  pinned image processor's channel mean/std.  Its resize/crop pipeline is
  never called.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .feature_precompute import (
    DinoFactualFeatures,
    FeaturePrecomputeError,
    extract_dino_factual_features,
)


_PINNED_REVISION = re.compile(r"[0-9a-fA-F]{7,64}\Z")
_IMAGE_SIZE = (224, 224)
LIBERO_IMAGE_PROFILE = "libero"
ROBOTWIN_IMAGE_PROFILE = "robotwin"
_IMAGE_PROFILES = frozenset({LIBERO_IMAGE_PROFILE, ROBOTWIN_IMAGE_PROFILE})
_ROBOTWIN_PROCESSOR_SIZE = (240, 320)
_ROBOTWIN_HEAD_SIZE = (256, 320)
_ROBOTWIN_WRIST_SIZE = (128, 160)
_ROBOTWIN_COMPOSITE_SIZE = (384, 320)


class ServerFeatureEncodingError(FeaturePrecomputeError):
    """Raised when a server encoder violates the factual feature contract."""


def _immutable_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Return a C-contiguous array backed by immutable bytes."""

    contiguous = np.ascontiguousarray(value, dtype=dtype)
    frozen = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return frozen.reshape(contiguous.shape)


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{field} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ServerFeatureEncodingError(f"{field} must be a positive integer")
    return result


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{field} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ServerFeatureEncodingError(f"{field} must be a non-negative integer")
    return result


def _size_pair(value: object, *, field: str) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        size = _positive_int(value, field=field)
        return size, size
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ServerFeatureEncodingError(
            f"{field} must be a positive integer or (height, width)"
        )
    if len(value) != 2:
        raise ServerFeatureEncodingError(
            f"{field} must be a positive integer or (height, width)"
        )
    return (
        _positive_int(value[0], field=f"{field}[0]"),
        _positive_int(value[1], field=f"{field}[1]"),
    )


def _to_numpy(name: str, value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        detached = value.detach()
        cpu_value = detached.cpu()
        try:
            return cpu_value.numpy()
        except (TypeError, RuntimeError):
            # NumPy has no native bfloat16 dtype.  DINO normally runs in bf16
            # on the feature server, so convert the model output explicitly
            # before crossing the tensor/NumPy boundary.
            torch = _require_torch()
            return detached.to(device="cpu", dtype=torch.float32).numpy()
    except (AttributeError, TypeError, RuntimeError) as exc:
        raise ServerFeatureEncodingError(
            f"{name} must be a NumPy array or CPU-convertible tensor"
        ) from exc


def _default_tensor_factory(value: np.ndarray) -> Any:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - server dependency path
        raise RuntimeError(
            "FastWAM image preprocessing requires torch on the feature server; "
            "inject tensor_factory for a contract test"
        ) from exc
    return torch.as_tensor(value)


def _benchmark_profile(value: object) -> str:
    if not isinstance(value, str) or value not in _IMAGE_PROFILES:
        raise ServerFeatureEncodingError(
            "benchmark_profile must be exactly 'libero' or 'robotwin'"
        )
    return value


def _default_spatial_resize(value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Match ``RobotVideoDataset``'s bilinear antialiased tensor resize."""

    try:
        torch = importlib.import_module("torch")
        transforms_f = importlib.import_module("torchvision.transforms.functional")
    except ImportError as exc:  # pragma: no cover - server dependency path
        raise RuntimeError(
            "RoboTwin image preprocessing requires torch and torchvision on the "
            "feature server; inject spatial_resize for a contract test"
        ) from exc
    tensor = torch.as_tensor(np.ascontiguousarray(value, dtype=np.float32))
    resized = transforms_f.resize(
        tensor,
        size=list(size),
        interpolation=transforms_f.InterpolationMode.BILINEAR,
        antialias=True,
    )
    return np.asarray(_to_numpy("spatially resized camera", resized))


def _validate_camera_keys(camera_keys: Sequence[str]) -> tuple[str, ...]:
    if isinstance(camera_keys, (str, bytes)):
        raise TypeError("camera_keys must be a sequence of camera names")
    result = tuple(camera_keys)
    if not result:
        raise ServerFeatureEncodingError("at least one camera key is required")
    if any(
        not isinstance(key, str) or not key or key.strip() != key for key in result
    ):
        raise ServerFeatureEncodingError(
            "camera keys must be non-empty normalized strings"
        )
    if len(set(result)) != len(result):
        raise ServerFeatureEncodingError("camera keys must be unique")
    return result


@dataclass(frozen=True, slots=True)
class PreparedFastWAMImages:
    """Canonical per-camera DINO inputs and concatenated VAE inputs."""

    camera_frames: Mapping[str, np.ndarray]
    vae_frames: np.ndarray
    camera_keys: tuple[str, ...]
    concat_mode: str
    benchmark_profile: str = LIBERO_IMAGE_PROFILE

    def __post_init__(self) -> None:
        if not isinstance(self.camera_frames, MappingProxyType):
            raise TypeError("camera_frames must be an immutable mapping")
        if tuple(self.camera_frames) != self.camera_keys:
            raise ServerFeatureEncodingError(
                "camera_frames order must exactly match camera_keys"
            )
        rows: int | None = None
        for key, value in self.camera_frames.items():
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float32
                or value.flags.writeable
                or value.ndim != 4
                or value.shape[1:] != (3, *_IMAGE_SIZE)
            ):
                raise ServerFeatureEncodingError(
                    f"camera {key!r} must contain immutable float32 "
                    "[N,3,224,224] frames"
                )
            if (
                not np.isfinite(value).all()
                or np.any(value < 0.0)
                or np.any(value > 1.0)
            ):
                raise ServerFeatureEncodingError(
                    f"camera {key!r} must contain finite [0,1] pixels"
                )
            if rows is None:
                rows = int(value.shape[0])
            elif value.shape[0] != rows:
                raise ServerFeatureEncodingError("all cameras must share N")
        vae = self.vae_frames
        if (
            not isinstance(vae, np.ndarray)
            or vae.dtype != np.float32
            or vae.flags.writeable
            or vae.ndim != 4
            or vae.shape[:2] != (rows, 3)
        ):
            raise ServerFeatureEncodingError(
                "vae_frames must be immutable float32 [N,3,H,W]"
            )
        profile = _benchmark_profile(self.benchmark_profile)
        if profile == ROBOTWIN_IMAGE_PROFILE:
            if self.concat_mode != "robotwin" or len(self.camera_keys) != 3:
                raise ServerFeatureEncodingError(
                    "RoboTwin prepared images require three cameras and robotwin layout"
                )
            expected_spatial = _ROBOTWIN_COMPOSITE_SIZE
        elif self.concat_mode == "horizontal":
            expected_spatial = (224, 224 * len(self.camera_keys))
        elif self.concat_mode == "vertical":
            expected_spatial = (224 * len(self.camera_keys), 224)
        else:
            raise ServerFeatureEncodingError(
                "concat_mode must be 'horizontal' or 'vertical'"
            )
        if vae.shape[-2:] != expected_spatial:
            raise ServerFeatureEncodingError(
                "vae_frames spatial shape does not match camera layout: "
                f"{vae.shape[-2:]} != {expected_spatial}"
            )
        if not np.isfinite(vae).all() or np.any(vae < -1.0) or np.any(vae > 1.0):
            raise ServerFeatureEncodingError("vae_frames must be finite in [-1,1]")


class FastWAMImageAdapter:
    """Apply FastWAM's exact validation image path to full episodes.

    Args:
        processor: Instantiated FastWAM processor.  Only ``val_transforms`` is
            accessed; training augmentation and ``processor.preprocess`` are
            never invoked.
        camera_keys: Ordered camera keys.  This order also defines camera
            concatenation.
        concat_mode: ``"horizontal"``/``"vertical"`` for LIBERO, or the
            exact ``"robotwin"`` head-over-wrists composite.
        benchmark_profile: Explicit artifact profile.  The default preserves
            the original LIBERO behavior.
        tensor_factory: Optional NumPy-uint8-to-tensor adapter for tests.  The
            production default lazily calls ``torch.as_tensor``.

    ``preprocess`` returns one immutable float32 ``[N,3,224,224]`` array in
    ``[0,1]`` for every camera.  ``vae_frames`` concatenates those arrays and
    converts them to ``[-1,1]``.  ``prepare`` performs both operations.
    """

    def __init__(
        self,
        processor: Any,
        camera_keys: Sequence[str],
        concat_mode: str,
        *,
        benchmark_profile: str = LIBERO_IMAGE_PROFILE,
        tensor_factory: Callable[[np.ndarray], Any] | None = None,
        spatial_resize: Callable[[np.ndarray, tuple[int, int]], Any] | None = None,
    ) -> None:
        try:
            inspect.getattr_static(processor, "val_transforms")
        except AttributeError as exc:
            raise TypeError("processor is missing required 'val_transforms'") from exc
        profile = _benchmark_profile(benchmark_profile)
        allowed_layouts = (
            {"robotwin"}
            if profile == ROBOTWIN_IMAGE_PROFILE
            else {"horizontal", "vertical"}
        )
        if concat_mode not in allowed_layouts:
            raise ServerFeatureEncodingError(
                f"concat_mode {concat_mode!r} is invalid for {profile!r} profile"
            )
        if tensor_factory is not None and not callable(tensor_factory):
            raise TypeError("tensor_factory must be callable")
        if spatial_resize is not None and not callable(spatial_resize):
            raise TypeError("spatial_resize must be callable")
        self._processor = processor
        self._camera_keys = _validate_camera_keys(camera_keys)
        if profile == ROBOTWIN_IMAGE_PROFILE and len(self._camera_keys) != 3:
            raise ServerFeatureEncodingError(
                "RoboTwin image profile requires exactly three ordered cameras: "
                "head, left wrist, right wrist"
            )
        self._concat_mode = concat_mode
        self._benchmark_profile = profile
        self._tensor_factory = tensor_factory or _default_tensor_factory
        self._spatial_resize = spatial_resize or _default_spatial_resize

    @property
    def camera_keys(self) -> tuple[str, ...]:
        return self._camera_keys

    @property
    def concat_mode(self) -> str:
        return self._concat_mode

    @property
    def benchmark_profile(self) -> str:
        return self._benchmark_profile

    def _transforms_for(self, camera_key: str) -> tuple[Callable[[Any], Any], ...]:
        transforms = self._processor.val_transforms
        if isinstance(transforms, Mapping):
            if camera_key not in transforms:
                raise ServerFeatureEncodingError(
                    f"processor.val_transforms has no entry for {camera_key!r}"
                )
            transforms = transforms[camera_key]
        if isinstance(transforms, (str, bytes)):
            raise ServerFeatureEncodingError(
                "processor.val_transforms must be a transform sequence or "
                "a camera-keyed mapping of sequences"
            )
        try:
            result = tuple(transforms)
        except TypeError as exc:
            raise ServerFeatureEncodingError(
                "processor.val_transforms must be a transform sequence or "
                "a camera-keyed mapping of sequences"
            ) from exc
        if not result or any(not callable(transform) for transform in result):
            raise ServerFeatureEncodingError(
                f"validation transforms for {camera_key!r} must be non-empty callables"
            )
        return result

    def _validation_frames(self, images: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Apply only the baseline per-camera validation transform path.

        The input must be the full reader's float RGB mapping in ``[0,1]``.
        Values are passed through the exact baseline ``*255 -> uint8``
        conversion, then through the configured validation transforms for that
        camera.  Out-of-range decoder output is rejected rather than hidden by
        clipping.
        """

        if not isinstance(images, Mapping):
            raise TypeError("images must be a camera-keyed mapping")
        if tuple(images) != self._camera_keys:
            raise ServerFeatureEncodingError(
                "image keys and order must exactly match camera_keys: "
                f"{tuple(images)!r} != {self._camera_keys!r}"
            )

        result: dict[str, np.ndarray] = {}
        rows: int | None = None
        expected_size = (
            _ROBOTWIN_PROCESSOR_SIZE
            if self._benchmark_profile == ROBOTWIN_IMAGE_PROFILE
            else _IMAGE_SIZE
        )
        for key in self._camera_keys:
            raw = np.asarray(images[key])
            if raw.dtype != np.float32:
                raise ServerFeatureEncodingError(
                    f"decoded camera {key!r} must be float32 RGB"
                )
            if raw.ndim != 4 or raw.shape[0] <= 0 or raw.shape[1] != 3:
                raise ServerFeatureEncodingError(
                    f"decoded camera {key!r} must have shape [N,3,H,W]"
                )
            if raw.shape[2] <= 0 or raw.shape[3] <= 0 or not np.isfinite(raw).all():
                raise ServerFeatureEncodingError(
                    f"decoded camera {key!r} must have finite non-empty pixels"
                )
            if np.any(raw < 0.0) or np.any(raw > 1.0):
                raise ServerFeatureEncodingError(
                    f"decoded camera {key!r} must contain pixels in [0,1]"
                )
            if rows is None:
                rows = int(raw.shape[0])
            elif raw.shape[0] != rows:
                raise ServerFeatureEncodingError("all cameras must share N")

            # This intentionally mirrors BaseLerobotDataset._get_image after
            # validating the decoder's factual [0,1] contract above.
            uint8 = (raw * 255.0).astype(np.uint8)
            transformed: Any = self._tensor_factory(
                np.ascontiguousarray(uint8, dtype=np.uint8)
            )
            for transform in self._transforms_for(key):
                transformed = transform(transformed)
            array = np.asarray(_to_numpy(f"transformed camera {key!r}", transformed))
            if not np.issubdtype(array.dtype, np.floating):
                raise ServerFeatureEncodingError(
                    f"validation transforms for {key!r} must output floating point"
                )
            if array.shape != (rows, 3, *expected_size):
                raise ServerFeatureEncodingError(
                    f"validation transforms for {key!r} must output "
                    f"[{rows},3,{expected_size[0]},{expected_size[1]}], "
                    f"got {array.shape}"
                )
            if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
                raise ServerFeatureEncodingError(
                    f"validation transforms for {key!r} must output finite [0,1] pixels"
                )
            result[key] = _immutable_array(array, dtype=np.dtype(np.float32))
        return result

    def _resize(self, value: np.ndarray, size: tuple[int, int], *, field: str) -> np.ndarray:
        resized = np.asarray(self._spatial_resize(value, size))
        expected = (value.shape[0], 3, *size)
        if resized.shape != expected or not np.issubdtype(resized.dtype, np.floating):
            raise ServerFeatureEncodingError(
                f"{field} resize must return floating {expected}, got "
                f"{resized.shape} {resized.dtype}"
            )
        if not np.isfinite(resized).all() or np.any(resized < 0.0) or np.any(resized > 1.0):
            raise ServerFeatureEncodingError(
                f"{field} resize must preserve finite [0,1] pixels"
            )
        return _immutable_array(resized, dtype=np.dtype(np.float32))

    def preprocess(self, images: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Return canonical 224-square per-camera DINO audit frames.

        LIBERO's validation path already emits 224-square tensors.  RoboTwin's
        baseline path emits 240x320 tensors; each is resized only for this
        semantic/audit branch.  The VAE branch is independently assembled from
        the untouched 240x320 validation outputs in :meth:`prepare`.
        """

        baseline = self._validation_frames(images)
        if self._benchmark_profile == LIBERO_IMAGE_PROFILE:
            return baseline
        return {
            key: self._resize(value, _IMAGE_SIZE, field=f"DINO camera {key!r}")
            for key, value in baseline.items()
        }

    def vae_frames(self, processed_images: Mapping[str, Any]) -> np.ndarray:
        """Concatenate preprocessed cameras and map ``[0,1]`` to ``[-1,1]``."""

        if self._benchmark_profile == ROBOTWIN_IMAGE_PROFILE:
            raise ServerFeatureEncodingError(
                "RoboTwin VAE frames require the independent 240x320 baseline "
                "branch; call prepare(raw_images)"
            )

        if not isinstance(processed_images, Mapping):
            raise TypeError("processed_images must be a camera-keyed mapping")
        if tuple(processed_images) != self._camera_keys:
            raise ServerFeatureEncodingError(
                "processed image keys and order must exactly match camera_keys"
            )
        arrays: list[np.ndarray] = []
        rows: int | None = None
        for key in self._camera_keys:
            array = np.asarray(processed_images[key])
            if array.shape[1:] != (3, *_IMAGE_SIZE) or array.ndim != 4:
                raise ServerFeatureEncodingError(
                    f"processed camera {key!r} must have shape [N,3,224,224]"
                )
            if not np.issubdtype(array.dtype, np.floating):
                raise ServerFeatureEncodingError(
                    f"processed camera {key!r} must be floating point"
                )
            if rows is None:
                rows = int(array.shape[0])
                if rows <= 0:
                    raise ServerFeatureEncodingError("processed cameras must be non-empty")
            elif array.shape[0] != rows:
                raise ServerFeatureEncodingError("all processed cameras must share N")
            if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
                raise ServerFeatureEncodingError(
                    f"processed camera {key!r} must contain finite [0,1] pixels"
                )
            arrays.append(np.asarray(array, dtype=np.float32))

        dimension = -1 if self._concat_mode == "horizontal" else -2
        composite = np.concatenate(arrays, axis=dimension)
        return _immutable_array(composite * 2.0 - 1.0, dtype=np.dtype(np.float32))

    def prepare(self, images: Mapping[str, Any]) -> PreparedFastWAMImages:
        """Preprocess per-camera images and construct the matching VAE frames."""

        if self._benchmark_profile == LIBERO_IMAGE_PROFILE:
            camera_frames = self.preprocess(images)
            vae = self.vae_frames(camera_frames)
        else:
            baseline = self._validation_frames(images)
            camera_frames = {
                key: self._resize(value, _IMAGE_SIZE, field=f"DINO camera {key!r}")
                for key, value in baseline.items()
            }
            head_key, left_key, right_key = self._camera_keys
            head = self._resize(
                baseline[head_key], _ROBOTWIN_HEAD_SIZE, field="RoboTwin head"
            )
            left = self._resize(
                baseline[left_key], _ROBOTWIN_WRIST_SIZE, field="RoboTwin left wrist"
            )
            right = self._resize(
                baseline[right_key], _ROBOTWIN_WRIST_SIZE, field="RoboTwin right wrist"
            )
            bottom = np.concatenate([left, right], axis=-1)
            composite = np.concatenate([head, bottom], axis=-2)
            if composite.shape[-2:] != _ROBOTWIN_COMPOSITE_SIZE:
                raise ServerFeatureEncodingError(
                    "RoboTwin composite does not match RobotVideoDataset [384,320]"
                )
            vae = _immutable_array(
                composite * np.float32(2.0) - np.float32(1.0),
                dtype=np.dtype(np.float32),
            )
        immutable_mapping = MappingProxyType(dict(camera_frames))
        return PreparedFastWAMImages(
            camera_frames=immutable_mapping,
            vae_frames=vae,
            camera_keys=self._camera_keys,
            concat_mode=self._concat_mode,
            benchmark_profile=self._benchmark_profile,
        )


def _require_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - server dependency path
        raise RuntimeError(
            "DINOv2 feature extraction requires torch on the feature server"
        ) from exc


def _require_transformers() -> Any:
    try:
        return importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - server dependency path
        raise RuntimeError(
            "DINOv2 feature extraction requires transformers on the feature server"
        ) from exc


def _channel_statistics(
    image_processor: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    try:
        mean_raw = tuple(image_processor.image_mean)
        std_raw = tuple(image_processor.image_std)
    except (AttributeError, TypeError) as exc:
        raise ServerFeatureEncodingError(
            "AutoImageProcessor must expose image_mean and image_std"
        ) from exc
    if len(mean_raw) != 3 or len(std_raw) != 3:
        raise ServerFeatureEncodingError("DINO image mean/std must have 3 channels")
    try:
        mean = tuple(float(value) for value in mean_raw)
        std = tuple(float(value) for value in std_raw)
    except (TypeError, ValueError) as exc:
        raise ServerFeatureEncodingError("DINO image mean/std must be numeric") from exc
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or any(
        value <= 0.0 for value in std
    ):
        raise ServerFeatureEncodingError(
            "DINO image mean/std must be finite and std must be positive"
        )
    return mean, std


class DinoV2FactualEncoder:
    """Frozen, pinned DINOv2 encoder for independent factual observations."""

    def __init__(
        self,
        model: Any,
        image_processor: Any,
        *,
        model_id: str,
        revision: str,
        device: str,
        torch_backend: Any | None = None,
        torch_dtype: Any | None = None,
        expected_image_size: int | Sequence[int] = _IMAGE_SIZE,
        register_token_count: int | None = None,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ServerFeatureEncodingError("model_id must be a non-empty string")
        if not isinstance(revision, str) or not _PINNED_REVISION.fullmatch(revision):
            raise ServerFeatureEncodingError(
                "revision must be a pinned hexadecimal commit (7-64 characters)"
            )
        if not isinstance(device, str) or not device.strip():
            raise ServerFeatureEncodingError("device must be a non-empty string")
        if not callable(model):
            raise TypeError("DINO model must be callable")
        config = getattr(model, "config", None)
        if config is None:
            raise TypeError("DINO model must expose config")
        if str(getattr(config, "model_type", "")) != "dinov2":
            raise ServerFeatureEncodingError(
                "DINO model config.model_type must be 'dinov2'"
            )

        image_size = _size_pair(expected_image_size, field="expected_image_size")
        # The snapshot's pretraining resolution (e.g. 518 for facebook/
        # dinov2-base) does not have to equal the WARM inference resolution:
        # Dinov2 interpolates position embeddings at forward time. The strict
        # token-count contract in encode() still fails closed if the model
        # does not produce the expected patch grid for ``image_size`` inputs.
        config_image_size = getattr(config, "image_size", None)
        if config_image_size is not None:
            _size_pair(config_image_size, field="model.config.image_size")
        patch_size = _size_pair(
            getattr(config, "patch_size", None), field="model.config.patch_size"
        )
        if image_size[0] % patch_size[0] or image_size[1] % patch_size[1]:
            raise ServerFeatureEncodingError(
                f"image size {image_size} is not divisible by patch size {patch_size}"
            )
        hidden_size = _positive_int(
            getattr(config, "hidden_size", None), field="model.config.hidden_size"
        )
        if register_token_count is None:
            configured_registers = getattr(
                config,
                "num_register_tokens",
                getattr(config, "num_registers", 0),
            )
            registers = _nonnegative_int(
                configured_registers, field="model.config.num_register_tokens"
            )
        else:
            registers = _nonnegative_int(
                register_token_count, field="register_token_count"
            )

        mean, std = _channel_statistics(image_processor)
        eval_method = getattr(model, "eval", None)
        freeze_method = getattr(model, "requires_grad_", None)
        if not callable(eval_method) or not callable(freeze_method):
            raise TypeError("DINO model must expose eval() and requires_grad_()")
        eval_method()
        freeze_method(False)

        self._model = model
        self._image_processor = image_processor
        self._model_id = model_id
        self._revision = revision.lower()
        self._device = device
        self._torch = torch_backend
        self._torch_dtype = torch_dtype
        self._image_size = image_size
        self._patch_size = patch_size
        self._patch_grid = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self._hidden_size = hidden_size
        self._register_token_count = registers
        self._image_mean = mean
        self._image_std = std

    @classmethod
    def from_pretrained(
        cls,
        local_checkpoint: str | Path,
        *,
        model_id: str,
        revision: str,
        device: str,
        torch_dtype: Any | None = None,
        expected_image_size: int | Sequence[int] = _IMAGE_SIZE,
        register_token_count: int | None = None,
        torch_backend: Any | None = None,
        transformers_backend: Any | None = None,
    ) -> "DinoV2FactualEncoder":
        """Load a frozen DINO model and processor from an existing snapshot.

        ``local_checkpoint`` must be a local directory.  ``local_files_only``
        is always set, so this method can never download or silently update a
        model.  ``model_id`` and the hexadecimal ``revision`` are retained as
        the cache provenance contract.  If Transformers exposes the snapshot's
        commit hash, it must match ``revision``.
        """

        checkpoint = Path(local_checkpoint).expanduser().resolve()
        if not checkpoint.is_dir():
            raise ServerFeatureEncodingError(
                f"local_checkpoint is not a directory: {checkpoint}"
            )
        if not isinstance(model_id, str) or not model_id.strip():
            raise ServerFeatureEncodingError("model_id must be a non-empty string")
        if not isinstance(revision, str) or not _PINNED_REVISION.fullmatch(revision):
            raise ServerFeatureEncodingError(
                "revision must be a pinned hexadecimal commit (7-64 characters)"
            )
        torch = torch_backend if torch_backend is not None else _require_torch()
        transformers = (
            transformers_backend
            if transformers_backend is not None
            else _require_transformers()
        )
        auto_model = getattr(transformers, "AutoModel", None)
        auto_processor = getattr(transformers, "AutoImageProcessor", None)
        if auto_model is None or auto_processor is None:
            raise TypeError(
                "transformers backend must expose AutoModel and AutoImageProcessor"
            )

        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "revision": revision,
            "trust_remote_code": False,
        }
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        model = auto_model.from_pretrained(str(checkpoint), **load_kwargs)
        image_processor = auto_processor.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            revision=revision,
            trust_remote_code=False,
        )
        config_commit = getattr(getattr(model, "config", None), "_commit_hash", None)
        if config_commit is not None and str(config_commit).lower() != revision.lower():
            raise ServerFeatureEncodingError(
                "local DINO checkpoint commit does not match pinned revision: "
                f"{config_commit!r} != {revision!r}"
            )
        to_method = getattr(model, "to", None)
        if not callable(to_method):
            raise TypeError("DINO model must expose to()")
        to_method(device=device)
        return cls(
            model,
            image_processor,
            model_id=model_id,
            revision=revision,
            device=device,
            torch_backend=torch,
            torch_dtype=torch_dtype,
            expected_image_size=expected_image_size,
            register_token_count=register_token_count,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def patch_size(self) -> tuple[int, int]:
        return self._patch_size

    @property
    def patch_grid_size(self) -> tuple[int, int]:
        return self._patch_grid

    @property
    def register_token_count(self) -> int:
        return self._register_token_count

    @property
    def image_mean(self) -> tuple[float, float, float]:
        return self._image_mean

    @property
    def image_std(self) -> tuple[float, float, float]:
        return self._image_std

    @property
    def image_size(self) -> tuple[int, int]:
        return self._image_size

    @property
    def compute_device(self) -> str:
        """Exact device string bound by the feature-encoder contract."""

        return self._device

    @property
    def compute_dtype(self) -> str:
        """Canonical dtype used for the frozen DINO forward pass."""

        if self._torch_dtype is None:
            return "float32"
        label = str(self._torch_dtype).strip().lower()
        if label.startswith("torch."):
            label = label[6:]
        return {"float": "float32", "half": "float16"}.get(label, label)

    def encode(self, frames: Any, *, batch_size: int = 32) -> DinoFactualFeatures:
        """Encode already-resized factual frames without processor resize/crop."""

        micro_batch = _positive_int(batch_size, field="batch_size")
        array = np.asarray(frames)
        if not np.issubdtype(array.dtype, np.floating):
            raise ServerFeatureEncodingError(
                "DINO frames must be floating-point FastWAM validation inputs"
            )
        expected_shape = (3, *self._image_size)
        if array.ndim != 4 or array.shape[0] <= 0 or array.shape[1:] != expected_shape:
            raise ServerFeatureEncodingError(
                "DINO frames must have shape "
                f"[N,{expected_shape[0]},{expected_shape[1]},{expected_shape[2]}], "
                f"got {array.shape}"
            )
        if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
            raise ServerFeatureEncodingError("DINO frames must be finite in [0,1]")

        torch = self._torch if self._torch is not None else _require_torch()
        mean = np.asarray(self._image_mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(self._image_std, dtype=np.float32).reshape(1, 3, 1, 1)
        normalized = (
            np.asarray(array, dtype=np.float32) - mean
        ) / std
        cls_batches: list[np.ndarray] = []
        spatial_batches: list[np.ndarray] = []
        inference_mode = getattr(torch, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else nullcontext()
        with context:
            for start in range(0, int(array.shape[0]), micro_batch):
                stop = min(start + micro_batch, int(array.shape[0]))
                tensor = torch.as_tensor(np.ascontiguousarray(normalized[start:stop]))
                to_kwargs: dict[str, Any] = {"device": self._device}
                if self._torch_dtype is not None:
                    to_kwargs["dtype"] = self._torch_dtype
                tensor = tensor.to(**to_kwargs)
                output = self._model(pixel_values=tensor, return_dict=True)
                if not hasattr(output, "last_hidden_state"):
                    raise ServerFeatureEncodingError(
                        "DINO model output must expose last_hidden_state"
                    )
                hidden = np.asarray(
                    _to_numpy("DINO last_hidden_state", output.last_hidden_state),
                    dtype=np.float32,
                )
                expected_tokens = (
                    1
                    + self._register_token_count
                    + self._patch_grid[0] * self._patch_grid[1]
                )
                if hidden.shape != (stop - start, expected_tokens, self._hidden_size):
                    raise ServerFeatureEncodingError(
                        "DINO hidden state violates batch/token/hidden contract: "
                        f"{hidden.shape} != "
                        f"{(stop - start, expected_tokens, self._hidden_size)}"
                    )
                try:
                    factual = extract_dino_factual_features(
                        hidden,
                        patch_grid_size=self._patch_grid,
                        register_token_count=self._register_token_count,
                    )
                except FeaturePrecomputeError as exc:
                    raise ServerFeatureEncodingError(
                        f"invalid DINO factual features: {exc}"
                    ) from exc
                cls_batches.append(factual.cls)
                spatial_batches.append(factual.spatial)

        return DinoFactualFeatures(
            cls=np.concatenate(cls_batches, axis=0),
            spatial=np.concatenate(spatial_batches, axis=0),
        )


__all__ = [
    "DinoV2FactualEncoder",
    "FastWAMImageAdapter",
    "PreparedFastWAMImages",
    "ServerFeatureEncodingError",
]
