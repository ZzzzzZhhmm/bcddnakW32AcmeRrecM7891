from __future__ import annotations

from contextlib import nullcontext
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fastwam.memory.server_feature_encoders as encoder_module
from fastwam.memory.server_feature_encoders import (
    DinoV2FactualEncoder,
    FastWAMImageAdapter,
    PreparedFastWAMImages,
    ROBOTWIN_IMAGE_PROFILE,
    ServerFeatureEncodingError,
)


def test_module_import_has_no_eager_torch_or_transformers_binding() -> None:
    module = importlib.import_module("fastwam.memory.server_feature_encoders")
    assert "torch" not in module.__dict__
    assert "transformers" not in module.__dict__


class _ToFloat:
    def __init__(self, camera: str, calls: list[tuple[str, np.dtype]]) -> None:
        self.camera = camera
        self.calls = calls

    def __call__(self, value: np.ndarray) -> np.ndarray:
        self.calls.append((self.camera, value.dtype))
        assert value.dtype == np.uint8
        return value.astype(np.float32) / 255.0


class _ForbiddenProcessorPreprocess:
    def __call__(self, _value: object) -> object:
        raise AssertionError("processor preprocess must never be called")


def _camera_frames(fill: float, *, rows: int = 2) -> np.ndarray:
    return np.full((rows, 3, 224, 224), fill, dtype=np.float32)


def test_image_adapter_uses_camera_val_transforms_and_exact_uint8_roundtrip() -> None:
    calls: list[tuple[str, np.dtype]] = []
    processor = SimpleNamespace(
        val_transforms={
            "image": [_ToFloat("image", calls)],
            "wrist_image": [_ToFloat("wrist_image", calls)],
        },
        preprocess=_ForbiddenProcessorPreprocess(),
    )
    adapter = FastWAMImageAdapter(
        processor,
        ("image", "wrist_image"),
        "horizontal",
        tensor_factory=lambda value: value,
    )
    image = _camera_frames(0.5)
    wrist = _camera_frames(0.25)

    prepared = adapter.prepare({"image": image, "wrist_image": wrist})

    assert isinstance(prepared, PreparedFastWAMImages)
    assert adapter.camera_keys == ("image", "wrist_image")
    assert adapter.concat_mode == "horizontal"
    assert calls == [("image", np.dtype(np.uint8)), ("wrist_image", np.dtype(np.uint8))]
    assert prepared.camera_frames["image"].shape == (2, 3, 224, 224)
    assert prepared.vae_frames.shape == (2, 3, 224, 448)
    # NumPy uint8 conversion truncates exactly like torch.to(torch.uint8).
    np.testing.assert_allclose(
        prepared.camera_frames["image"][1, 1, 1, 1],
        np.float32(127.0 / 255.0),
    )
    np.testing.assert_allclose(
        prepared.vae_frames[:, :, :, :224],
        prepared.camera_frames["image"] * 2.0 - 1.0,
    )
    np.testing.assert_allclose(
        prepared.vae_frames[:, :, :, 224:],
        prepared.camera_frames["wrist_image"] * 2.0 - 1.0,
    )
    for value in (*prepared.camera_frames.values(), prepared.vae_frames):
        assert value.dtype == np.float32
        assert value.flags.c_contiguous
        assert not value.flags.writeable


def test_image_adapter_vertical_concat_and_global_transform_list() -> None:
    calls: list[tuple[str, np.dtype]] = []
    processor = SimpleNamespace(val_transforms=[_ToFloat("shared", calls)])
    adapter = FastWAMImageAdapter(
        processor,
        ["left", "right"],
        "vertical",
        tensor_factory=lambda value: value,
    )

    processed = adapter.preprocess(
        {"left": _camera_frames(0.0, rows=1), "right": _camera_frames(1.0, rows=1)}
    )
    vae = adapter.vae_frames(processed)

    assert calls == [("shared", np.dtype(np.uint8)), ("shared", np.dtype(np.uint8))]
    assert vae.shape == (1, 3, 448, 224)
    np.testing.assert_array_equal(vae[:, :, :224], -1.0)
    np.testing.assert_array_equal(vae[:, :, 224:], 1.0)


def test_image_adapter_rejects_wrong_camera_contract_or_transform_range() -> None:
    processor = SimpleNamespace(val_transforms=[lambda value: value.astype(np.float32)])
    adapter = FastWAMImageAdapter(
        processor,
        ["image"],
        "horizontal",
        tensor_factory=lambda value: value,
    )
    with pytest.raises(ServerFeatureEncodingError, match="keys and order"):
        adapter.preprocess({"other": _camera_frames(0.5)})
    with pytest.raises(ServerFeatureEncodingError, match=r"pixels in \[0,1\]"):
        adapter.preprocess({"image": _camera_frames(1.01)})
    with pytest.raises(ServerFeatureEncodingError, match=r"finite \[0,1\]"):
        adapter.preprocess({"image": _camera_frames(0.5)})


class _RobotwinToFloat:
    def __call__(self, value: np.ndarray) -> np.ndarray:
        assert value.dtype == np.uint8
        assert value.shape[-2:] == (240, 320)
        return value.astype(np.float32) / 255.0


def _constant_resize(value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Sufficient reference resize for constant-image layout parity."""

    rows, channels = value.shape[:2]
    scalar = value[:, :, :1, :1]
    return np.broadcast_to(scalar, (rows, channels, *size)).copy()


def test_robotwin_dual_path_matches_robot_video_dataset_composite_layout() -> None:
    processor = SimpleNamespace(val_transforms=[_RobotwinToFloat()])
    adapter = FastWAMImageAdapter(
        processor,
        ("cam_high", "cam_left_wrist", "cam_right_wrist"),
        "robotwin",
        benchmark_profile=ROBOTWIN_IMAGE_PROFILE,
        tensor_factory=lambda value: value,
        spatial_resize=_constant_resize,
    )
    frames = {
        "cam_high": np.full((2, 3, 240, 320), 0.25, dtype=np.float32),
        "cam_left_wrist": np.full((2, 3, 240, 320), 0.5, dtype=np.float32),
        "cam_right_wrist": np.full((2, 3, 240, 320), 0.75, dtype=np.float32),
    }

    prepared = adapter.prepare(frames)

    assert prepared.benchmark_profile == "robotwin"
    assert prepared.concat_mode == "robotwin"
    assert adapter.benchmark_profile == "robotwin"
    assert prepared.camera_frames["cam_high"].shape == (2, 3, 224, 224)
    assert prepared.vae_frames.shape == (2, 3, 384, 320)
    # Exact RobotVideoDataset ordering: full-width head, then left/right wrists.
    head = np.float32(int(0.25 * 255.0) / 255.0 * 2.0 - 1.0)
    left = np.float32(int(0.5 * 255.0) / 255.0 * 2.0 - 1.0)
    right = np.float32(int(0.75 * 255.0) / 255.0 * 2.0 - 1.0)
    np.testing.assert_allclose(prepared.vae_frames[:, :, :256, :], head, atol=1e-7)
    np.testing.assert_allclose(
        prepared.vae_frames[:, :, 256:, :160], left, atol=1e-7
    )
    np.testing.assert_allclose(
        prepared.vae_frames[:, :, 256:, 160:], right, atol=1e-7
    )
    for value in (*prepared.camera_frames.values(), prepared.vae_frames):
        assert value.dtype == np.float32
        assert not value.flags.writeable


def test_robotwin_profile_rejects_non_exact_layout_and_direct_vae_shortcut() -> None:
    processor = SimpleNamespace(val_transforms=[_RobotwinToFloat()])
    with pytest.raises(ServerFeatureEncodingError, match="exactly three"):
        FastWAMImageAdapter(
            processor,
            ("head", "left"),
            "robotwin",
            benchmark_profile="robotwin",
        )
    adapter = FastWAMImageAdapter(
        processor,
        ("head", "left", "right"),
        "robotwin",
        benchmark_profile="robotwin",
        tensor_factory=lambda value: value,
        spatial_resize=_constant_resize,
    )
    with pytest.raises(ServerFeatureEncodingError, match="call prepare"):
        adapter.vae_frames(
            {
                key: np.zeros((1, 3, 224, 224), dtype=np.float32)
                for key in adapter.camera_keys
            }
        )


class _FakeTensor:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)

    def to(self, *, device: str, dtype: object | None = None) -> "_FakeTensor":
        assert device == "cuda:3"
        if dtype is None:
            return _FakeTensor(self.value)
        return _FakeTensor(self.value.astype(dtype))

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return np.array(self.value, copy=True)


class _FakeTorch:
    @staticmethod
    def as_tensor(value: object) -> _FakeTensor:
        return _FakeTensor(value)

    @staticmethod
    def inference_mode():
        return nullcontext()


class _BFloatLikeTensor:
    """Tensor double whose direct NumPy conversion mimics torch.bfloat16."""

    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)

    def detach(self) -> "_BFloatLikeTensor":
        return self

    def cpu(self) -> "_BFloatLikeTensor":
        return self

    def numpy(self) -> np.ndarray:
        raise TypeError("Got unsupported ScalarType BFloat16")

    def to(self, *, device: str, dtype: object) -> _FakeTensor:
        assert device == "cpu"
        assert dtype is np.float32
        return _FakeTensor(self.value.astype(np.float32))


def test_tensor_to_numpy_explicitly_converts_bfloat16_like_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        encoder_module,
        "_require_torch",
        lambda: SimpleNamespace(float32=np.float32),
    )

    result = encoder_module._to_numpy(
        "bf16 hidden",
        _BFloatLikeTensor([[1.0, 2.0]]),
    )

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [[1.0, 2.0]])


class _FakeImageProcessor:
    image_mean = [0.5, 0.25, 0.0]
    image_std = [0.5, 0.25, 2.0]

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("AutoImageProcessor resize/crop path must never run")


class _FakeDinoModel:
    def __init__(
        self,
        *,
        token_delta: int = 0,
        nonfinite: bool = False,
        commit: str | None = None,
        model_type: str = "dinov2",
    ) -> None:
        self.config = SimpleNamespace(
            model_type=model_type,
            image_size=4,
            patch_size=2,
            hidden_size=3,
            num_register_tokens=1,
            _commit_hash=commit,
        )
        self.token_delta = token_delta
        self.nonfinite = nonfinite
        self.eval_calls = 0
        self.freeze_values: list[bool] = []
        self.to_calls: list[str] = []
        self.inputs: list[np.ndarray] = []

    def eval(self) -> "_FakeDinoModel":
        self.eval_calls += 1
        return self

    def requires_grad_(self, value: bool) -> "_FakeDinoModel":
        self.freeze_values.append(value)
        return self

    def to(self, *, device: str) -> "_FakeDinoModel":
        self.to_calls.append(device)
        return self

    def __call__(self, *, pixel_values: _FakeTensor, return_dict: bool):
        assert return_dict is True
        self.inputs.append(np.array(pixel_values.value, copy=True))
        rows = pixel_values.value.shape[0]
        tokens = 1 + 1 + 4 + self.token_delta
        hidden = np.arange(rows * tokens * 3, dtype=np.float32).reshape(rows, tokens, 3)
        if self.nonfinite:
            hidden[0, 0, 0] = np.nan
        return SimpleNamespace(last_hidden_state=_FakeTensor(hidden))


def _fake_encoder(model: _FakeDinoModel) -> DinoV2FactualEncoder:
    return DinoV2FactualEncoder(
        model,
        _FakeImageProcessor(),
        model_id="facebook/dinov2-test",
        revision="a" * 40,
        device="cuda:3",
        torch_backend=_FakeTorch,
        expected_image_size=(4, 4),
    )


def test_dino_encoder_manual_normalization_batching_and_spatial_extraction() -> None:
    model = _FakeDinoModel()
    encoder = _fake_encoder(model)
    frames = np.zeros((3, 3, 4, 4), dtype=np.float32)
    frames[:, 0] = 1.0
    frames[:, 1] = 0.5
    frames[:, 2] = 0.25

    features = encoder.encode(frames, batch_size=2)

    assert model.eval_calls == 1
    assert model.freeze_values == [False]
    assert [value.shape[0] for value in model.inputs] == [2, 1]
    np.testing.assert_allclose(model.inputs[0][:, 0], 1.0)
    np.testing.assert_allclose(model.inputs[0][:, 1], 1.0)
    np.testing.assert_allclose(model.inputs[0][:, 2], 0.125)
    assert features.cls.shape == (3, 3)
    assert features.spatial.shape == (3, 4, 3)
    assert encoder.hidden_size == 3
    assert encoder.patch_size == (2, 2)
    assert encoder.patch_grid_size == (2, 2)
    assert encoder.register_token_count == 1
    assert encoder.image_mean == (0.5, 0.25, 0.0)
    assert encoder.image_std == (0.5, 0.25, 2.0)
    assert not features.cls.flags.writeable
    assert not features.spatial.flags.writeable


@pytest.mark.parametrize(
    ("model", "match"),
    [
        (_FakeDinoModel(token_delta=-1), "token/hidden contract"),
        (_FakeDinoModel(nonfinite=True), "finite"),
    ],
)
def test_dino_encoder_rejects_bad_token_count_or_nonfinite_hidden(
    model: _FakeDinoModel, match: str
) -> None:
    encoder = _fake_encoder(model)
    with pytest.raises(ServerFeatureEncodingError, match=match):
        encoder.encode(np.zeros((1, 3, 4, 4), dtype=np.float32))


def test_dino_encoder_rejects_unprocessed_or_wrong_sized_frames() -> None:
    encoder = _fake_encoder(_FakeDinoModel())
    with pytest.raises(ServerFeatureEncodingError, match="floating-point"):
        encoder.encode(np.zeros((1, 3, 4, 4), dtype=np.uint8))
    with pytest.raises(ServerFeatureEncodingError, match="shape"):
        encoder.encode(np.zeros((1, 3, 8, 8), dtype=np.float32))
    invalid = np.zeros((1, 3, 4, 4), dtype=np.float32)
    invalid[0, 0, 0, 0] = np.inf
    with pytest.raises(ServerFeatureEncodingError, match="finite"):
        encoder.encode(invalid)


def test_dino_encoder_rejects_non_dinov2_model_type() -> None:
    with pytest.raises(ServerFeatureEncodingError, match="model_type"):
        _fake_encoder(_FakeDinoModel(model_type="vit"))


class _FakeAutoModel:
    calls: list[tuple[str, dict[str, object]]] = []
    model: _FakeDinoModel

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> _FakeDinoModel:
        cls.calls.append((path, dict(kwargs)))
        return cls.model


class _FakeAutoImageProcessor:
    calls: list[tuple[str, dict[str, object]]] = []

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> _FakeImageProcessor:
        cls.calls.append((path, dict(kwargs)))
        return _FakeImageProcessor()


def test_from_pretrained_is_local_pinned_and_frozen(tmp_path: Path) -> None:
    revision = "b" * 40
    _FakeAutoModel.calls = []
    _FakeAutoImageProcessor.calls = []
    _FakeAutoModel.model = _FakeDinoModel(commit=revision)
    backend = SimpleNamespace(
        AutoModel=_FakeAutoModel,
        AutoImageProcessor=_FakeAutoImageProcessor,
    )
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()

    encoder = DinoV2FactualEncoder.from_pretrained(
        checkpoint,
        model_id="facebook/dinov2-small",
        revision=revision,
        device="cuda:3",
        torch_dtype=np.float16,
        expected_image_size=4,
        torch_backend=_FakeTorch,
        transformers_backend=backend,
    )

    assert encoder.model_id == "facebook/dinov2-small"
    assert encoder.revision == revision
    assert _FakeAutoModel.model.to_calls == ["cuda:3"]
    model_path, model_kwargs = _FakeAutoModel.calls[0]
    processor_path, processor_kwargs = _FakeAutoImageProcessor.calls[0]
    assert Path(model_path) == checkpoint.resolve()
    assert Path(processor_path) == checkpoint.resolve()
    assert model_kwargs == {
        "local_files_only": True,
        "revision": revision,
        "trust_remote_code": False,
        "torch_dtype": np.float16,
    }
    assert processor_kwargs == {
        "local_files_only": True,
        "revision": revision,
        "trust_remote_code": False,
    }


def test_from_pretrained_rejects_moving_revision_or_commit_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "snapshot"
    checkpoint.mkdir()
    backend = SimpleNamespace(
        AutoModel=_FakeAutoModel,
        AutoImageProcessor=_FakeAutoImageProcessor,
    )
    with pytest.raises(ServerFeatureEncodingError, match="pinned hexadecimal"):
        DinoV2FactualEncoder.from_pretrained(
            checkpoint,
            model_id="facebook/dinov2-small",
            revision="main",
            device="cuda:3",
            torch_backend=_FakeTorch,
            transformers_backend=backend,
        )

    _FakeAutoModel.model = _FakeDinoModel(commit="c" * 40)
    with pytest.raises(ServerFeatureEncodingError, match="does not match"):
        DinoV2FactualEncoder.from_pretrained(
            checkpoint,
            model_id="facebook/dinov2-small",
            revision="d" * 40,
            device="cuda:3",
            expected_image_size=4,
            torch_backend=_FakeTorch,
            transformers_backend=backend,
        )
