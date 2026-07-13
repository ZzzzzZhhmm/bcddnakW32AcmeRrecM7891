from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from fastwam.memory import feature_precompute_vae as module
from fastwam.memory.feature_precompute_vae import (
    FactualVAEEncodingError,
    encode_wan22_factual_frames,
)


class _FakeTensor:
    def __init__(self, value: object):
        self.value = np.asarray(value)

    @property
    def ndim(self) -> int:
        return self.value.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def __getitem__(self, item: object) -> "_FakeTensor":
        return _FakeTensor(self.value[item])

    def unsqueeze(self, dimension: int) -> "_FakeTensor":
        return _FakeTensor(np.expand_dims(self.value, axis=dimension))

    def detach(self) -> "_FakeTensor":
        return self

    def to(self, *, device: str, dtype=None) -> "_FakeTensor":
        del device
        if dtype is None:
            return _FakeTensor(self.value)
        return _FakeTensor(self.value.astype(dtype))

    def numpy(self) -> np.ndarray:
        return np.array(self.value, copy=True)


def _adaptive_average_pool(value: _FakeTensor, *, output_size: tuple[int, int]):
    data = value.value
    out_h, out_w = output_size
    if data.shape[-2] % out_h or data.shape[-1] % out_w:
        raise AssertionError("test fake only supports divisible pooling shapes")
    h_stride = data.shape[-2] // out_h
    w_stride = data.shape[-1] // out_w
    pooled = data.reshape(
        data.shape[0],
        data.shape[1],
        out_h,
        h_stride,
        out_w,
        w_stride,
    ).mean(axis=(3, 5))
    return _FakeTensor(pooled)


class _FakeTorch:
    float32 = np.float32
    nn = SimpleNamespace(
        functional=SimpleNamespace(adaptive_avg_pool2d=_adaptive_average_pool)
    )

    @staticmethod
    def as_tensor(value: object) -> _FakeTensor:
        return value if isinstance(value, _FakeTensor) else _FakeTensor(value)

    @staticmethod
    def is_floating_point(value: _FakeTensor) -> bool:
        return np.issubdtype(value.dtype, np.floating)

    @staticmethod
    def cat(values: list[_FakeTensor], dim: int) -> _FakeTensor:
        return _FakeTensor(np.concatenate([value.value for value in values], axis=dim))

    @staticmethod
    def inference_mode():
        return nullcontext()


class _FakeVAE:
    upsampling_factor = 1

    def __init__(self, *, output_timesteps: int = 1):
        self.calls: list[tuple[int, ...]] = []
        self.output_timesteps = output_timesteps

    def parameters(self):
        return iter((SimpleNamespace(dtype=np.float16),))

    def single_encode(self, video: _FakeTensor, device: str) -> _FakeTensor:
        assert device == "cuda:7"
        self.calls.append(video.shape)
        per_frame = video.value.mean(axis=1, keepdims=True)
        if self.output_timesteps != 1:
            per_frame = np.repeat(per_frame, self.output_timesteps, axis=2)
        return _FakeTensor(np.concatenate((per_frame, per_frame + 1), axis=1))

    def encode(self, *_args, **_kwargs):
        raise AssertionError("whole-video encode must never be called")


@pytest.fixture(autouse=True)
def _fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_require_torch", lambda: _FakeTorch)


def test_factual_encoder_microbatches_singleton_time_frames() -> None:
    frames = np.arange(5 * 3 * 8 * 16, dtype=np.float32).reshape(5, 3, 8, 16)
    vae = _FakeVAE()

    result = encode_wan22_factual_frames(
        vae,
        frames,
        device="cuda:7",
        batch_size=2,
    )

    assert vae.calls == [(2, 3, 1, 8, 16), (2, 3, 1, 8, 16), (1, 3, 1, 8, 16)]
    assert result.shape == (5, 2, 4, 8)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous


def test_factual_encoder_rejects_episode_video_tensor() -> None:
    vae = _FakeVAE()
    with pytest.raises(FactualVAEEncodingError, match="episode/video tensors are forbidden"):
        encode_wan22_factual_frames(
            vae,
            np.zeros((1, 3, 5, 8, 16), dtype=np.float32),
            device="cuda:7",
        )
    assert vae.calls == []


def test_factual_encoder_rejects_multi_timestep_latent() -> None:
    with pytest.raises(FactualVAEEncodingError, match="more than one latent timestep"):
        encode_wan22_factual_frames(
            _FakeVAE(output_timesteps=2),
            np.zeros((1, 3, 8, 16), dtype=np.float32),
            device="cuda:7",
        )


def test_factual_encoder_rejects_unprocessed_uint8_frames() -> None:
    with pytest.raises(FactualVAEEncodingError, match="canonical image preprocessor"):
        encode_wan22_factual_frames(
            _FakeVAE(),
            np.zeros((1, 3, 8, 16), dtype=np.uint8),
            device="cuda:7",
        )


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_factual_encoder_rejects_invalid_batch_size(batch_size: object) -> None:
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        encode_wan22_factual_frames(
            _FakeVAE(),
            np.zeros((1, 3, 8, 16), dtype=np.float32),
            device="cuda:7",
            batch_size=batch_size,  # type: ignore[arg-type]
        )
