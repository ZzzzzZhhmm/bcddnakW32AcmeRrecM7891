from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import torch

from .io import ModelConfig, hash_model_file, load_state_dict
from .state_dict_converters import (
    wan_video_vae_state_dict_converter,
)
from fastwam.utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..wan_video_dit import WanVideoDiT
    from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
    from ..wan_video_vae import WanVideoVAE38

logger = get_logger(__name__)
SKIPPED_PRETRAIN_SENTINEL = "SKIPPED_PRETRAIN"


@dataclass
class Wan22LoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE38
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None


@dataclass(frozen=True, slots=True)
class Wan22LoadedVAE:
    """A VAE-only Wan2.2 load result with checkpoint provenance.

    This deliberately contains no Video DiT, text encoder, or tokenizer.  It
    is the supported entry point for offline factual-frame feature extraction.
    """

    vae: WanVideoVAE38
    vae_path: str
    device: str
    torch_dtype: torch.dtype


WAN22_MODEL_REGISTRY = [
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
        "model_hash": "9c8818c2cbea55eca56c7b447df170da",
        "model_name": "wan_video_text_encoder",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors")
        "model_hash": "1f5ab7703c6fc803fdded85ff040c316",
        "model_name": "wan_video_dit",
    },
    {
        # Example: ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth")
        "model_hash": "e1de6c02cdac79f8b739f4d3698cd216",
        "model_name": "wan_video_vae",
        "state_dict_converter": wan_video_vae_state_dict_converter,
    },
]


def _resolve_model_class(model_name: str):
    """Import only the implementation requested by the current load path."""

    if model_name == "wan_video_vae":
        from ..wan_video_vae import WanVideoVAE38

        return WanVideoVAE38
    if model_name == "wan_video_dit":
        from ..wan_video_dit import WanVideoDiT

        return WanVideoDiT
    if model_name == "wan_video_text_encoder":
        from ..wan_video_text_encoder import WanTextEncoder

        return WanTextEncoder
    raise ValueError(f"Unsupported registered Wan2.2 model name: {model_name!r}")


def _validate_dit_config(dit_config: dict[str, Any]) -> dict[str, Any]:
    from ..wan_video_dit import WanVideoDiT

    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must be a dict, got {type(dit_config)}")

    validated = dict(dit_config)

    signature = inspect.signature(WanVideoDiT.__init__)
    allowed_keys = set()
    required_keys = set()
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        allowed_keys.add(name)
        if param.default is inspect.Signature.empty:
            required_keys.add(name)

    unknown_keys = sorted(set(validated) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in `dit_config`: {unknown_keys}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )

    missing_keys = sorted(required_keys - set(validated))
    if missing_keys:
        raise ValueError(
            f"Missing required keys in `dit_config`: {missing_keys}. "
            "Please specify all required WanVideoDiT constructor args."
        )

    return validated


def _load_registered_model(
    path,
    model_name: str,
    torch_dtype: torch.dtype,
    device: str,
    model_kwargs_override: dict[str, Any] | None = None,
):
    model_hash = hash_model_file(path)

    matched_config = None
    for config in WAN22_MODEL_REGISTRY:
        if config["model_hash"] == model_hash and config["model_name"] == model_name:
            matched_config = config
            break
    if matched_config is None:
        raise ValueError(
            f"Cannot detect model type for {model_name}. File: {path}. "
            f"Model hash: {model_hash}. This standalone package follows DiffSynth hash-based loading."
        )

    model_class = _resolve_model_class(model_name)
    model_kwargs = dict(matched_config.get("extra_kwargs", {}))
    if model_kwargs_override is not None:
        model_kwargs.update(model_kwargs_override)
    state_dict_converter = matched_config.get("state_dict_converter")

    model = model_class(**model_kwargs)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device="cpu")
    if state_dict_converter is not None:
        state_dict = state_dict_converter(state_dict)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=torch_dtype)
    return model


def _resolve_configs(model_id: str, tokenizer_model_id: str, redirect_common_files: bool = True):
    dit_config = ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors")
    text_config = ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth")
    vae_config = ModelConfig(model_id=model_id, origin_file_pattern="Wan2.2_VAE.pth")
    tokenizer_config = ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="google/umt5-xxl/")

    if redirect_common_files:
        redirect_dict = {
            "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
            "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
        }
        text_config.model_id, text_config.origin_file_pattern = redirect_dict[text_config.origin_file_pattern]
        vae_config.model_id, vae_config.origin_file_pattern = redirect_dict[vae_config.origin_file_pattern]
    return dit_config, text_config, vae_config, tokenizer_config


def _resolve_vae_config(
    *,
    model_id: str,
    redirect_common_files: bool,
    vae_path: str | Path | None,
) -> ModelConfig:
    """Resolve only the Wan2.2 VAE checkpoint, never a DiT/text artifact."""

    if vae_path is not None:
        path = Path(vae_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Wan2.2 VAE checkpoint does not exist: {path}")
        return ModelConfig(path=str(path.resolve()))

    config = ModelConfig(
        model_id=model_id,
        origin_file_pattern="Wan2.2_VAE.pth",
    )
    if redirect_common_files:
        config.model_id = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
        config.origin_file_pattern = "Wan2.2_VAE.safetensors"
    return config


def _single_checkpoint_path(value: str | list[str] | None) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                "Wan2.2 VAE loading requires exactly one checkpoint file, "
                f"resolved {len(value)} files"
            )
        value = value[0]
    if not isinstance(value, str) or not value:
        raise ValueError("Wan2.2 VAE checkpoint did not resolve to a file")
    return value


def load_wan22_vae_only(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    redirect_common_files: bool = True,
    vae_path: str | Path | None = None,
) -> Wan22LoadedVAE:
    """Load only the frozen Wan2.2 VAE for factual-frame encoding.

    Unlike :func:`load_wan22_ti2v_5b_components`, this function never resolves,
    downloads, constructs, or loads a Video DiT, text encoder, or tokenizer.
    ``vae_path`` can point at an already provisioned checkpoint on a server;
    otherwise the existing DiffSynth ``ModelConfig`` download policy is used.
    """

    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")

    logger.info("Loading Wan2.2 VAE only...")
    start = time.time()
    vae_config = _resolve_vae_config(
        model_id=model_id,
        redirect_common_files=bool(redirect_common_files),
        vae_path=vae_path,
    )
    vae_config.download_if_necessary()
    resolved_path = _single_checkpoint_path(vae_config.path)
    vae = _load_registered_model(
        resolved_path,
        "wan_video_vae",
        torch_dtype=torch_dtype,
        device=device,
    )
    vae.eval().requires_grad_(False)
    logger.info("Finished loading Wan2.2 VAE in %.2f seconds.", time.time() - start)
    return Wan22LoadedVAE(
        vae=vae,
        vae_path=str(Path(resolved_path).resolve()),
        device=device,
        torch_dtype=torch_dtype,
    )


def load_wan22_ti2v_5b_components(
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
    tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
    tokenizer_max_len: int = 512,
    redirect_common_files: bool = True,
    dit_config: dict[str, Any] | None = None,
    skip_dit_load_from_pretrain: bool = False,
    load_text_encoder: bool = True,
):
    logger.info("Loading Wan2.2-TI2V-5B components...")
    start = time.time()

    if dit_config is None:
        raise ValueError("`dit_config` is required for Wan2.2-TI2V-5B loading.")
    validated_dit_config = _validate_dit_config(dit_config)

    dit_model_config, text_config, vae_config, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )

    vae_config.download_if_necessary()
    if load_text_encoder:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()

    if skip_dit_load_from_pretrain:
        from ..wan_video_dit import WanVideoDiT

        logger.info(
            "Skipping pretrained video DiT load (`skip_dit_load_from_pretrain=True`); "
            "initializing video expert randomly and expecting checkpoint override."
        )
        dit: WanVideoDiT = WanVideoDiT(**validated_dit_config).to(device=device, dtype=torch_dtype)
        dit_path = SKIPPED_PRETRAIN_SENTINEL
    else:
        dit_model_config.download_if_necessary()
        dit = _load_registered_model(
            dit_model_config.path,
            "wan_video_dit",
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs_override=validated_dit_config,
        )
        dit_path = str(dit_model_config.path)
    text_encoder: WanTextEncoder | None = None
    tokenizer: HuggingfaceTokenizer | None = None
    text_encoder_path: str | None = None
    tokenizer_path: str | None = None
    if load_text_encoder:
        from ..wan_video_text_encoder import HuggingfaceTokenizer

        text_encoder = _load_registered_model(
            text_config.path,
            "wan_video_text_encoder",
            torch_dtype=torch_dtype,
            device=device,
        )
        tokenizer = HuggingfaceTokenizer(
            name=tokenizer_config.path,
            seq_len=int(tokenizer_max_len),
            clean="whitespace",
        )
        text_encoder_path = str(text_config.path)
        tokenizer_path = str(tokenizer_config.path)
    else:
        logger.info(
            "Skipping pretrained text encoder/tokenizer load (`load_text_encoder=False`); "
            "training must provide cached `context/context_mask`."
        )
    vae: WanVideoVAE38 = _load_registered_model(vae_config.path, "wan_video_vae", torch_dtype=torch_dtype, device=device)
    logger.info("Finished loading Wan2.2-TI2V-5B components in %.2f seconds.", time.time() - start)
    return Wan22LoadedComponents(
        dit=dit,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        dit_path=dit_path,
        vae_path=str(vae_config.path),
        text_encoder_path=text_encoder_path,
        tokenizer_path=tokenizer_path,
    )
