from __future__ import annotations

from pathlib import Path

import pytest

from fastwam.memory.processor_contract import (
    extract_m1_libero_processor_recipe,
    validate_processor_instance,
)


def test_real_hydra_processor_instance_matches_m1_recipe() -> None:
    """Exercise Hydra's real ListConfig/object conversion, not a list mock."""

    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    hydra_utils = pytest.importorskip("hydra.utils")
    omegaconf = pytest.importorskip("omegaconf")

    repo = Path(__file__).resolve().parents[2]
    raw = omegaconf.OmegaConf.load(repo / "configs" / "data" / "libero_2cam.yaml")
    wrapped = omegaconf.OmegaConf.create({"data": raw})
    omegaconf.OmegaConf.resolve(wrapped)
    resolved = omegaconf.OmegaConf.to_container(wrapped, resolve=True)
    recipe = extract_m1_libero_processor_recipe(resolved)

    processor = hydra_utils.instantiate(wrapped.data.train.processor)
    validate_processor_instance(processor, recipe)

