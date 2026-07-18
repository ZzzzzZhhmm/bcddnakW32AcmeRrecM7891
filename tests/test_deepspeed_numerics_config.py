from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("stage", (1, 2))
def test_deepspeed_bf16_configs_block_overflow_before_update(stage: int) -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "ds_configs" / f"ds_zero{stage}_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["zero_optimization"]["stage"] == stage
    assert payload["gradient_clipping"] == "auto"
    assert payload["bf16"] == {
        "enabled": "auto",
        "check_grad_overflow": True,
    }
