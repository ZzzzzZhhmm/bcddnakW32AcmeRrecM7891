from __future__ import annotations

import os

import pytest

if os.environ.get("WARM_REQUIRE_TORCH_TESTS") == "1":
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - server contract
        raise RuntimeError(
            "WARM_REQUIRE_TORCH_TESTS=1 but PyTorch is unavailable"
        ) from error
else:
    torch = pytest.importorskip("torch")

from fastwam.datasets.warm_candidates import WARM_CANDIDATE_FIELDS, WARM_QUERY_SPLIT
from fastwam.datasets.warm_retrospective import (
    WARM_CANDIDATE_EVENT_ORDINAL,
    WARM_CANDIDATE_NORMALIZED_PHASE,
    WARM_RETROSPECTIVE_FIELDS,
)
from fastwam.trainer import WARM_EVAL_TENSOR_RANKS, Wan22Trainer


def test_eval_tensor_ranks_cover_retrospective_fields() -> None:
    required = (set(WARM_RETROSPECTIVE_FIELDS) | set(WARM_CANDIDATE_FIELDS)) - {
        WARM_QUERY_SPLIT
    }
    missing = sorted(required - set(WARM_EVAL_TENSOR_RANKS))
    assert missing == []


def test_eval_batch_keeps_candidate_phase_and_ordinal() -> None:
    sample = {
        "video": torch.zeros(3, 2, 8, 8),
        "prompt": "pick up the banana and place it in the box",
        "action": torch.zeros(4, 7),
        "proprio": torch.zeros(4, 7),
        WARM_CANDIDATE_NORMALIZED_PHASE: torch.linspace(0, 1, 32),
        WARM_CANDIDATE_EVENT_ORDINAL: torch.arange(32, dtype=torch.long),
        WARM_QUERY_SPLIT: "dev",
    }
    batched = Wan22Trainer._to_batched_eval_sample(sample)
    assert batched[WARM_CANDIDATE_NORMALIZED_PHASE].shape == (1, 32)
    assert batched[WARM_CANDIDATE_EVENT_ORDINAL].shape == (1, 32)
    assert batched[WARM_QUERY_SPLIT] == "dev"
    assert torch.equal(
        batched[WARM_CANDIDATE_NORMALIZED_PHASE][0],
        sample[WARM_CANDIDATE_NORMALIZED_PHASE],
    )
    assert torch.equal(
        batched[WARM_CANDIDATE_EVENT_ORDINAL][0],
        sample[WARM_CANDIDATE_EVENT_ORDINAL],
    )
