from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fastwam.memory.candidate_cache import QueryId
from fastwam.memory.online_episode_controller import (
    FactualObservationBindingError,
    OnlineEpisodeController,
    bind_factual_world_tokens,
)


class _Retriever:
    def __init__(self) -> None:
        self.episode: int | None = None

    def begin_episode(self, episode: int) -> None:
        self.episode = episode

    def make_query_id(self, frame: int) -> QueryId:
        assert self.episode is not None
        return QueryId("rmbench", 0, self.episode, frame)


class _History:
    def model_kwargs(self):
        return {"episode_tokens": np.ones((1, 2), dtype=np.float32)}

    def evidence(self):
        return {"history": True}


class _Memory:
    def __init__(self) -> None:
        self.pending = None
        self.records = []

    def begin_episode(self, episode: int) -> None:
        self.episode = episode

    def history_inputs(self, *, executed_actions_since_previous):
        self.pending = executed_actions_since_previous
        return _History()

    def record_factual_observation(
        self, *, frame_index, factual_payload, executed_actions_since_previous
    ):
        self.records.append(
            (frame_index, factual_payload, executed_actions_since_previous)
        )
        return {"frame_index": frame_index}

    def end_episode(self):
        return {"sealed": True}


def _controller() -> OnlineEpisodeController:
    return OnlineEpisodeController(
        contract=SimpleNamespace(),
        retriever=_Retriever(),
        retrospective_episode_memory=_Memory(),
    )


def test_controller_exposes_only_executed_prefix_then_commits_real_feature() -> None:
    controller = _controller()
    controller.begin_episode(3)
    query = controller.issue_query_id(0)
    assert query.episode_index == 3
    kwargs, evidence = controller.retrospective_history_kwargs()
    assert kwargs["episode_tokens"].shape == (1, 2)
    assert evidence == {"history": True}

    controller.note_executed_action(
        np.arange(14, dtype=np.float32),
        model_space_action=np.arange(14, dtype=np.float32) / 2,
    )
    controller.issue_query_id(1)
    controller.retrospective_history_kwargs()
    memory = controller.retrospective_episode_memory
    assert memory.pending.shape == (1, 14)

    result = controller.commit_factual_replan_observation(
        frame_index=1,
        model_output={"warm_factual_observation": {"semantic_tokens": [1]}},
    )
    assert result["executed_environment_prefix_count"] == 1
    assert memory.records[-1][2].shape == (1, 14)
    sealed = controller.end_episode()
    assert sealed["sealed"] is True
    assert controller.active_episode_index is None


def test_controller_rejects_reused_or_nonmonotonic_identity() -> None:
    controller = _controller()
    controller.begin_episode(0)
    controller.issue_query_id(5)
    with pytest.raises(ValueError, match="increase strictly"):
        controller.issue_query_id(5)
    controller.end_episode()
    with pytest.raises(ValueError, match="already evaluated"):
        controller.begin_episode(0)


def test_controller_requires_explicit_episode_seal() -> None:
    controller = _controller()
    controller.begin_episode(0)
    with pytest.raises(RuntimeError, match="end_episode"):
        controller.begin_episode(1)


def test_factual_token_binding_replaces_only_bridge_world_tokens() -> None:
    bridge = np.full((4, 3), -7.0, dtype=np.float32)
    vae = np.full((2, 2), 5.0, dtype=np.float32)
    proprio = np.arange(14, dtype=np.float32)
    output = {
        "action": np.zeros((1, 32, 14), dtype=np.float32),
        "warm_factual_observation": {
            "world_tokens": bridge,
            "vae_latent": vae,
            "proprio": proprio,
        },
    }
    source = np.arange(12, dtype=np.float32).reshape(4, 3)
    tokens = np.frombuffer(source.tobytes(), dtype=np.float32).reshape(4, 3)

    replaced = bind_factual_world_tokens(output, tokens)

    assert replaced is not output
    assert replaced["warm_factual_observation"] is not output[
        "warm_factual_observation"
    ]
    assert replaced["warm_factual_observation"]["world_tokens"] is tokens
    assert replaced["warm_factual_observation"]["vae_latent"] is vae
    assert replaced["warm_factual_observation"]["proprio"] is proprio
    assert output["warm_factual_observation"]["world_tokens"] is bridge

    with pytest.raises(FactualObservationBindingError, match="immutable"):
        bind_factual_world_tokens(output, source)
    wrong_shape = np.frombuffer(
        np.zeros((4, 2), dtype=np.float32).tobytes(), dtype=np.float32
    ).reshape(4, 2)
    with pytest.raises(FactualObservationBindingError, match="shapes differ"):
        bind_factual_world_tokens(output, wrong_shape)
