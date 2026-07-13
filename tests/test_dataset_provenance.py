from __future__ import annotations

import pytest

from fastwam.datasets.provenance import copy_provenance, global_episode_id


class ScalarLike:
    def __init__(self, value: int):
        self.value = value

    def item(self) -> int:
        return self.value


def test_copy_provenance_preserves_values_and_ignores_unrelated_fields() -> None:
    source = {
        "dataset_index": ScalarLike(3),
        "episode_index": ScalarLike(7),
        "frame_index": 11,
        "timestamp": 0.55,
        "task_index": 2,
        "action": "large payload",
    }
    destination: dict[str, object] = {"idx": 100}

    result = copy_provenance(source, destination)

    assert result is destination
    assert destination["dataset_index"] is source["dataset_index"]
    assert destination["episode_index"] is source["episode_index"]
    assert "action" not in destination


def test_global_episode_id_uses_dataset_and_episode_pair() -> None:
    assert global_episode_id(
        {"dataset_index": ScalarLike(1), "episode_index": ScalarLike(4)}
    ) == (1, 4)
    assert global_episode_id({"dataset_index": 2, "episode_index": 4}) != (1, 4)


@pytest.mark.parametrize(
    "sample,exception",
    [
        ({"dataset_index": 1}, KeyError),
        ({"dataset_index": -1, "episode_index": 0}, ValueError),
        ({"dataset_index": True, "episode_index": 0}, TypeError),
    ],
)
def test_global_episode_id_rejects_unsafe_metadata(sample, exception) -> None:
    with pytest.raises(exception):
        global_episode_id(sample)
