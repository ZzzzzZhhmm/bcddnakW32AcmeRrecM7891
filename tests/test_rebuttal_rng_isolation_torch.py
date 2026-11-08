"""Counterfactual diagnostics cannot alter the factual random stream."""
import random
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from fastwam.research.query_capture import isolated_generation_rng


def test_counterfactual_rng_is_restored_even_when_generation_raises():
    random.seed(3407);np.random.seed(3407);torch.manual_seed(3407)
    expected=(random.random(),np.random.rand(),torch.rand(3))
    random.seed(3407);np.random.seed(3407);torch.manual_seed(3407)
    with pytest.raises(RuntimeError,match='diagnostic'):
        with isolated_generation_rng():
            random.seed(17);random.random();np.random.seed(17);np.random.rand();torch.manual_seed(17);torch.rand(11)
            raise RuntimeError('diagnostic failed')
    assert random.random()==expected[0] and np.random.rand()==expected[1]
    assert torch.equal(torch.rand(3),expected[2])
