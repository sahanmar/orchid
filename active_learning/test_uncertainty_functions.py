import torch

from active_learning.uncertainty_functions import entropy
import pytest


def test_entropy() -> None:
    scores = torch.Tensor([[1, 0], [0.5, 0.5], [0, 1]])
    expected = [0, 0.7, 0]
    assert all(
        [
            pytest.approx(exp, abs=0.1) == score
            for exp, score in zip(expected, entropy(scores).tolist())
        ]
    )
