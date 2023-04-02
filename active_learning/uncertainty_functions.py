import torch
from typing import cast


def entropy(scores: torch.Tensor) -> torch.Tensor:
    """
    Returns entropy calculated on softmax scores.

    scores size = [n, n (sofmax output)]
    """
    return cast(
        torch.Tensor, torch.distributions.Categorical(probs=scores).entropy()
    )
