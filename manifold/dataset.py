"""
Here lie the basics of working with Manifold Learning datasets
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


# region PyTorch Manifold Data Manipulations


class ManifoldDataset(Dataset):
    def __init__(self, inputs: torch.Tensor):
        super(ManifoldDataset, self).__init__()
        self._x = inputs

    def __getitem__(self, item: int) -> torch.Tensor:
        return self._x[item]

    def __len__(self) -> int:
        return int(self._x.size(0))


class ManifoldDataloader(DataLoader):
    def __init__(
        self, inputs: torch.Tensor, batch_size: int, shuffle: bool = True
    ):
        super(ManifoldDataloader, self).__init__(
            dataset=ManifoldDataset(inputs=inputs),
            batch_size=batch_size,
            shuffle=shuffle,
        )


# endregion

# region Artificial Datasets


def get_embedded_2d_ellipse(
    dim: int = 2,
    a: float = 1.0,
    b: float = 1.0,
    x0: float = 0.0,
    y0: float = 0,
    steps: int = 100,
    noise: float = 1e-2,
    random_state: Optional[float] = None,
) -> torch.Tensor:
    """
    Generate a 2D ellipsis that is then embedded into a multidimensional space.

    Parameters
    ----------

    steps : int
        The amount of points to sample in one direction (X/Y)
    noise : float
        The amount of noise to add to extra dimensions
    random_state : float
        Random state for reproducibility
    y0 : float
        Y-coordinate of the center
    x0 : float
        X-coordinate of the center
    b : float
        Semi-minor axis
    a : float
        Semi-major axis
    dim : int
        Target dimension of the ellipsis, must be >= 2

    """
    assert dim >= 2
    assert a > 0
    assert b > 0
    assert steps > 2
    assert noise > 0
    phi = torch.linspace(-torch.pi, torch.pi, steps=steps)

    x = torch.permute(
        torch.linspace(0.0, a, steps=steps).expand([1, steps]), dims=[1, 0]
    ) * torch.cos(phi)
    x = x.view(steps**2) + x0
    x = x.expand([1, x.size(0)])
    y = torch.permute(
        torch.linspace(0.0, b, steps=steps).expand([1, steps]), dims=[1, 0]
    ) * torch.sin(phi)
    y = y.view(steps**2) + y0
    y = y.expand([1, y.size(0)])

    # Embedding the ellipsis into a space of a higher dimension
    extra_dim = torch.zeros(dim - 2, x.size(1))
    if random_state is not None:
        torch.set_rng_state(
            new_state=torch.tensor(random_state, dtype=torch.int)
        )
    # Add noise to extra dimensions
    noise_tensor = torch.randn(*extra_dim.size()) * noise
    extra_dim += noise_tensor
    ell = torch.cat((x, y, extra_dim), dim=0)
    ell = torch.permute(ell, dims=[1, 0])
    return ell  # [n, dim]


# endregion
