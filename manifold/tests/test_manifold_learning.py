"""
Testing basic functionality of the manifold learning sub-package.
"""
import pytest
import torch

from config import Config
from manifold.dataset import get_embedded_2d_ellipse
from manifold.linear import BasePCA


@pytest.fixture(scope="module")
def manifold_learning_test_data(config: Config) -> torch.Tensor:
    assert config.manifold.standalone.input_dimensionality is not None
    assert config.manifold.standalone.output_dimensionality is not None

    data = get_embedded_2d_ellipse(
        dim=config.manifold.standalone.input_dimensionality,
        a=5,
        y0=10,
        x0=5,
        noise=1,
    )
    assert data.shape[1] == config.manifold.standalone.input_dimensionality
    return data


def test_standalone_manifold_learning(
    config: Config,
    manifold_learning_test_data: torch.Tensor,
) -> None:
    # Run BasePCA
    pca_custom = BasePCA.from_config(
        config=config,
    )
    pca_custom.train_on_inputs(inputs=manifold_learning_test_data)
    output = pca_custom(manifold_learning_test_data)
    assert (
        output.embeddings.shape[1]
        == config.manifold.standalone.output_dimensionality
    )
