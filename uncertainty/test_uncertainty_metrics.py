import torch
import pytest
from config import Config
from typing import Tuple
from uncertainty.uncertainty_metrics import pavpu_metric

CONFIG = Config.load_default_config(section="debug")


@pytest.fixture()
def data() -> Tuple[torch.Tensor, torch.Tensor]:
    pred = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 0.5, 0.25, 0.25],
            [1, 0, 0, 0],
            [0.24, 0.26, 0.25, 0.25],
            [0, 0, 0, 1],
        ],
        device=CONFIG.training_params.device,
    )

    target = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        device=CONFIG.training_params.device,
    )
    return pred, target


def test_pavpu_metric(data: Tuple[torch.Tensor, torch.Tensor]) -> None:
    threshold = CONFIG.metrics.pavpu.static_theshold_value
    pred, target = data
    expected = pytest.approx(0.4, 0.01)
    assert (
        pavpu_metric(pred, target, uncertainty_threshold=threshold) == expected
    )
