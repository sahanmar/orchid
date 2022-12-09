import torch
import pytest
from config import Config
from config.metrics import PAVPU
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


def test_pavpu_metric_static_threshold(
    data: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    pred, target = data
    expected = [pytest.approx(0.4, 0.01)]
    assert (
        pavpu_metric(
            pred,
            target,
            PAVPU(static_theshold_value=0.5, sliding_threshold=False),
        )
        == expected
    )


def test_pavpu_metric_sliding_threshold(
    data: Tuple[torch.Tensor, torch.Tensor]
) -> None:
    pred, target = data
    expected = [
        pytest.approx(i, 0.01)
        for i in [0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6]
    ]
    assert (
        pavpu_metric(
            pred,
            target,
            PAVPU(static_theshold_value=0.5, sliding_threshold=True),
        )
        == expected
    )
