from active_learning.exploration import GreedySampling
from config.config_utils import overwrite_config
from dataclasses import dataclass

from typing import Dict, Any


@dataclass
class ActiveLearning:
    # Active Learning parameters
    parameters_samples: int
    # Active Learning sampling strategy.
    sampling_strategy: GreedySampling

    @staticmethod
    @overwrite_config
    def load_config(
        parameters_samples: int, sampling_strategy: Dict[str, Any]
    ) -> "ActiveLearning":
        return ActiveLearning(
            parameters_samples, GreedySampling.load_config(**sampling_strategy)
        )
