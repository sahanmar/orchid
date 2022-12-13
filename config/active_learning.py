from active_learning.exploration import GreedySampling
from config.config_utils import overwrite_config
from dataclasses import dataclass

from typing import Dict, Any


@dataclass
class Simulation:
    # Number of instances used for the first training iteration
    initial_sample_size: int
    # Active learning steps to perform
    active_learning_steps: int


@dataclass
class ActiveLearning:
    # Active Learning parameters
    parameters_samples: int
    # Active Learning sampling strategy.
    sampling_strategy: GreedySampling

    simulation: Simulation

    @staticmethod
    @overwrite_config
    def load_config(
        parameters_samples: int,
        sampling_strategy: dict[str, Any],
        simulation: dict[str, Any],
    ) -> "ActiveLearning":
        return ActiveLearning(
            parameters_samples,
            GreedySampling.load_config(**sampling_strategy),
            Simulation(**simulation),
        )
