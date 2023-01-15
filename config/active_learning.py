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
    # Span sampling instead of documents sampling
    span_sampling: bool
    # Active Learning parameters
    parameters_samples: int
    # Active Learning sampling strategy.
    sampling_strategy: GreedySampling

    simulation: Simulation

    @staticmethod
    @overwrite_config
    def load_config(
        span_sampling: bool,
        parameters_samples: int,
        sampling_strategy: dict[str, Any],
        simulation: dict[str, Any],
    ) -> "ActiveLearning":
        return ActiveLearning(
            span_sampling,
            parameters_samples,
            GreedySampling.load_config(**sampling_strategy),
            Simulation(**simulation),
        )
