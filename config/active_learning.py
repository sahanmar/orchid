from active_learning.exploration import GreedySampling, NaiveSampling
from config.config_utils import overwrite_config
from dataclasses import dataclass

from typing import Any, Union
from enum import Enum


class SamplingStrategy(Enum):
    naive_sampling = "naive_sampling"
    greedy_sampling = "greedy_sampling"


class InstanceSampling(Enum):
    document = "document"
    token = "token"
    mention = "mention"


@dataclass
class Simulation:
    # Number of instances used for the first training iteration
    initial_sample_size: int
    # Active learning loops to perform
    active_learning_loops: int


@dataclass
class ActiveLearning:
    # Instance type to sample
    instance_sampling: InstanceSampling
    # Active Learning parameters
    parameters_samples: int

    strategy_type: SamplingStrategy

    # Active Learning sampling strategy.
    sampling_strategy: Union[GreedySampling, NaiveSampling]

    simulation: Simulation

    @staticmethod
    @overwrite_config
    def load_config(
        instance_sampling: str,
        parameters_samples: int,
        strategy: str,
        sampling_strategy: dict[str, Any],
        simulation: dict[str, Any],
    ) -> "ActiveLearning":
        strategy_type = SamplingStrategy(strategy)
        if strategy_type == SamplingStrategy.greedy_sampling:
            strategy_config: Union[
                GreedySampling, NaiveSampling
            ] = GreedySampling.load_config(**sampling_strategy[strategy])
        if strategy_type == SamplingStrategy.naive_sampling:
            strategy_config = NaiveSampling.load_config(
                **sampling_strategy[strategy]
            )

        return ActiveLearning(
            InstanceSampling(instance_sampling),
            parameters_samples,
            strategy_type,
            strategy_config,
            Simulation(**simulation),
        )
