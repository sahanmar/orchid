from enum import Enum
import math

from dataclasses import dataclass, field
from typing import Callable, Union
from random import random

from coref.const import Doc, SampledData
from active_learning.acquisition_functions import (
    # random_sampling,
    token_sampling,
    mentions_sampling,
)


class AcquisitionFunctionsType(Enum):
    random_token = "random_token"
    random_mention = "random_mention"
    entropy_mention = "entropy_mention"


acquisition_func_typing = Callable[
    [list[Doc], int, int, dict[str, list[int]]], SampledData
]


ACQUISITION_FUNCTION_MAPPER: dict[
    AcquisitionFunctionsType, acquisition_func_typing
] = {
    AcquisitionFunctionsType.random_token: token_sampling,
    AcquisitionFunctionsType.random_mention: mentions_sampling,
}


@dataclass
class NaiveSampling:
    docs_of_interest: int
    batch_size: int
    acquisition_function_type: AcquisitionFunctionsType
    total_number_of_iterations: int

    def __post_init__(self) -> None:
        self.acquisition_function = ACQUISITION_FUNCTION_MAPPER[
            self.acquisition_function_type
        ]

    def step(
        self, instances: list[Doc], indices: dict[str, list[int]]
    ) -> SampledData:
        return self.acquisition_function(
            instances, self.batch_size, self.docs_of_interest, indices
        )

    @staticmethod
    def load_config(
        docs_of_interest: int,
        batch_size: int,
        acquisition_function_type: str,
        total_number_of_iterations: int,
    ) -> "NaiveSampling":
        return NaiveSampling(
            docs_of_interest,
            batch_size,
            AcquisitionFunctionsType(acquisition_function_type),
            total_number_of_iterations,
        )


@dataclass
class GreedySampling:
    """
    Pseudo epsilon greedy strategy. We formalize the task for two possible actions.
    The conventional greedy strategy selects an action with the highest expected reward
    with prob 1 - 'epsilon' and then selects randomly from all possible actions with
    probability 'epsilon'.

    In our case we select a random strategy with the decaying probability 'epsilon' and
    the acquisition function strategy with the probability 1 - epsilon.

    The decaying occurs with the iterative change of the random strategy to another.

    args:
        acquisition_function_type - the function name to sample new data

        docs_of_interest - strategy which prioritizes taking tokens from 0th to docs_of_interest-th
        idex, given the batch size

        batch_size - Batch size to sample

        strategy_flip - Prob[random_strategy|current_sampling_iteration/total_number_of_iterations] = 0.5
        In other words strategy flip is the number of iterations after which the acquisition function
        strategy will have higher probability to be chosen

        total_number_of_iterations - the total number of planned samplings
    """

    acquisition_function_type: AcquisitionFunctionsType
    docs_of_interest: int
    batch_size: int
    strategy_flip: float
    total_number_of_iterations: int
    acquisition_function: acquisition_func_typing = field(init=False)
    epsilon_greedy_prob: float = field(init=False)
    current_sampling_iteration: int = field(init=False)
    flip_iteration: int = field(init=False)
    normalizing_coef: float = field(init=False)

    def __post_init__(self) -> None:
        self.acquisition_function = ACQUISITION_FUNCTION_MAPPER[
            self.acquisition_function_type
        ]
        self.epsilon_greedy_prob = 1.0
        self.current_sampling_iteration = 0
        self.flip_iteration = int(
            self.strategy_flip * self.total_number_of_iterations
        )
        # coefficient that squeezes sigmoid's boundaries to have (almost) 0 prob
        # for current_sampling_iteration = 0
        self.normalizing_coef = 5 / self.flip_iteration

    def step(self, instances: list[Doc]) -> SampledData:
        ##### This check is done because of mypy #####
        if (
            self.epsilon_greedy_prob is None
            or self.current_sampling_iteration is None
        ):
            raise (
                ValueError(
                    "Post init wasn't called. We have some problems, chief!..."
                )
            )
        ##### end of check #####
        self.current_sampling_iteration += 1
        self.epsilon_greedy_prob = 1 - self.sigmoid()
        if random() <= self.epsilon_greedy_prob:
            return token_sampling(
                instances, self.batch_size, self.docs_of_interest
            )
        return self.acquisition_function(
            instances, self.batch_size, self.docs_of_interest, {}
        )

    def sigmoid(self) -> float:
        return 1 / (
            1
            + math.exp(
                -(self.current_sampling_iteration - self.flip_iteration)
                * self.normalizing_coef
            )
        )

    @staticmethod
    def load_config(
        acquisition_function_type: str,
        docs_of_interest: int,
        batch_size: int,
        strategy_flip: float,
        total_number_of_iterations: int,
    ) -> "GreedySampling":
        return GreedySampling(
            AcquisitionFunctionsType(acquisition_function_type),
            docs_of_interest,
            batch_size,
            strategy_flip,
            total_number_of_iterations,
        )
