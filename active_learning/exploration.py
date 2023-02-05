from enum import Enum
import math

from dataclasses import dataclass, field
from typing import Callable, Dict, List
from random import random

from coref.const import Doc, SampledData
from active_learning.acquisition_functions import (
    # random_sampling,
    token_sampling,
)


class AcquisitionFunctionsType(Enum):
    random = "random"


acquisition_func_typing = Callable[[List[Doc], int, bool], SampledData]


ACQUISITION_FUNCTION_MAPPER: Dict[
    AcquisitionFunctionsType, acquisition_func_typing
] = {AcquisitionFunctionsType.random: token_sampling}


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

        exhaust_document - strategy flag which allows to samples all tokens from a document,
        given the batch size. ( TODO Will not be binary in future iterations )

        batch_size - Batch size to sample

        strategy_flip - Prob[random_strategy|current_sampling_iteration/total_number_of_iterations] = 0.5
        In other words strategy flip is the number of iterations after which the acquisition function
        strategy will have higher probability to be chosen

        total_number_of_iterations - the total number of planned samplings
    """

    acquisition_function_type: AcquisitionFunctionsType
    exhaust_document: bool
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

    def step(self, instances: List[Doc]) -> SampledData:
        ##### This check is done because mypy is bitching to that #####
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
                instances, self.batch_size, self.exhaust_document
            )
        return self.acquisition_function(
            instances, self.batch_size, self.exhaust_document
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
        exhaust_document: bool,
        batch_size: int,
        strategy_flip: float,
        total_number_of_iterations: int,
    ) -> "GreedySampling":
        return GreedySampling(
            AcquisitionFunctionsType(acquisition_function_type),
            exhaust_document,
            batch_size,
            strategy_flip,
            total_number_of_iterations,
        )
