from typing import List, Tuple
from pytest import approx

from config import Config
from coref.const import Doc
from coref.data_utils import get_docs, DataType


def get_default_setup() -> Tuple[List[Doc], Config]:
    config = Config.load_default_config(section="debug")
    data = get_docs(DataType.test, config)
    return data, config


def test_exploration() -> None:
    data, config = get_default_setup()
    assert len(data) == 1  # By default there is only one document in debug test
    augmented_data = 10 * data  # lets make 10 for test purposed
    # First step
    sampled_data = config.active_learning.sampling_strategy.step(augmented_data)
    assert (
        len(sampled_data.instances)
        == config.active_learning.sampling_strategy.batch_size
    )
    assert (
        config.active_learning.sampling_strategy.current_sampling_iteration == 1
    )
    # Second step
    sampled_data = config.active_learning.sampling_strategy.step(augmented_data)
    assert (
        len(sampled_data.instances)
        == config.active_learning.sampling_strategy.batch_size
    )
    assert (
        config.active_learning.sampling_strategy.current_sampling_iteration == 2
    )
    assert (
        config.active_learning.sampling_strategy.epsilon_greedy_prob
        == approx(0.5)
    )
