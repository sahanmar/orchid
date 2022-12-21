from typing import List, Tuple
from pytest import approx

from config import Config
from coref.const import Doc


def test_exploration(config: Config, dev_data: list[Doc]) -> None:
    # data, config = get_default_setup()
    assert (
        len(dev_data) == 1
    )  # By default there is only one document in debug test
    augmented_data = 10 * dev_data  # lets make 10 for test purposed
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
