from pytest import approx

from config import Config
from coref.const import Doc
from active_learning.exploration import GreedySampling, NaiveSampling
from coref.models import load_coref_model
from run import output_running_time


# def test_greedy_sampling(config: Config, dev_data: list[Doc]) -> None:

#     assert isinstance(config.active_learning.sampling_strategy, GreedySampling)

#     assert len(dev_data) == 1  # By default there is only one document in debug test
#     augmented_data = 10 * dev_data  # lets make 10 for test purposed
#     # First step
#     sampled_data = config.active_learning.sampling_strategy.step(augmented_data)
#     assert (
#         len(sampled_data.instances)
#         == config.active_learning.sampling_strategy.batch_size
#     )
#     assert config.active_learning.sampling_strategy.current_sampling_iteration == 1
#     # Second step
#     sampled_data = config.active_learning.sampling_strategy.step(augmented_data)
#     assert (
#         len(sampled_data.instances)
#         == config.active_learning.sampling_strategy.batch_size
#     )
#     assert config.active_learning.sampling_strategy.current_sampling_iteration == 2
#     assert config.active_learning.sampling_strategy.epsilon_greedy_prob == approx(0.5)


def test_naive_sampling(al_config: Config, dev_data: list[Doc]) -> None:

    assert isinstance(al_config.active_learning.sampling_strategy, NaiveSampling)  # type: ignore

    model = load_coref_model(al_config)
    sampled_data = model.sample_unlabled_data(dev_data)

    sampled_mentions = sampled_data.instances[0].simulation_token_annotations.tokens

    print(sampled_mentions)

    assert (
        len(sampled_mentions)
        >= al_config.active_learning.sampling_strategy.naive_sampling.batch.size  # type: ignore
    )
