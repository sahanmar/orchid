from pytest import approx

from config import Config
from coref.const import Doc
from active_learning.exploration import GreedySampling, NaiveSampling
from coref.models import load_coref_model
from copy import deepcopy


def test_greedy_sampling(config: Config, dev_data: list[Doc]) -> None:

    assert isinstance(config.active_learning.sampling_strategy, GreedySampling)

    assert (
        len(dev_data) == 1
    )  # By default there is only one document in debug test

    augmented_data = [
        deepcopy(dev_data[0]) for _ in range(10)
    ]  # lets make 10 for test purposed

    for i, doc in enumerate(augmented_data):
        doc.orchid_id = str(i)
    mentions = {
        doc.orchid_id: [(i, 0.0) for i in range(len(doc.cased_words))]
        for doc in augmented_data
        if doc.orchid_id
    }

    # First step
    sampled_data = config.active_learning.sampling_strategy.step(
        augmented_data, mentions
    )
    assert (
        len(sampled_data.instances)
        == config.active_learning.sampling_strategy.batch_size
    )
    assert (
        config.active_learning.sampling_strategy.current_sampling_iteration == 1
    )
    # Second step
    sampled_data = config.active_learning.sampling_strategy.step(
        augmented_data, mentions
    )
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


def test_naive_sampling(al_config: Config, dev_data: list[Doc]) -> None:

    assert isinstance(al_config.active_learning.sampling_strategy, NaiveSampling)  # type: ignore

    model = load_coref_model(al_config)
    sampled_data = model.sample_unlabled_data(deepcopy(dev_data))

    sampled_mentions = sampled_data.instances[
        0
    ].simulation_token_annotations.tokens

    print(sampled_mentions)

    assert (
        len(sampled_mentions)
        >= al_config.active_learning.sampling_strategy.batch_size  # type: ignore
    )
