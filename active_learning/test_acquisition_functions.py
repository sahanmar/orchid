from coref.const import Doc
from active_learning.acquisition_functions import (
    random_sampling,
    token_sampling,
)
from copy import deepcopy

BATCH_SIZE = 7
TOO_LARGE_BATCH_SIZE = 11


def test_random_sampling(dev_data: list[Doc]) -> None:
    instances = 10 * dev_data  # we have only one debug doc, so make 10
    sampled_data = random_sampling(instances, BATCH_SIZE)
    assert len(sampled_data.indices) == BATCH_SIZE
    assert sampled_data.instances[0].document_id == "bc/cctv/00/cctv_0000"


def test_random_sampling_w_too_large_batch_size(dev_data: list[Doc]) -> None:
    sampled_data = random_sampling(dev_data, TOO_LARGE_BATCH_SIZE)
    assert len(sampled_data.indices) == len(dev_data)
    assert sampled_data.instances[0].document_id == "bc/cctv/00/cctv_0000"


def test_token_sampling(dev_data: list[Doc]) -> None:
    # if we some batch size, we will sample at least
    # the amount we want to sample due to the possible spans
    assert (
        len(
            token_sampling(dev_data, BATCH_SIZE, 100_000)
            .instances[0]
            .simulation_token_annotations.tokens
        )
        >= BATCH_SIZE
    )

    # if ask to sample more that we have in docs we will get all
    # tokens from all docs
    assert len(
        token_sampling(dev_data, 1_000_000, 100_000)
        .instances[0]
        .simulation_token_annotations.tokens
    ) == len(dev_data[0].cased_words)

    # Test multiple docs
    new_docs = [
        deepcopy(dev_data[0]),
        deepcopy(dev_data[0]),
        deepcopy(dev_data[0]),
    ]
    new_docs[0].orchid_id = "id_0"
    new_docs[1].orchid_id = "id_1"
    new_docs[2].orchid_id = "id_3"

    sampled_tokens = token_sampling(new_docs, 1_000_000, 100_000)

    assert len(
        sampled_tokens.instances[0].simulation_token_annotations.tokens
    ) == len(new_docs[sampled_tokens.indices[0]].cased_words)
    assert len(
        sampled_tokens.instances[1].simulation_token_annotations.tokens
    ) == len(new_docs[sampled_tokens.indices[1]].cased_words)
    assert len(
        sampled_tokens.instances[2].simulation_token_annotations.tokens
    ) == len(new_docs[sampled_tokens.indices[2]].cased_words)
