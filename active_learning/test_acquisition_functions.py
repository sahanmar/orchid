from coref.const import Doc
from active_learning.acquisition_functions import (
    random_sampling,
    token_sampling,
    mentions_sampling,
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
    # if we sample some batch size, we will sample at least
    # the amount we want to sample due to the possible spans
    assert (
        len(
            token_sampling(deepcopy(dev_data), BATCH_SIZE, 100_000)
            .instances[0]
            .simulation_token_annotations.tokens
        )
        >= BATCH_SIZE
    )

    # if we ask to sample more than we have in docs we will get all
    # tokens from all docs
    assert len(
        token_sampling(deepcopy(dev_data), 1_000_000, 100_000)
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


def test_mentions_sampling(dev_data: list[Doc]) -> None:
    # if we sample some batch size, we will sample at least
    # the amount we want to sample due to the possible spans
    doc_ids = [doc.orchid_id for doc in dev_data if doc.orchid_id is not None]
    mentions = {doc_ids[0]: [1, 10, 100]}
    sampled_mentions = (
        mentions_sampling(deepcopy(dev_data), BATCH_SIZE, 100_000, mentions)
        .instances[0]
        .simulation_token_annotations.tokens
    )

    assert len(sampled_mentions) >= len(mentions)

    # Test multiple docs
    new_docs = [
        deepcopy(dev_data[0]),
        deepcopy(dev_data[0]),
        deepcopy(dev_data[0]),
    ]
    new_docs[0].orchid_id = "id_0"
    new_docs[1].orchid_id = "id_1"
    new_docs[2].orchid_id = "id_3"

    mentions = {
        new_docs[0].orchid_id: [1, 10, 100],
        new_docs[1].orchid_id: [50, 30, 115],
        new_docs[2].orchid_id: [1, 10, 100, 200],
    }

    sampled_tokens = mentions_sampling(new_docs, 1_000_000, 100_000, mentions)

    assert len(
        sampled_tokens.instances[0].simulation_token_annotations.tokens
    ) == len(new_docs[sampled_tokens.indices[0]].cased_words)
    assert len(
        sampled_tokens.instances[1].simulation_token_annotations.tokens
    ) == len(new_docs[sampled_tokens.indices[1]].cased_words)
    assert len(
        sampled_tokens.instances[2].simulation_token_annotations.tokens
    ) == len(new_docs[sampled_tokens.indices[2]].cased_words)
