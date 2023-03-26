from coref.const import Doc
from active_learning.acquisition_functions import (
    random_sampling,
    mentions_sampling,
)
from copy import deepcopy
from typing import Tuple, Optional
from config import Config
from coref.models import load_coref_model
from config.active_learning import InstanceSampling
from itertools import chain, repeat

BATCH_SIZE = 7
TOO_LARGE_BATCH_SIZE = 11


# def get_mentions(
#     docs: list[Doc], pred_mentions: dict[Optional[str], set[int]] = {}
# ) -> dict[str, list[Tuple[int, float]]]:
#     return {
#         doc.orchid_id: [
#             (i, 1.0 if i in pred_mentions.get(doc.orchid_id, set()) else 0.0)
#             for i in range(len(doc.cased_words))
#         ]
#         for doc in docs
#         if doc.orchid_id
#     }


# def test_random_sampling(dev_data: list[Doc]) -> None:
#     instances = 10 * dev_data  # we have only one debug doc, so make 10
#     sampled_data = random_sampling(instances, BATCH_SIZE)
#     assert len(sampled_data.indices) == BATCH_SIZE
#     assert sampled_data.instances[0].document_id == "bc/cctv/00/cctv_0000"


# def test_random_sampling_w_too_large_batch_size(dev_data: list[Doc]) -> None:
#     sampled_data = random_sampling(dev_data, TOO_LARGE_BATCH_SIZE)
#     assert len(sampled_data.indices) == len(dev_data)
#     assert sampled_data.instances[0].document_id == "bc/cctv/00/cctv_0000"


# def test_token_sampling(dev_data: list[Doc]) -> None:
#     # if we sample some batch size, we will sample at least
#     # the amount we want to sample due to the possible spans

#     mentions = get_mentions(dev_data)

#     assert (
#         len(
#             mentions_sampling(deepcopy(dev_data), BATCH_SIZE, 100_000, mentions)
#             .instances[0]
#             .simulation_token_annotations.tokens
#         )
#         >= BATCH_SIZE
#     )

#     # if we ask to sample more than we have in docs we will get all
#     # tokens from all docs

#     assert (
#         len(
#             mentions_sampling(deepcopy(dev_data), 1_000_000, 100_000, mentions)
#             .instances[0]
#             .simulation_token_annotations.tokens
#         )
#         >= BATCH_SIZE
#     )

#     # Test multiple docs
#     new_docs = [
#         deepcopy(dev_data[0]),
#         deepcopy(dev_data[0]),
#         deepcopy(dev_data[0]),
#     ]
#     new_docs[0].orchid_id = "id_0"
#     new_docs[1].orchid_id = "id_1"
#     new_docs[2].orchid_id = "id_3"

#     mentions = get_mentions(new_docs)

#     sampled_tokens = mentions_sampling(new_docs, 1_000_000, 100_000, mentions)

#     assert len(sampled_tokens.instances[0].simulation_token_annotations.tokens) == len(
#         new_docs[sampled_tokens.indices[0]].cased_words
#     )
#     assert len(sampled_tokens.instances[1].simulation_token_annotations.tokens) == len(
#         new_docs[sampled_tokens.indices[1]].cased_words
#     )
#     assert len(sampled_tokens.instances[2].simulation_token_annotations.tokens) == len(
#         new_docs[sampled_tokens.indices[2]].cased_words
#     )


# def test_mentions_sampling(dev_data: list[Doc]) -> None:
#     # if we sample some batch size, we will sample at least
#     # the amount we want to sample due to the possible spans
#     pred_mentions = {dev_data[0].orchid_id: {1, 10, 100}}

#     mentions = get_mentions(dev_data, pred_mentions)

#     sampled_mentions = (
#         mentions_sampling(deepcopy(dev_data), BATCH_SIZE, 100_000, mentions)
#         .instances[0]
#         .simulation_token_annotations.tokens
#     )

#     assert len(sampled_mentions) >= len(mentions.values())

#     # Test multiple docs
#     new_docs = [
#         deepcopy(dev_data[0]),
#         deepcopy(dev_data[0]),
#         deepcopy(dev_data[0]),
#     ]
#     new_docs[0].orchid_id = "id_0"
#     new_docs[1].orchid_id = "id_1"
#     new_docs[2].orchid_id = "id_3"

#     pred_mentions = {
#         new_docs[0].orchid_id: {1, 10, 100},
#         new_docs[1].orchid_id: {50, 30, 115},
#         new_docs[2].orchid_id: {1, 10, 100, 200},
#     }

#     mentions = get_mentions(new_docs, pred_mentions)

#     sampled_tokens = mentions_sampling(new_docs, 1_000_000, 100_000, mentions)

#     assert len(sampled_tokens.instances[0].simulation_token_annotations.tokens) == len(
#         new_docs[sampled_tokens.indices[0]].cased_words
#     )
#     assert len(sampled_tokens.instances[1].simulation_token_annotations.tokens) == len(
#         new_docs[sampled_tokens.indices[1]].cased_words
#     )
#     assert len(sampled_tokens.instances[2].simulation_token_annotations.tokens) == len(
#         new_docs[sampled_tokens.indices[2]].cased_words
#     )


# def test_entropy_sampling(dev_data: list[Doc]) -> None:
#     # if we sample some batch size, we will sample at least
#     # the amount we want to sample due to the possible spans
#     doc_ids = [doc.orchid_id for doc in dev_data if doc.orchid_id is not None]
#     mentions = {doc_ids[0]: [(1, 0.0), (10, 0.5), (100, 1.0)]}
#     sampled_mentions = (
#         mentions_sampling(deepcopy(dev_data), BATCH_SIZE, 100_000, mentions)
#         .instances[0]
#         .simulation_token_annotations.tokens
#     )

#     assert len(sampled_mentions) >= len(mentions.values())

#     # Check if ordering is correct
#     sampled_mentions = (
#         mentions_sampling(deepcopy(dev_data), 1, 100_000, mentions)
#         .instances[0]
#         .simulation_token_annotations.tokens
#     )

#     assert 100 in sampled_mentions and len(sampled_mentions) == 1

#     # Test multiple docs
#     new_docs = [
#         deepcopy(dev_data[0]),
#         deepcopy(dev_data[0]),
#         deepcopy(dev_data[0]),
#     ]
#     new_docs[0].orchid_id = "id_0"
#     new_docs[1].orchid_id = "id_1"
#     new_docs[2].orchid_id = "id_3"

#     mentions = {
#         new_docs[0].orchid_id: [(1, 0.5), (10, 0.5), (100, 0.5)],
#         new_docs[1].orchid_id: [(50, 0.5), (30, 0.5), (115, 0.5)],
#         new_docs[2].orchid_id: [(1, 0.5), (10, 0.5), (100, 0.5), (200, 0.5)],
#     }

#     sampled_tokens = mentions_sampling(new_docs, 1_000_000, 100_000, mentions)

#     assert len(sampled_tokens.instances[0].simulation_token_annotations.tokens) == len(
#         mentions[sampled_tokens.instances[0].orchid_id]  # type:ignore
#     )
#     assert len(sampled_tokens.instances[1].simulation_token_annotations.tokens) == len(
#         mentions[sampled_tokens.instances[1].orchid_id]  # type:ignore
#     )
#     assert len(sampled_tokens.instances[2].simulation_token_annotations.tokens) == len(
#         mentions[sampled_tokens.instances[2].orchid_id]  # type:ignore
#     )


def test_hac_entropy_sampling(dev_data: list[Doc], al_config: Config) -> None:
    config = deepcopy(al_config)
    config.active_learning.instance_sampling = (
        InstanceSampling.hac_entropy_mention
    )
    model = load_coref_model(config)

    # Test multiple docs
    new_docs = []
    for i, doc in zip(range(5), repeat(dev_data[0])):
        new_doc = deepcopy(doc)
        new_doc.orchid_id = f"id_{i}"
        new_docs.append(new_doc)

    sampled_data = model.sample_unlabeled_data(new_docs)
    sampled_tokens = list(
        chain.from_iterable(
            [
                doc.simulation_token_annotations.tokens
                for doc in sampled_data.instances
            ]
        )
    )
    assert (
        len(sampled_tokens)
        >= al_config.active_learning.sampling_strategy.batch_size  # type: ignore
    )
