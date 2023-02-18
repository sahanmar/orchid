from typing import Tuple
from random import sample, choice

from coref.const import Doc, SampledData, Optional
from copy import deepcopy


def random_sampling(instances: list[Doc], batch_size: int) -> SampledData:
    """
    Provides random sampling from given instances
    """
    if len(instances) < batch_size:
        indices = list(range(0, len(instances)))
    else:
        indices = sample(list(range(0, len(instances))), batch_size)
    return SampledData(indices, [instances[i] for i in indices])


def token_sampling(
    docs: list[Doc],
    token_batch: int,
    docs_of_interest: int,
    _: list[list[int]] = [],
) -> SampledData:
    """
    The method samples random tokens from docs in the following way:

    Repeat until # sampled tokens equals to the token_batch
        - Sample a doc (
            if docs_of_interest are set, then tokens will be
            preferred from first docs_of_interest docs
        )
        - Sample a token

        if a token belongs to a coreference span and has other mentions
            - take the full span and find the closest mention
        else
            - take the token as it is

        if the doc was already sampled before
            - extend the simulation tokens field and save a doc
        else
            - save a doc and write in simulation tokens

    return all created docs with sampled tokens
    """

    sampled_tokens_counter = 0
    counter = 0  # variable to control infinite looping
    sampled_data = SampledData([], [])
    doc_w_their_position = {doc.orchid_id: i for i, doc in enumerate(docs)}
    while sampled_tokens_counter < token_batch:
        # sample the doc and solve the use-cases
        sampled_doc_ids_w_order_id = {
            d.orchid_id: i for i, d in enumerate(sampled_data.instances) if d.orchid_id
        }
        doc = deepcopy(
            _choose_the_doc_for_token_sampling(
                docs, sampled_data, sampled_doc_ids_w_order_id, docs_of_interest
            )
        )
        # if no tokens to sample return what we already have
        if doc is None:
            sampled_data.indices = [
                doc_w_their_position[doc.orchid_id] for doc in sampled_data.instances
            ]
            return sampled_data

        # choose new tokens given that we already sampled from the doc
        if doc.orchid_id in sampled_doc_ids_w_order_id:
            sampled_tokens = sampled_data.instances[
                sampled_doc_ids_w_order_id[doc.orchid_id]
            ].simulation_token_annotations.tokens
            token = choice(
                [
                    token_id
                    for token_id in range(len(doc.cased_words))
                    if token_id not in sampled_tokens
                ]
            )
        else:
            token = choice(list(range(len(doc.cased_words))))

        token_in_cluster = _get_coref_if_token_in_cluster(token, doc.span_clusters)
        if token_in_cluster is None:
            token_in_cluster = {token}

        doc_tokens_number = len(doc.simulation_token_annotations.tokens)
        doc.simulation_token_annotations.tokens = (
            doc.simulation_token_annotations.tokens.union(token_in_cluster)
        )
        sampled_tokens_counter += (
            len(doc.simulation_token_annotations.tokens) - doc_tokens_number
        )

        docs[doc_w_their_position[doc.orchid_id]] = deepcopy(doc)
        if doc.orchid_id in sampled_doc_ids_w_order_id:
            sampled_data.instances[sampled_doc_ids_w_order_id[doc.orchid_id]] = doc
        else:
            sampled_data.instances.append(doc)

        if counter == 1_000_000:
            raise ValueError("Smth went wrong... The loop got infinite")
        counter += 1

    sampled_data.indices = [
        doc_w_their_position[doc.orchid_id] for doc in sampled_data.instances
    ]
    return sampled_data


def _get_coref_if_token_in_cluster(
    token: int, span_clusters: list[list[Tuple[int, int]]]
) -> Optional[set[int]]:
    """
    If a chosen token is in a coreference cluster then return the span tokens and
    the closest mention. If not, returns None
    """
    for cluster in span_clusters:
        for i, span in enumerate(cluster):
            start, end = span
            if token in range(start, end):
                if i == 0:
                    closest_span = cluster[i + 1]
                else:
                    closest_span = cluster[i - 1]
                return set(
                    [
                        token
                        for start, end in [span, closest_span]
                        for token in range(start, end)
                    ]
                )
    return None


def _choose_the_doc_for_token_sampling(
    docs: list[Doc],
    sampled_data: SampledData,
    sampled_doc_ids_w_order_id: dict[str, int],
    docs_of_interest: int,
) -> Optional[Doc]:
    """
    docs: list of Docs to sample from
    sampled_data: the docs that the tokens were already sampled from
    sampled_doc_ids_w_order_id: mapping of sampled doc ids to their
    order in the SampleData instances list
    docs_of_interest: strategy that will always prefer selection from the
    docs_of_interest indices.

    Returns doc and its id in the docs list

    Edge cases:
        - If all tokens were sampled from the doc, returns None
    """
    if not docs:
        return None
    doc_id = (
        choice(range(len(docs)))
        if len(docs) < docs_of_interest
        else choice(range(docs_of_interest))
    )
    doc = docs[doc_id]
    if doc.orchid_id in sampled_doc_ids_w_order_id and (
        set(range(len(doc.cased_words)))
        == sampled_data.instances[
            sampled_doc_ids_w_order_id[doc.orchid_id]
        ].simulation_token_annotations.tokens
    ):
        new_docs_of_interest = docs_of_interest - 1
        return _choose_the_doc_for_token_sampling(
            [d for i, d in enumerate(docs) if i != doc_id],
            sampled_data,
            sampled_doc_ids_w_order_id,
            new_docs_of_interest if new_docs_of_interest >= 1 else 1,
        )
    return doc


# TODO This must be fully refactored because 95% of code was taken from "token_sampling"
def mentions_sampling(
    docs: list[Doc],
    token_batch: int,
    docs_of_interest: int,
    mention_indices: list[list[int]],
) -> SampledData:
    sampled_tokens_counter = 0
    exhausted_doc_mentions: set[str] = {
        doc.orchid_id
        for mention_id, doc in zip(mention_indices, docs)
        if not mention_id and doc.orchid_id
    }
    counter = 0  # variable to control infinite looping
    sampled_data = SampledData([], [])
    doc_w_their_position = {doc.orchid_id: i for i, doc in enumerate(docs)}
    while sampled_tokens_counter < token_batch:
        # sample the doc and solve the use-cases
        sampled_doc_ids_w_order_id = {
            d.orchid_id: i for i, d in enumerate(sampled_data.instances) if d.orchid_id
        }
        docs_w_mentions_to_sample = [
            doc for doc in docs if doc.orchid_id not in exhausted_doc_mentions
        ]
        doc = deepcopy(
            _choose_the_doc_for_token_sampling(
                docs_w_mentions_to_sample if docs_w_mentions_to_sample else docs,
                sampled_data,
                sampled_doc_ids_w_order_id,
                docs_of_interest,
            )
        )
        # if no tokens to sample return what we already have
        if doc is None:
            sampled_data.indices = [
                doc_w_their_position[doc.orchid_id] for doc in sampled_data.instances
            ]
            return sampled_data

        mention_tokens = mention_indices[doc_w_their_position[doc.orchid_id]]

        # choose new tokens given that we already sampled from the doc
        if doc.orchid_id in sampled_doc_ids_w_order_id:
            sampled_tokens = sampled_data.instances[
                sampled_doc_ids_w_order_id[doc.orchid_id]
            ].simulation_token_annotations.tokens
            ######## MENTION CHANGE ########
            if docs_w_mentions_to_sample:
                tokens = [
                    token_id
                    for token_id in mention_tokens
                    if token_id not in sampled_tokens
                ]
                if not tokens:
                    exhausted_doc_mentions.add(doc.orchid_id)
                else:
                    token = choice(tokens)
            ###### MENTION CHANGE END ######
            else:
                token = choice(
                    [
                        token_id
                        for token_id in range(len(doc.cased_words))
                        if token_id not in sampled_tokens
                    ]
                )
        else:
            ######## MENTION CHANGE ########
            if docs_w_mentions_to_sample:
                token = choice(mention_tokens)
            ###### MENTION CHANGE END ######
            else:
                token = choice(list(range(len(doc.cased_words))))

        token_in_cluster = _get_coref_if_token_in_cluster(token, doc.span_clusters)
        if token_in_cluster is None:
            print()
            print(token)
            token_in_cluster = {token}

        doc_tokens_number = len(doc.simulation_token_annotations.tokens)
        doc.simulation_token_annotations.tokens = (
            doc.simulation_token_annotations.tokens.union(token_in_cluster)
        )
        sampled_tokens_counter += (
            len(doc.simulation_token_annotations.tokens) - doc_tokens_number
        )

        docs[doc_w_their_position[doc.orchid_id]] = deepcopy(doc)
        if doc.orchid_id in sampled_doc_ids_w_order_id:
            sampled_data.instances[sampled_doc_ids_w_order_id[doc.orchid_id]] = doc
        else:
            sampled_data.instances.append(doc)

        if counter == 1_000_000:
            raise ValueError("Smth went wrong... The loop got infinite")
        counter += 1

    sampled_data.indices = [
        doc_w_their_position[doc.orchid_id] for doc in sampled_data.instances
    ]
    return sampled_data
