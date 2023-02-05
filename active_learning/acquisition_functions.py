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
    docs: list[Doc], token_batch: int, exhaust_doc: bool = True
) -> SampledData:
    """
    The method samples random tokens from docs in the following way:

    Repeat until # sampled tokens equals to the token_batch
        - Sample a doc (
            if exhaust_doc == True, then tokens will be
            sampled till the doc is fully exhausted of tokens
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
            d.orchid_id: i
            for i, d in enumerate(sampled_data.instances)
            if d.orchid_id
        }
        doc = deepcopy(
            _choose_the_doc_for_token_sampling(
                docs, sampled_data, sampled_doc_ids_w_order_id, exhaust_doc
            )
        )
        # if no tokens to sample return what we already have
        if doc is None:
            sampled_data.indices = [
                doc_w_their_position[doc.orchid_id]
                for doc in sampled_data.instances
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

        token_in_cluster = _get_coref_if_token_in_cluster(
            token, doc.span_clusters
        )
        doc_tokens_number = len(doc.simulation_token_annotations.tokens)
        if token_in_cluster is None:
            doc.simulation_token_annotations.tokens.add(token)
        else:
            doc.simulation_token_annotations.tokens = (
                doc.simulation_token_annotations.tokens.union(token_in_cluster)
            )
        sampled_tokens_counter += (
            len(doc.simulation_token_annotations.tokens) - doc_tokens_number
        )

        docs[doc_w_their_position[doc.orchid_id]] = deepcopy(doc)
        if doc.orchid_id in sampled_doc_ids_w_order_id:
            sampled_data.instances[
                sampled_doc_ids_w_order_id[doc.orchid_id]
            ] = doc
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
    exhaust_doc: bool,
) -> Optional[Doc]:
    """
    docs: list of Docs to sample from
    sampled_data: the docs that the tokens were already sampled from
    sampled_doc_ids_w_order_id: mapping of sampled doc ids to their
    order in the SampleData instances list
    exhaust_doc: strategy to always sample the first doc

    Returns doc and its id in the docs list

    Edge cases:
        - If all tokens were sampled from the doc, returns None
    """
    if not docs:
        return None
    doc_id = choice(range(len(docs))) if not exhaust_doc else 0
    doc = docs[doc_id]
    if doc.orchid_id in sampled_doc_ids_w_order_id and (
        set(range(len(doc.cased_words)))
        == sampled_data.instances[
            sampled_doc_ids_w_order_id[doc.orchid_id]
        ].simulation_token_annotations.tokens
    ):
        return _choose_the_doc_for_token_sampling(
            [d for i, d in enumerate(docs) if i != doc_id],
            sampled_data,
            sampled_doc_ids_w_order_id,
            exhaust_doc,
        )
    return doc
