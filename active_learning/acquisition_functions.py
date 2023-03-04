from typing import Tuple, cast
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
    _: dict[str, list[int]] = {},
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

    (
        sampled_tokens_counter,
        counter,
        sampled_data,
        docs_w_their_positions,
    ) = _setup(docs)
    while sampled_tokens_counter < token_batch:
        # sample the doc and solve the use-cases
        sampled_doc_ids_w_order_id = {
            d.orchid_id: i
            for i, d in enumerate(sampled_data.instances)
            if d.orchid_id
        }
        doc = deepcopy(
            _choose_the_doc_for_token_sampling(
                docs, sampled_data, sampled_doc_ids_w_order_id, docs_of_interest
            )
        )
        # if no tokens to sample return what we already have
        if doc is None:
            return _fill_sampled_data_indices(
                sampled_data, docs_w_their_positions
            )

        if not doc.orchid_id:
            raise ValueError("No doc id. This is bad...")

        # choose new tokens given that we already sampled from the doc
        previously_sampled_tokens = doc.simulation_token_annotations.tokens
        if doc.orchid_id in sampled_doc_ids_w_order_id:
            currently_sampled_tokens = sampled_data.instances[
                sampled_doc_ids_w_order_id[doc.orchid_id]
            ].simulation_token_annotations.tokens
            tokens = _filtered_tokens_to_sample(
                list(range(len(doc.cased_words))),
                currently_sampled_tokens.union(previously_sampled_tokens),
            )
        else:
            tokens = _filtered_tokens_to_sample(
                list(range(len(doc.cased_words))), previously_sampled_tokens
            )

        if not tokens:
            continue

        token = choice(tokens)

        # Handle sampled data extension
        doc_tokens_number = len(doc.simulation_token_annotations.tokens)
        _handle_sampled_token_into_sampled_data_mutable(
            token,
            doc,
            sampled_data,
            sampled_doc_ids_w_order_id.get(doc.orchid_id),
        )
        sampled_tokens_counter += (
            len(doc.simulation_token_annotations.tokens) - doc_tokens_number
        )
        docs[docs_w_their_positions[doc.orchid_id]] = deepcopy(doc)

        counter = _counter_update(counter)
    return _fill_sampled_data_indices(sampled_data, docs_w_their_positions)


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


def mentions_sampling(
    docs: list[Doc],
    token_batch: int,
    docs_of_interest: int,
    mentions: dict[str, list[int]],
) -> SampledData:

    """
    The method samples random tokens from mentions predicted by a rough scorer:

    Repeat until # sampled tokens equals to the token_batch
        - Sample a doc (
            if docs_of_interest are set, then tokens will be
            preferred from first docs_of_interest docs
        )

        - Sample a token iteratively given mentions (until all mentions are exhausted)

        if a token belongs to a coreference span and has other coref mentions
            - take the full span and find the closest mention
        else
            - take the token as it is

        if the doc was already sampled before
            - extend the simulation tokens field and save a doc
        else
            - save a doc and write in simulation tokens

    return all created docs with sampled tokens
    """
    orchid_id_nullability_check(docs)
    exhausted_doc_mentions: set[str] = {
        doc.orchid_id
        for doc in docs
        if doc.orchid_id
        and set(mentions[doc.orchid_id])
        - doc.simulation_token_annotations.tokens
    }
    all_docs_w_mentions = len(exhausted_doc_mentions)
    (
        sampled_tokens_counter,
        counter,
        sampled_data,
        docs_w_their_positions,
    ) = _setup(docs)
    while sampled_tokens_counter < token_batch:
        # sample the doc and solve the edge-cases
        sampled_doc_ids_w_order_id = {
            d.orchid_id: i
            for i, d in enumerate(sampled_data.instances)
            if d.orchid_id
        }
        docs_w_mentions_to_sample = [
            doc for doc in docs if doc.orchid_id not in exhausted_doc_mentions
        ]
        doc = deepcopy(
            _choose_the_doc_for_token_sampling(
                docs_w_mentions_to_sample
                if docs_w_mentions_to_sample
                else docs,
                sampled_data,
                sampled_doc_ids_w_order_id,
                docs_of_interest,
            )
        )
        # if no tokens to sample return what we already have
        if doc is None:
            return _fill_sampled_data_indices(
                sampled_data, docs_w_their_positions
            )

        if not doc.orchid_id:
            raise ValueError("No doc id. This is bad...")
        tokens_to_sample = (
            list(range(len(doc.cased_words)))
            if doc.orchid_id in exhausted_doc_mentions
            else mentions[doc.orchid_id]
        )

        # choose new tokens given that we already sampled from the doc
        if doc.orchid_id in sampled_doc_ids_w_order_id:
            sampled_tokens = sampled_data.instances[
                sampled_doc_ids_w_order_id[doc.orchid_id]
            ].simulation_token_annotations.tokens
            # sample from mentions
            if docs_w_mentions_to_sample:
                tokens = _filtered_tokens_to_sample(
                    tokens_to_sample, sampled_tokens
                )
                if not tokens:
                    exhausted_doc_mentions.add(doc.orchid_id)
                    continue
            # if no mention docs left sample ordinary token
            elif len(exhausted_doc_mentions) == all_docs_w_mentions:
                tokens = _filtered_tokens_to_sample(
                    list(range(len(doc.cased_words))), sampled_tokens
                )
            else:
                continue
        else:
            # sample from mentions or if no mention docs left sample ordinary token
            if (
                docs_w_mentions_to_sample
                or len(exhausted_doc_mentions) == all_docs_w_mentions
            ):
                tokens = _filtered_tokens_to_sample(
                    tokens_to_sample, doc.simulation_token_annotations.tokens
                )
                if not tokens:
                    exhausted_doc_mentions.add(doc.orchid_id)
                    continue
            else:
                continue

        if not tokens:
            raise ValueError(
                "No tokens available... This should never happened. Do your refactor bruv"
            )

        token = choice(tokens)

        # Handle sampled data extension
        doc_tokens_number = len(doc.simulation_token_annotations.tokens)
        _handle_sampled_token_into_sampled_data_mutable(
            token,
            doc,
            sampled_data,
            sampled_doc_ids_w_order_id.get(doc.orchid_id),
        )
        sampled_tokens_counter += (
            len(doc.simulation_token_annotations.tokens) - doc_tokens_number
        )
        docs[docs_w_their_positions[doc.orchid_id]] = deepcopy(doc)

        counter = _counter_update(counter)

    return _fill_sampled_data_indices(sampled_data, docs_w_their_positions)


def _handle_sampled_token_into_sampled_data_mutable(
    token: int,
    doc: Doc,
    sampled_data: SampledData,
    doc_idx_in_sampled_data: Optional[int],
) -> None:
    """
    The function mutates doc and sampled data

    if a token belongs to a coreference span and has other mentions
        - take the full span and find the closest mention
    else
        - take the token as it is

    if the doc was already sampled before
        - extend the simulation tokens field and save a doc
    else
        - save a doc and write in simulation tokens
    """
    token_in_cluster = _get_coref_if_token_in_cluster(token, doc.span_clusters)
    if token_in_cluster is None:
        token_in_cluster = {token}

    doc.simulation_token_annotations.tokens = (
        doc.simulation_token_annotations.tokens.union(token_in_cluster)
    )

    if doc_idx_in_sampled_data is not None:
        sampled_data.instances[doc_idx_in_sampled_data] = doc
    else:
        sampled_data.instances.append(doc)


def orchid_id_nullability_check(docs: list[Doc]) -> None:
    for doc in docs:
        if doc.orchid_id is None:
            raise ValueError(
                "Its kinda has optional typing but it can't be None"
            )


def _fill_sampled_data_indices(
    sampled_data: SampledData, doc_w_their_position: dict[str, int]
) -> SampledData:
    orchid_id_nullability_check(sampled_data.instances)
    sampled_data.indices = [
        doc_w_their_position[doc.orchid_id] for doc in sampled_data.instances  # type: ignore
    ]
    return sampled_data


def _setup(docs: list[Doc]) -> Tuple[int, int, SampledData, dict[str, int]]:

    # initial setup for acquisition functions

    orchid_id_nullability_check(docs)
    sampled_tokens_counter = 0
    counter = 0  # variable to control infinite looping
    sampled_data = SampledData([], [])
    doc_w_their_position = {doc.orchid_id: i for i, doc in enumerate(docs)}
    return sampled_tokens_counter, counter, sampled_data, doc_w_their_position  # type: ignore


def _filtered_tokens_to_sample(
    token_ids: list[int], sampled_tokens: set[int]
) -> list[int]:
    """
    returns a list of tokens filtered given already sampled tokens
    """
    return [
        token_id for token_id in token_ids if token_id not in sampled_tokens
    ]


def _counter_update(
    num_of_iters: int, max_num_of_iters: int = 1_000_000, step: int = 1
) -> int:
    if num_of_iters == max_num_of_iters:
        raise ValueError("Smth went wrong... The loop got infinite")
    return num_of_iters + step
