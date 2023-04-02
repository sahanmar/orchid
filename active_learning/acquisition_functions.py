from typing import Tuple, Optional, Any
from random import sample, shuffle

from coref.const import Doc, SampledData
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


MentionType = Tuple[int, float]


def mentions_sampling(
    docs: list[Doc],
    token_batch: int,
    docs_of_interest: int,
    mentions: dict[str, Any],  # list[MentionType]
) -> SampledData:
    """
    The method samples tokens from mentions given scores:

    Repeat until # sampled tokens equals to the token_batch
        - Get only mentions that are not sampled

        - Sort all mentions given scores, doc_id and then mention id

        - Choose priority docs given the score

        - Take a first mention in a priority list

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

    # Get only mentions that are not sampled
    filtered_mentions: dict[str, Any] = {
        doc.orchid_id: [  # type: ignore
            (ment, score)
            for ment, score in mentions[doc.orchid_id]  # type: ignore
            if ment not in doc.simulation_token_annotations.tokens
        ]
        for doc in docs
    }
    docs_dict = {doc.orchid_id: doc for doc in docs}

    (
        sampled_tokens_counter,
        counter,
        sampled_data,
        docs_w_their_positions,
    ) = _setup(docs)

    # Get a monstrous sorted array
    ment_score_doc_id_unsorted: list[Tuple[float, int, str]] = [
        (score, ment, doc_id)
        for doc_id, ments_w_scores in filtered_mentions.items()
        for ment, score in ments_w_scores
    ]
    shuffle(ment_score_doc_id_unsorted)
    ment_score_doc_id_sorted = sorted(
        ment_score_doc_id_unsorted,
        key=lambda x: x[0],  # lambda x: (x[0], x[2]) if doc exhaust
        reverse=True,
    )
    del ment_score_doc_id_unsorted

    # Choose priority docs given the score
    priority_docs = set()
    for _, _, doc_id in ment_score_doc_id_sorted:
        if doc_id not in priority_docs:
            priority_docs.add(doc_id)
        if len(priority_docs) == docs_of_interest:
            break

    if docs_of_interest != 0:
        ment_score_doc_id_prio = []
        ment_score_doc_id_deprio = []
        for score, token, doc_id in ment_score_doc_id_sorted:
            if doc_id in priority_docs:
                ment_score_doc_id_prio.append((score, token, doc_id))
            else:
                ment_score_doc_id_deprio.append((score, token, doc_id))

        ment_score_doc_id = ment_score_doc_id_prio + ment_score_doc_id_deprio
    elif docs_of_interest < 0:
        raise ValueError(
            f"Wrong input for docs of interest = {docs_of_interest}..."
        )

    # Sample the data
    while sampled_tokens_counter < token_batch:
        sampled_doc_ids_w_order_id = {
            d.orchid_id: i
            for i, d in enumerate(sampled_data.instances)
            if d.orchid_id
        }
        if not ment_score_doc_id:
            return _fill_sampled_data_indices(
                sampled_data, docs_w_their_positions
            )
        score, token, doc_id = ment_score_doc_id.pop(0)
        doc = deepcopy(docs_dict[doc_id])

        # Handle sampled data extension
        doc_tokens_number = len(doc.simulation_token_annotations.tokens)
        _handle_sampled_token_into_sampled_data_mutable(
            token,
            doc,
            sampled_data,
            sampled_doc_ids_w_order_id.get(doc.orchid_id),  # type: ignore
        )
        sampled_tokens_counter += (
            len(doc.simulation_token_annotations.tokens) - doc_tokens_number
        )
        docs_dict[doc.orchid_id] = deepcopy(doc)  # type: ignore

        counter = _counter_update(counter)

    return _fill_sampled_data_indices(sampled_data, docs_w_their_positions)


def orchid_id_nullability_check(docs: list[Doc]) -> None:
    for doc in docs:
        if doc.orchid_id is None:
            raise ValueError(
                "Its kinda has optional typing but it can't be None"
            )


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


# def _filtered_tokens_to_sample(
#     token_ids: list[int], sampled_tokens: set[int]
# ) -> list[int]:
#     """
#     returns a list of tokens filtered given already sampled tokens
#     """
#     return [token_id for token_id in token_ids if token_id not in sampled_tokens]


def _counter_update(
    num_of_iters: int, max_num_of_iters: int = 1_000_000, step: int = 1
) -> int:
    if num_of_iters == max_num_of_iters:
        raise ValueError("Smth went wrong... The loop got infinite")
    return num_of_iters + step
