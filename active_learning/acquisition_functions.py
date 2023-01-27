from typing import Tuple, Any
from random import sample, choice

from coref.const import Doc, SampledData, Optional
from itertools import chain
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


def span_sampling(docs: list[Doc], span_batch: int) -> SampledData:
    """
    The method samples random spans from docs in the following way:

    Repeat until # sampled spans equals to the span_batch
        - Sample a doc
        - Sample a span

        if a span has antecedents
            - find the closest one
        else pass

        if the doc was already sampled before
            - extend the simulation spans field
        else
            - create a pseudo doc and write in simulation spans

    return all created pseudo docs with sampled spans
    """

    sampled_spans_counter = 0
    counter = 0  # variable to control infinite looping
    sampled_data = SampledData([], [])
    docs_copy = deepcopy(docs)
    while sampled_spans_counter < span_batch:
        # sample the doc and solve the use-cases
        sampled_doc_ids_w_order_id = {
            d.document_id: i for i, d in enumerate(sampled_data.instances)
        }
        doc_w_doc_id = _choose_the_doc_for_span_sampling(
            docs_copy, sampled_data, sampled_doc_ids_w_order_id
        )
        # if no spans to sample return what we already have
        if doc_w_doc_id is None:
            return sampled_data

        doc, doc_id = doc_w_doc_id
        # choose new spans given that we already sampled from the doc
        if doc.document_id in sampled_doc_ids_w_order_id:
            sampled_spans = set(
                sampled_data.instances[
                    sampled_doc_ids_w_order_id[doc.document_id]
                ].simulation_span_annotations.spans
            )
            docs_copy[doc_id].span_clusters = [
                [span for span in cluster if span not in sampled_spans]
                for cluster in doc.span_clusters
            ]
        elif len(list(chain.from_iterable(docs_copy[doc_id].span_clusters))) > 0:
            sampled_data.indices.append(doc_id)
        else:
            return sampled_data

        cluster = choice(doc.span_clusters)
        spans: list[Tuple[int, int]] = []
        if len(cluster) > 1:
            # choose the closes coreference
            spans.extend(choice(_get_consecutive_pairs(cluster)))
        elif len(cluster) == 1:
            spans.extend(cluster)

        if cluster:
            sampled_spans_counter += 1
            doc.simulation_span_annotations.spans.extend(spans)
            sampled_data.instances.append(doc.create_simulation_pseudodoc())

        if counter == 1_000_000:
            raise ValueError("Smth went wrong... The loop got infinite")
        counter += 1
        print("*****")
        print(doc.simulation_span_annotations.spans)
        print("*")
        print(docs[doc_id].span_clusters)
        print("*****")
    return sampled_data


def _get_consecutive_pairs(list_to_pair: list[Any]) -> list[list[Any]]:
    return [
        [list_to_pair[i], list_to_pair[i + 1]] for i in range(len(list_to_pair) - 1)
    ]


def _choose_the_doc_for_span_sampling(
    docs: list[Doc],
    sampled_data: SampledData,
    sampled_doc_ids_w_order_id: dict[str, int],
) -> Optional[Tuple[Doc, int]]:
    """
    docs: list of Docs to sample from
    sampled_data: the docs that the spans were already sampled from
    sampled_doc_ids_w_order_id: mapping of sampled doc ids to their
    order in the SampleData instances list

    Returns doc and its id in the docs list

    Edge cases:
        - If all spans were sampled from the doc, returns None
    """
    if not docs:
        return None
    doc_id = choice(range(len(docs)))
    doc = docs[doc_id]
    if doc.document_id in sampled_doc_ids_w_order_id and (
        len(set(chain.from_iterable(doc.span_clusters)))
        == len(
            set(
                chain.from_iterable(
                    sampled_data.instances[
                        sampled_doc_ids_w_order_id[doc.document_id]
                    ].simulation_span_annotations.spans
                )
            )
        )
    ):
        return _choose_the_doc_for_span_sampling(
            [d for i, d in enumerate(docs) if i != doc_id],
            sampled_data,
            sampled_doc_ids_w_order_id,
        )
    return doc, doc_id


I sample from a wrong field! I have to sample for spans field and then take the cluster info if I get to a cluster span