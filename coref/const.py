""" Contains type aliases for coref module """

from dataclasses import dataclass
from typing import Tuple, Optional, Iterable
import hashlib
import torch
import numpy as np
from itertools import groupby

EPSILON = 1e-7
LARGE_VALUE = 1000  # used instead of inf due to bug #16762 in pytorch

Span = Tuple[int, int]


@dataclass
class SimulationSpanAnnotations:
    tokens: set[int]
    old_2_new_ids_map: dict[int, int]
    original_subtokens_ids: list[int]


@dataclass
class Doc:
    document_id: str
    cased_words: list[str]
    sent_id: list[int]
    part_id: int
    speaker: list[str]
    pos: list[str]
    deprel: list[str]
    head: list[Optional[str]]
    head2span: list[list[int]]
    word_clusters: list[list[int]]
    span_clusters: list[list[Tuple[int, int]]]
    word2subword: list[Tuple[int, int]]
    subwords: list[str]
    word_id: list[int]
    orchid_id: Optional[str] = None
    simulation_token_annotations: SimulationSpanAnnotations = (
        SimulationSpanAnnotations(set(), {}, [])
    )

    def __post_init__(self) -> None:
        if not self.orchid_id:
            self.orchid_id = hashlib.sha1(
                "".join(self.cased_words).encode("utf-8")
            ).hexdigest()

    def create_simulation_pseudodoc(self) -> "Doc":

        if not self.simulation_token_annotations.tokens:
            word_ids = list(range(len(self.cased_words)))
        else:
            word_ids = sorted(self.simulation_token_annotations.tokens)
        # create the mapping
        new_w_ids_map: dict[int, int] = {
            w_id: new_w_id
            for w_id, new_w_id in zip(word_ids, range(len(word_ids)))
        }

        # reindex head to spans
        head2span: list[list[int]] = []
        for head, start, end in self.head2span:
            if head in new_w_ids_map and start in new_w_ids_map:
                reindexed_head2span = [
                    new_w_ids_map[head],
                    new_w_ids_map[start],
                    new_w_ids_map[start] + (end - start),
                ]
                head2span.append(reindexed_head2span)

        # reindex word clusters
        word_clusters: list[list[int]] = []
        for cluster in self.word_clusters:
            reindexed_cluster = [
                new_w_ids_map[i] for i in cluster if i in new_w_ids_map
            ]
            if reindexed_cluster:
                word_clusters.append(reindexed_cluster)

        # reindex span clusters
        span_clusters: list[list[Tuple[int, int]]] = []
        for span_cluster in self.span_clusters:
            reindexed_span_cluster: list[Tuple[int, int]] = [
                (new_w_ids_map[start], new_w_ids_map[start] + end - start)
                for start, end in span_cluster
                if start in new_w_ids_map
            ]
            if reindexed_span_cluster:
                span_clusters.append(reindexed_span_cluster)

        # word to subword
        word2subword: list[Tuple[int, int]] = []
        shift = 0
        for w in word_ids:
            start, end = self.word2subword[w]
            diff = end - start
            word2subword.append((shift, shift + diff))
            shift += diff

        # subwords
        subwords = [
            word
            for i, (start, end) in enumerate(self.word2subword)
            if i in word_ids
            for word in self.subwords[start:end]
        ]

        return Doc(
            document_id=self.document_id,
            orchid_id=self.orchid_id,
            cased_words=[self.cased_words[w] for w in word_ids],
            sent_id=[self.sent_id[w] for w in word_ids],
            part_id=self.part_id,
            speaker=[self.speaker[w] for w in word_ids],
            pos=[self.pos[w] for w in word_ids],
            deprel=[self.deprel[w] for w in word_ids],
            head=[],  # unused
            head2span=head2span,
            word_clusters=word_clusters,
            span_clusters=span_clusters,
            word2subword=word2subword,
            subwords=subwords,
            word_id=list(range(len(subwords))),
            simulation_token_annotations=SimulationSpanAnnotations(
                self.simulation_token_annotations.tokens,
                new_w_ids_map,
                [
                    subtoken
                    for i, (start, end) in enumerate(self.word2subword)
                    if i in word_ids
                    for subtoken in range(start, end)
                ],
            ),
        )

    def subwords_2_words_w_payload(
        self, subwords: Iterable[int], payload: Iterable[float]
    ) -> list[Tuple[int, float]]:
        words_2_subwords_map = {
            subtoken: i
            for i, (start, end) in enumerate(self.word2subword)
            for subtoken in range(start, end)
        }
        subword_w_payload = [
            (words_2_subwords_map[s], p) for s, p in zip(subwords, payload)
        ]
        grouped_res = [
            (key, [payload for _, payload in group])
            for key, group in groupby(
                sorted(subword_w_payload, key=lambda x: x[0]),
                key=lambda x: x[0],
            )
        ]

        return [(token, np.mean(payload)) for token, payload in grouped_res]


@dataclass
class SampledData:
    indices: list[int]
    instances: list[Doc]


@dataclass
class CorefResult:
    coref_scores: torch.Tensor = None  # [n_words, k + 1]
    coref_y: torch.Tensor = None  # [n_words, k + 1]

    word_clusters: Optional[list[list[int]]] = None
    span_clusters: Optional[list[list[Span]]] = None

    span_scores: Optional[torch.Tensor] = None  # [n_heads, n_words, 2]
    span_y: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # [n_heads] x2


@dataclass
class ReducedDimensionalityCorefResult(CorefResult):
    inputs: Optional[torch.Tensor] = None  # [n_subwords, initial_dim]
    embeddings: Optional[torch.Tensor] = None  # [n_subwords, target_dim]
