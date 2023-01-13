""" Contains type aliases for coref module """

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch

EPSILON = 1e-7
LARGE_VALUE = 1000  # used instead of inf due to bug #16762 in pytorch

Span = Tuple[int, int]


@dataclass
class Doc:
    document_id: str
    cased_words: list[str]
    sent_id: list[int]
    part_id: int
    speaker: list[str]
    pos: list[str]
    deprel: list[str]
    head: Optional[list[str]]
    head2span: list[list[int]]
    word_clusters: list[list[int]]
    span_clusters: list[list[Tuple[int, int]]]
    word2subword: list[Tuple[int, int]]
    subwords: list[str]
    word_id: list[int]


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
