""" Contains type aliases for coref module """

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch

EPSILON = 1e-7
LARGE_VALUE = 1000  # used instead of inf due to bug #16762 in pytorch

Doc = Dict[str, Any]
Span = Tuple[int, int]


@dataclass
class SampledData:
    indices: List[int]
    instances: List[Doc]


@dataclass
class CorefResult:
    coref_scores: torch.Tensor = None  # [n_words, k + 1]
    coref_y: torch.Tensor = None  # [n_words, k + 1]

    word_clusters: Optional[List[List[int]]] = None
    span_clusters: Optional[List[List[Span]]] = None

    span_scores: Optional[torch.Tensor] = None  # [n_heads, n_words, 2]
    span_y: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # [n_heads] x2
