"""
Describes RoughScorer, a simple bilinear module to calculate rough
anaphoricity scores.
"""

from typing import Tuple, Callable, Optional, cast

import torch


class RoughScorer(torch.nn.Module):
    """
    Is needed to give a roughly estimate of the anaphoricity of two candidates,
    only top scoring candidates are considered on later steps to reduce
    computational complexity.
    """

    def __init__(self, features: int, rough_k: int, dropout_rate: float):
        super().__init__()
        features = features
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.bilinear = torch.nn.Linear(features, features)

        self.k = rough_k

    def forward(
        self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
        mentions: torch.Tensor,
        scoring_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        If scoring function is None,
            Returns rough anaphoricity scores for candidates, which consist of
            the bilinear output of the current model summed with mention scores
            with sizes [n_mentions, top_k_mentions] (both tensors).

        Else,
            Return a score of every mention with size [n_mentions, 1]
        """
        # [n_mentions, n_mentions]
        pair_mask = self._get_pair_mask(mentions)

        bilinear_scores = self.dropout(self.bilinear(mentions)).mm(mentions.T)

        if scoring_fn:
            # [n_mentions, 1]
            scores = scoring_fn(torch.softmax(bilinear_scores, dim=1))
            return cast(Tuple[torch.Tensor, torch.Tensor], torch.sort(scores))

        # [n_mentions, top_k_mentions]
        rough_scores = pair_mask + bilinear_scores

        return self._prune(rough_scores)

    @staticmethod
    def _get_pair_mask(mentions: torch.Tensor) -> torch.Tensor:
        # [n_mentions, n_mentions]
        pair_mask = torch.arange(mentions.shape[0])
        pair_mask = pair_mask.unsqueeze(1) - pair_mask.unsqueeze(0)
        pair_mask = torch.log((pair_mask > 0).to(torch.float))
        pair_mask = pair_mask.to(mentions.device)
        return pair_mask

    def _prune(self, rough_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects top-k rough antecedent scores for each mention.

        Args:
            rough_scores: tensor of shape [n_mentions, n_mentions], containing
                rough antecedent scores of each mention-antecedent pair.

        Returns:
            FloatTensor of shape [n_mentions, k], top rough scores
            LongTensor of shape [n_mentions, k], top indices
        """
        top_scores, indices = torch.topk(
            rough_scores, k=min(self.k, len(rough_scores)), dim=1, sorted=False
        )
        return top_scores, indices


class MCDropoutRoughScorer(RoughScorer):
    # TODO add documentation
    def __init__(
        self,
        features: int,
        rough_k: int,
        dropout_rate: float,
        parameters_samples: int,
    ):
        self.parameters_samples = parameters_samples
        super().__init__(features, rough_k, dropout_rate)

    def forward(
        self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
        mentions: torch.Tensor,
        scoring_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns rough anaphoricity scores for candidates, which consist of
        the bilinear output of the current model summed with mention scores.
        """
        pair_mask = self._get_pair_mask(mentions)

        if scoring_fn:
            # Turn on dropout only when sampling is happening

            # Average over empirical distribution samples
            # TODO return all 10 samples with words
            bilinear_scores = torch.mean(
                torch.stack(
                    [
                        self.dropout(self.bilinear(mentions)).mm(mentions.T)
                        for _ in range(self.parameters_samples)
                    ]
                ),
                dim=0,
            )

            # [n_mentions, 1]
            scores = scoring_fn(torch.softmax(bilinear_scores, dim=1))
            return cast(Tuple[torch.Tensor, torch.Tensor], torch.sort(scores))
        else:
            bilinear_scores = self.bilinear(mentions).mm(mentions.T)

        rough_scores = pair_mask + bilinear_scores

        return self._prune(rough_scores)
