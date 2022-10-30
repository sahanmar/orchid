""" Describes RoughScorer, a simple bilinear module to calculate rough
anaphoricity scores.
"""

from typing import Tuple

import torch

from config import Config


class RoughScorer(torch.nn.Module):
    """
    Is needed to give a roughly estimate of the anaphoricity of two candidates,
    only top scoring candidates are considered on later steps to reduce
    computational complexity.
    """

    def __init__(self, features: int, config: Config):
        super().__init__()
        self.dropout = torch.nn.Dropout(config.training_params.dropout_rate)
        self.bilinear = torch.nn.Linear(features, features)

        self.k = config.model_params.rough_k

    def forward(
        self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
        mentions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns rough anaphoricity scores for candidates, which consist of
        the bilinear output of the current model summed with mention scores.
        """
        # [n_mentions, n_mentions]
        pair_mask = self._get_pair_mask(mentions)

        bilinear_scores = self.dropout(self.bilinear(mentions)).mm(mentions.T)

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
    def __init__(self, features: int, config: Config):
        self.parameters_samples = config.active_learning.parameters_samples
        super().__init__(features, config)

    def forward(
        self,  # type: ignore  # pylint: disable=arguments-differ  #35566 in pytorch
        mentions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns rough anaphoricity scores for candidates, which consist of
        the bilinear output of the current model summed with mention scores.
        """
        pair_mask = self._get_pair_mask(mentions)

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

        rough_scores = pair_mask + bilinear_scores

        return self._prune(rough_scores)
