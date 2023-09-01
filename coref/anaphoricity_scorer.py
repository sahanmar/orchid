""" Describes AnaphicityScorer, a torch module that for a matrix of
mentions produces their anaphoricity scores.
"""
import torch

from coref import utils
from config import Config
from typing import Optional, Callable, cast, Tuple


class AnaphoricityScorer(torch.nn.Module):
    """Calculates anaphoricity scores by passing the inputs into a FFNN"""

    def __init__(self, in_features: int, config: Config):
        super().__init__()
        hidden_size = config.model_params.hidden_size
        if not config.model_params.n_hidden_layers:
            hidden_size = in_features
        layers = []
        for i in range(config.model_params.n_hidden_layers):
            layers.extend(
                [
                    torch.nn.Linear(hidden_size if i else in_features, hidden_size),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(config.training_params.dropout_rate),
                ]
            )
        self.hidden = torch.nn.Sequential(*layers)
        self.out = torch.nn.Linear(hidden_size, out_features=1)

    def run(
        self,
        all_mentions: torch.Tensor,
        mentions_batch: torch.Tensor,
        pw_batch: torch.Tensor,
        top_indices_batch: torch.Tensor,
        top_rough_scores_batch: torch.Tensor,
        scoring_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Builds a pairwise matrix, scores the pairs and returns the scores.

        Args:
            all_mentions (torch.Tensor): [n_mentions, mention_emb]
            mentions_batch (torch.Tensor): [batch_size, mention_emb]
            pw_batch (torch.Tensor): [batch_size, n_ants, pw_emb]
            top_indices_batch (torch.Tensor): [batch_size, n_ants]
            top_rough_scores_batch (torch.Tensor): [batch_size, n_ants]

        Returns:
            torch.Tensor [batch_size, n_ants + 1]
                anaphoricity scores for the pairs + a dummy column
        """
        # [batch_size, n_ants, pair_emb]
        pair_matrix = self._get_pair_matrix(
            all_mentions, mentions_batch, pw_batch, top_indices_batch
        )

        # [batch_size, n_ants]
        scores = top_rough_scores_batch + self._ffnn(pair_matrix)
        scores = utils.add_dummy(scores, eps=True)

        if scoring_fn:
            # [n_mentions, 1]
            return scoring_fn(torch.softmax(scores, dim=1))

        return scores

    def forward(
        self,
        all_mentions: torch.Tensor,
        mentions_batch: torch.Tensor,
        pw_batch: torch.Tensor,
        top_indices_batch: torch.Tensor,
        top_rough_scores_batch: torch.Tensor,
        scoring_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self.run(
            all_mentions,
            mentions_batch,
            pw_batch,
            top_indices_batch,
            top_rough_scores_batch,
            scoring_fn,
        )

    def _ffnn(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates anaphoricity scores.

        Args:
            x: tensor of shape [batch_size, n_ants, n_features]

        Returns:
            tensor of shape [batch_size, n_ants]
        """
        x = self.out(self.hidden(x))
        return x.squeeze(2)

    @staticmethod
    def _get_pair_matrix(
        all_mentions: torch.Tensor,
        mentions_batch: torch.Tensor,
        pw_batch: torch.Tensor,
        top_indices_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Builds the matrix used as input for AnaphoricityScorer.

        Args:
            all_mentions (torch.Tensor): [n_mentions, mention_emb],
                all the valid mentions of the document,
                can be on a different device
            mentions_batch (torch.Tensor): [batch_size, mention_emb],
                the mentions of the current batch,
                is expected to be on the current device
            pw_batch (torch.Tensor): [batch_size, n_ants, pw_emb],
                pairwise features of the current batch,
                is expected to be on the current device
            top_indices_batch (torch.Tensor): [batch_size, n_ants],
                indices of antecedents of each mention

        Returns:
            torch.Tensor: [batch_size, n_ants, pair_emb]
        """
        emb_size = mentions_batch.shape[1]
        n_ants = pw_batch.shape[1]

        a_mentions = mentions_batch.unsqueeze(1).expand(-1, n_ants, emb_size)
        b_mentions = all_mentions[top_indices_batch]
        similarity = a_mentions * b_mentions

        out = torch.cat((a_mentions, b_mentions, similarity, pw_batch), dim=2)
        return out


class MCDropoutAnaphoricityScorer(AnaphoricityScorer):
    """
    MC Dropout Anaphoricity Scorer

    The model assumes dropout to be already on. For example
    activated through coreference model which uses that submodule
    """

    def __init__(self, in_features: int, config: Config):
        self.parameters_samples = config.active_learning.parameters_samples
        super().__init__(in_features, config)

    def forward(
        self,
        all_mentions: torch.Tensor,
        mentions_batch: torch.Tensor,
        pw_batch: torch.Tensor,
        top_indices_batch: torch.Tensor,
        top_rough_scores_batch: torch.Tensor,
        scoring_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        return torch.mean(
            torch.stack(
                [
                    self.run(
                        all_mentions,
                        mentions_batch,
                        pw_batch,
                        top_indices_batch,
                        top_rough_scores_batch,
                        scoring_fn,
                    )
                    for _ in range(self.parameters_samples)
                ]
            ),
            dim=0,
        )
