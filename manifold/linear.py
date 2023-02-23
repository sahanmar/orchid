"""
Linear dimensionality reduction
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from config.config import ManifoldLearningParams
from .base import ManifoldLearningModule, ManifoldLearningForwardOutput
from .dataset import ManifoldDataloader


class BasePCA(ManifoldLearningModule):
    def __init__(self, args: ManifoldLearningParams):
        super(BasePCA, self).__init__(args=args)

        self.pca_layer = nn.Linear(
            in_features=self.args.standalone.input_dimensionality,
            out_features=self.args.standalone.output_dimensionality,
        )
        self.optimizer = optim.Adam(
            params=self.parameters(), lr=self.args.standalone.learning_rate
        )

    def forward(self, inputs: torch.Tensor) -> ManifoldLearningForwardOutput:
        embeddings = self.pca_layer(inputs)  # [n, target_dim]
        if self.args.loss_in_forward:
            loss = self.evaluate_criterion(
                inputs=inputs,
                embeddings=embeddings,
            )
        else:
            loss = None

        output = ManifoldLearningForwardOutput(embeddings=embeddings, loss=loss)
        return output

    def evaluate_criterion(
        self,
        inputs: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        inputs_reconstructed = self.loss.reconstruct_from_linear(
            embeddings=embeddings,
            linear_layer=self.pca_layer,
        )  # [n, input_dim]
        loss_ = self.loss(inputs=inputs, outputs=inputs_reconstructed)

        # Multiply the output loss with a scaling factor
        loss_ *= self.loss_alpha
        return loss_

    def train_step(self, inputs: torch.Tensor) -> ManifoldLearningForwardOutput:
        self.optimizer.zero_grad()
        assert self.args.loss_in_forward, (
            f"The loss has to be computed in the forward step "
            f"to preserve memory"
        )
        with torch.enable_grad():
            result: ManifoldLearningForwardOutput = self(inputs)
            assert (
                result.loss is not None
            ), f"Step loss is missing in the results"
            result.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return result

    def train_on_inputs(self, inputs: torch.Tensor) -> None:
        pbar = tqdm(range(self.args.standalone.epochs))
        dataloader = ManifoldDataloader(
            inputs=inputs,
            batch_size=self.args.standalone.batch_size,
            shuffle=self.args.standalone.shuffle,
        )
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            for b in dataloader:
                res = self.train_step(inputs=b)
                pf = {
                    k_: v_.detach().numpy()
                    for k_, v_ in res.as_dict().items()
                    if k_ in self.args.verbose_outputs
                }
                pbar.set_postfix(pf)
