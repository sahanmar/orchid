"""
Linear dimensionality reduction
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from config.config import ManifoldLearningParams
from .base import ManifoldLearningModule
from .dataset import ManifoldDataloader


class BasePCA(ManifoldLearningModule):
    def __init__(self, args: ManifoldLearningParams):
        super(BasePCA, self).__init__(args=args)

        self.pca_layer = nn.Linear(
            in_features=self.args.input_dimensionality,
            out_features=self.args.output_dimensionality,
        )
        self.optimizer = optim.Adam(
            params=self.parameters(), lr=self.args.learning_rate
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.pca_layer(inputs)
        return embeddings  # [n, target_dim]

    def train_step(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.optimizer.zero_grad()
        with torch.enable_grad():
            embeddings = self(inputs)
            inputs_reconstructed = self.loss.reconstruct_from_linear(
                embeddings=embeddings,
                linear_layer=self.pca_layer,
            )
            loss_step = self.loss(inputs=inputs, outputs=inputs_reconstructed)
            loss_step.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return {"embeddings": embeddings, "loss": loss_step}

    def train_on_inputs(self, inputs: torch.Tensor) -> None:
        pbar = tqdm(range(self.args.epochs))
        dataloader = ManifoldDataloader(
            inputs=inputs,
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle,
        )
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            for b in dataloader:
                res = self.train_step(inputs=b)
                pf = {
                    k_: v_.detach().numpy()
                    for k_, v_ in res.items()
                    if k_ in self.args.verbose_outputs
                }
                pbar.set_postfix(pf)
