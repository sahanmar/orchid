"""
Definition of the base module for Manifold Learning
"""

import abc

import torch

from config.config import ManifoldLearningParams, Config
from .losses import get_loss_by_name


class ManifoldLearningModule(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, args: ManifoldLearningParams):
        super(ManifoldLearningModule, self).__init__()
        self._args = args

        self.loss = get_loss_by_name(name=self._args.loss_name)

    @classmethod
    def from_config(cls, config: Config) -> "ManifoldLearningModule":
        return cls(config.manifold)

    @property
    def args(self) -> ManifoldLearningParams:
        return self._args

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass
