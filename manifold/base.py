"""
Definition of the base module for Manifold Learning
"""

import abc

import torch

from config.config import ManifoldLearningParams, Config
from .losses import get_loss_by_name


class ManifoldLearningModule(torch.nn.Module, metaclass=abc.ABCMeta):
    """The base manifold learning module that is to be sub-classed"""

    def __init__(self, args: ManifoldLearningParams):
        super(ManifoldLearningModule, self).__init__()
        self._args = args

        self.loss_alpha = torch.tensor(self._args.loss_alpha, dtype=torch.long)
        self.loss = get_loss_by_name(name=self._args.loss_name)

    @classmethod
    def from_config(cls, config: Config) -> "ManifoldLearningModule":
        """Initialization from the Config object"""
        return cls(config.manifold)

    @property
    def args(self) -> ManifoldLearningParams:
        return self._args

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Placeholder for the forward method to be overridden"""
        pass
