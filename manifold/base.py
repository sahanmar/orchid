"""
Definition of the base module for Manifold Learning
"""

import abc
from dataclasses import dataclass, fields
from typing import Optional, Dict

import torch

from config.config import ManifoldLearningParams, Config
from .losses import get_loss_by_name


@dataclass
class ManifoldLearningForwardOutput:
    embeddings: torch.Tensor
    loss: Optional[torch.Tensor] = None

    def as_dict(self) -> Dict[str, torch.Tensor]:
        assert (
            self.loss is not None
        ), f"Only non-None loss is allowed to be converted"

        # Done manually to avoid deep copies of tensors
        result = {}
        for field in fields(self):
            result[field.name] = getattr(self, field.name)
        return result


class ManifoldLearningModule(torch.nn.Module, metaclass=abc.ABCMeta):
    """The base manifold learning module that is to be sub-classed"""

    def __init__(self, args: ManifoldLearningParams):
        super(ManifoldLearningModule, self).__init__()
        self._args = args

        self.loss_alpha = torch.tensor(self._args.loss_alpha, dtype=torch.float)
        self.loss = get_loss_by_name(name=self._args.loss_name)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"loss_alpha={self.loss_alpha.detach().item()},"
            f"loss={self.loss.name},"
            f")"
        )

    @classmethod
    def from_config(cls, config: Config) -> "ManifoldLearningModule":
        """Initialization from the Config object"""
        return cls(config.manifold)

    @property
    def args(self) -> ManifoldLearningParams:
        return self._args

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor) -> ManifoldLearningForwardOutput:
        """Placeholder for the forward method to be overridden"""
        pass
