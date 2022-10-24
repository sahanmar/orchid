import abc
from typing import Optional, Type, Dict, Any, List

import torch


class ManifoldLearningLoss(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract Structure for Manifold Learning Losses"""

    name: Optional[str] = None

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super(ManifoldLearningLoss, self).__init__()

    @staticmethod
    def reconstruct_from_linear(
        embeddings: torch.Tensor, linear_layer: torch.nn.Linear
    ) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        pass


class SquaredReconstructionLoss(ManifoldLearningLoss):
    """Squared Reconstruction Error"""

    name: str = "sq_rec_loss"

    def __init__(self) -> None:
        super(SquaredReconstructionLoss, self).__init__()
        pass

    @staticmethod
    def reconstruct_from_linear(
        embeddings: torch.Tensor, linear_layer: torch.nn.Linear
    ) -> torch.Tensor:
        return torch.matmul(
            (embeddings - linear_layer.bias), linear_layer.weight
        )  # [n, original_dim]

    def forward(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(torch.subtract(inputs, outputs) ** 2)


class ManifoldLearningLossFactory:
    def __init__(self) -> None:
        self._losses: Dict[str, Type[ManifoldLearningLoss]] = {}

    def register_loss(
        self, name: str, loss: Type[ManifoldLearningLoss]
    ) -> None:
        self._losses[name] = loss

    def get_loss(
        self, name: str, **kwargs: Dict[str, Any]
    ) -> ManifoldLearningLoss:
        creator = self._losses.get(name)
        if not creator:
            raise ValueError(
                f'Loss "{name}" is not registered: {list(self._losses.keys())}'
            )
        return creator(**kwargs)


def get_loss_by_name(
    name: str, **kwargs: Dict[str, Any]
) -> ManifoldLearningLoss:
    loss_factory = ManifoldLearningLossFactory()
    loss_factory.register_loss(
        name=SquaredReconstructionLoss.name, loss=SquaredReconstructionLoss
    )

    return loss_factory.get_loss(name=name, **kwargs)


if __name__ == "__main__":
    pass
