from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def get_embedded_2d_ellipsis(
    dim: int = 2,
    a: float = 1.0,
    b: float = 1.0,
    x0: float = 0.0,
    y0: float = 0,
    scale: float = 1.0,
    steps: int = 100,
    noise: float = 1e-2,
    random_state: Optional[float] = None,
) -> torch.Tensor:
    assert dim >= 2
    assert a > 0
    assert b > 0
    assert scale > 0
    assert steps > 2
    assert noise > 0
    phi = torch.linspace(-torch.pi, torch.pi, steps=steps)

    x = torch.permute(
        torch.linspace(0.0, a, steps=steps).expand([1, steps]), dims=[1, 0]
    ) * torch.cos(phi)
    x = x.view(steps**2) + x0
    x = x.expand([1, x.size(0)])
    y = torch.permute(
        torch.linspace(0.0, b, steps=steps).expand([1, steps]), dims=[1, 0]
    ) * torch.sin(phi)
    y = y.view(steps**2) + y0
    y = y.expand([1, y.size(0)])

    # Embedding the ellipsis into a space of a higher dimension
    extra_dim = torch.zeros(dim - 2, x.size(1))
    if random_state is not None:
        torch.set_rng_state(
            new_state=torch.tensor(random_state, dtype=torch.int)
        )
    # Add noise to extra dimensions
    noise_tensor = torch.randn(*extra_dim.size()) * noise
    extra_dim += noise_tensor
    ell = torch.cat((x, y, extra_dim), dim=0)
    ell = torch.permute(ell, dims=[1, 0])
    return ell  # [n, dim]


class ManifoldDataset(Dataset):
    def __init__(self, inputs: torch.Tensor):
        super(ManifoldDataset, self).__init__()
        self._x = inputs

    def __getitem__(self, item: int) -> torch.Tensor:
        return self._x[item]

    def __len__(self) -> int:
        return int(self._x.size(0))


class ManifoldDataloader(DataLoader):
    def __init__(self, inputs: torch.Tensor):
        super(ManifoldDataloader, self).__init__(
            dataset=ManifoldDataset(inputs=inputs),
            batch_size=32,
            shuffle=True,
        )


class BasePCA(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(BasePCA, self).__init__()

        self.pca_layer = nn.Linear(
            in_features=in_features, out_features=out_features
        )

        self.optimizer = optim.Adam(params=self.pca_layer.parameters(), lr=1e-2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.pca_layer(inputs)
        return embeddings  # [n, target_dim]

    def train_step(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.optimizer.zero_grad()
        with torch.enable_grad():
            embeddings = self(inputs)
            loss_step = self.loss(inputs=inputs, embeddings=embeddings)
            loss_step.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return {"embeddings": embeddings, "loss": loss_step}

    def train_on_inputs(self, inputs: torch.Tensor, epochs: int = 10) -> None:
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            for b in ManifoldDataloader(inputs=inputs):
                res = self.train_step(inputs=b)
                pf = {
                    k_: v_.detach().numpy()
                    for k_, v_ in res.items()
                    if k_ in ["loss"]
                }
                pbar.set_postfix(pf)

    def loss(
        self, inputs: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Squared Reconstruction Error"""
        x_rec = torch.matmul(
            (embeddings - self.pca_layer.bias), self.pca_layer.weight
        )  # [n, original_dim]
        return torch.sum(torch.subtract(inputs, x_rec) ** 2)


if __name__ == "__main__":
    # TMP debugging code
    data = get_embedded_2d_ellipsis(dim=10, a=5, y0=10, x0=5, noise=1)
    data_np = data.numpy()

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    data_emb = pca.fit_transform(X=data_np)

    # torch "pca_lowrank"
    u, s, v = torch_pca = torch.pca_lowrank(data, niter=10)
    # Project the data to the first k principal components
    k = 2
    data_emb_torch = torch.matmul(data, v[:, :k])

    # Custom PCA
    pca_custom = BasePCA(in_features=data.size(1), out_features=2)
    pca_custom.train_on_inputs(inputs=data)
    data_emb_custom = pca_custom(data)

    # Plotting
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)

    ax.scatter(
        data_np[:, 0], data_np[:, 1], alpha=0.75, label="original-first-2d"
    )
    ax.scatter(data_emb[:, 0], data_emb[:, 1], alpha=0.75, label="pca-sklearn")
    ax.scatter(
        data_emb_torch[:, 0],
        data_emb_torch[:, 1],
        alpha=0.75,
        label="pca-pca_lowrank",
    )
    ax.scatter(
        data_emb_custom.detach().numpy()[:, 0],
        data_emb_custom.detach().numpy()[:, 1],
        alpha=0.75,
        label="pca-custom",
    )

    ax.set_xlabel("$Y^1$")
    ax.set_ylabel("$Y^2$")
    fig.legend()
    plt.show()
