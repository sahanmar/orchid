import torch
from typing import Any
from sklearn.cluster import AgglomerativeClustering


def hac_sampling(
    instances: list[torch.Tensor],
    entropy_values: list[int],
    docs_of_interest: int,
    criterion: str = "size",
) -> list[int]:

    model = AgglomerativeClustering(linkage="average").fit(
        torch.stack(instances).numpy()
    )
    cluster_tree = dict(enumerate(model.children_, model.n_leaves_))

    cluster_criterion = []
    high_entropy_clusters: dict[int, list[int]] = {}
    for high_entropy_val_idx in entropy_values:
        for cluster, components in cluster_tree.items():
            if high_entropy_val_idx in components:
                high_entropy_clusters.setdefault(cluster, []).append(
                    high_entropy_val_idx
                )
                if cluster not in cluster_criterion:
                    cluster_criterion.append(cluster)

    if criterion == "size":
        cluster_criterion = [
            cluster
            for cluster, _ in sorted(
                [
                    (key, _get_cluster_sizes(key, cluster_tree))
                    for key in cluster_tree
                ],
                key=lambda x: x[1],
            )
        ]

    ids_2_return: list[int] = []
    for cluster_id in cluster_criterion:
        if len(ids_2_return) > docs_of_interest:
            break
        if (
            cluster_id in high_entropy_clusters
            and high_entropy_clusters[cluster_id]
        ):
            index_to_add = high_entropy_clusters[cluster_id].pop(0)
            ids_2_return.append(index_to_add)

    if len(ids_2_return) > docs_of_interest:
        print(
            f"Smth went wrong, only {len(ids_2_return)} values sampled instead of {docs_of_interest}"
        )

    return ids_2_return


def _get_cluster_sizes(key: int, cluster_tree: dict[int, Any]) -> int:
    cluster_size = 0
    for leaf in cluster_tree[key]:
        if leaf in cluster_tree:
            cluster_size += _get_cluster_sizes(leaf, cluster_tree)
        else:
            cluster_size += 1

    return cluster_size
