import argparse
import numpy as np
import matplotlib.gridspec as gridspec

import json
from pathlib import Path
from typing import Any, Callable
from itertools import groupby

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from tabulate import tabulate


def get_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data")
    return argparser.parse_args()


def format_data(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "loop": int(row["loop"]) + 1,
            "round": int(row["round"]),
            "documents": int(row["documents"]),
            "tokens": int(row["tokens"]),
            "avg_labels_ratio_per_doc": float(
                row["sampling_stats"]["avg_labels_ratio_per_doc"]
            ),
            # "docs_of_interest": int(row["docs_of_interest"]),
            # "acquisition_function": row["acquisition_function"],
            # "coref_model": row["coref_model"],
            "training": {
                "loss": [float(m["loss"]) for m in row["eval_metrics"][:-1]],
                "f1_lea": [float(m["f1_lea"]) for m in row["eval_metrics"][:-1]],
                "precision_lea": [
                    float(m["precision_lea"]) for m in row["eval_metrics"][:-1]
                ],
                "recall_lea": [
                    float(m["recall_lea"]) for m in row["eval_metrics"][:-1]
                ],
                "epoch": [int(m["epoch"]) + 1 for m in row["eval_metrics"][:-1]],
            },
            "evaluation": {
                "loss": float(row["eval_metrics"][-1]["loss"]),
                "f1_lea": float(row["eval_metrics"][-1]["f1_lea"]),
                "precision_lea": float(row["eval_metrics"][-1]["precision_lea"]),
                "recall_lea": float(row["eval_metrics"][-1]["recall_lea"]),
                "epoch": int(row["eval_metrics"][-1]["epoch"]) + 1,
            },
        }
        for row in data
    ]


def get_data(path: Path) -> dict[str, list[Any]]:
    combined_data: dict[str, Any] = {}

    for file in path.iterdir():
        if file.suffix == ".jsonl":
            data: list[dict[str, Any]] = []
            al_simulation: dict[str, Any] = {}
            eval_metrics: list[dict[str, Any]] = []

            with open(file, "r") as f:
                for line in f.readlines():
                    json_line = json.loads(line)["message"]
                    if "al_simulation" in json_line:
                        if al_simulation:
                            data.append(
                                {
                                    **al_simulation,
                                    **{"eval_metrics": eval_metrics},
                                    **{"sampling_stats": sampling_stats},
                                }
                            )
                            eval_metrics = []

                        al_simulation = json_line["al_simulation"]

                    elif "eval_metrics" in json_line:
                        eval_metrics.append(json_line["eval_metrics"])
                    elif "sampling_stats" in json_line:
                        sampling_stats = json_line["sampling_stats"]

                data.append(
                    {
                        **al_simulation,
                        **{"eval_metrics": eval_metrics},
                        **{"sampling_stats": sampling_stats},
                    }
                )

            if file.stem in combined_data:
                raise KeyError("Date will be overwritten... Its bad...")
            combined_data[file.stem] = format_data(data)

    return combined_data


def plot_evolutions(ax: plt.Axes, f1_data: dict[str, Any], colors: list[str]) -> None:
    ax.grid(alpha=0.2)
    absolute_max_f1 = 0

    sorted_f1_data = sorted(
        [(" ".join(k.split("_")), v) for k, v in f1_data.items()],
        key=lambda x: x[0],
    )

    # TMP HACK
    colors = ["b", "r"]

    for c, (simulation, results) in zip(colors, sorted_f1_data):
        avg_f1 = np.median(results, axis=0)
        if results.shape[0] > 1:
            max_f1 = np.partition(results, -2, axis=0)[-1]
            min_f1 = np.partition(results, 1, axis=0)[1]
        else:
            max_f1 = np.partition(results, 0, axis=0)[0]
            min_f1 = np.partition(results, 0, axis=0)[0]

        x_axis = list(range(1, len(avg_f1) + 1))

        ax.plot(x_axis, max_f1, linestyle="-", color=c, alpha=0.1)
        ax.plot(x_axis, min_f1, linestyle="-", color=c, alpha=0.1)

        ax.fill_between(
            x_axis,
            min_f1,
            max_f1,
            alpha=0.05,
            color=c,
        )

        max_f1_candidate = max(avg_f1)
        if max_f1_candidate > absolute_max_f1:
            absolute_max_f1 = max_f1_candidate
        ax.plot(
            x_axis,
            avg_f1,
            linestyle="--",
            markevery=5,
            marker="*",
            lw=1,
            label=simulation,
            color=c,
        )

    ax.plot(
        list(range(0, len(avg_f1) + 2)),
        [absolute_max_f1 for _ in range(0, len(avg_f1) + 2)],
        linestyle="--",
        lw=0.3,
        color="k",
    )

    ax.set_yticks(sorted(list(ax.get_yticks()) + [absolute_max_f1]))

    ax.set_xlabel("Active learning iterations")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, len(avg_f1) + 1)

    ax.set_ylabel("LEA")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # ax.set_title("Coreference resolution AL F1 evaluation")
    ax.legend(loc="lower right")


def plot_documents_to_read(
    ax: plt.Axes, documents_stats: dict[str, Any], colors: list[str]
) -> None:
    ax.grid(alpha=0.2)

    sorted_documents_stats = sorted(
        [(" ".join(k.split("_")), v) for k, v in documents_stats.items()],
        key=lambda x: x[0],
    )

    # TMP HACK
    colors = ["b", "r"]

    for c, (simulation, results) in zip(colors, sorted_documents_stats):
        avg_docs = np.mean(results, axis=0)
        max_docs = np.max(results, axis=0)
        min_docs = np.min(results, axis=0)

        x_axis = list(range(1, len(avg_docs) + 1))

        ax.plot(x_axis, max_docs, linestyle="-", color=c, alpha=0.1)
        ax.plot(x_axis, min_docs, linestyle="-", color=c, alpha=0.1)

        ax.fill_between(
            x_axis,
            min_docs,
            max_docs,
            alpha=0.05,
            color=c,
        )

        ax.plot(
            x_axis,
            avg_docs,
            linestyle="--",
            marker="+",
            lw=1,
            markevery=5,
            label=simulation,
            color=c,
        )

    ax.set_xlabel("Active learning iterations")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, len(avg_docs) + 1)

    ax.set_ylabel("Documents to read")

    # ax.set_title("Documents to read given AL iteration")
    ax.legend(loc="lower right")


def plot_labels_ratio_per_doc(
    ax: plt.Axes, labels_stats: dict[str, Any], colors: list[str]
):
    ax.grid(alpha=0.2)

    sorted_labels_stats = sorted(
        [(" ".join(k.split("_")), v) for k, v in labels_stats.items()],
        key=lambda x: x[0],
    )

    # TMP HACK
    colors = ["b", "r"]

    for c, (simulation, results) in zip(colors, sorted_labels_stats):
        avg_labels = np.mean(results, axis=0)
        max_labels = np.max(results, axis=0)
        min_labels = np.min(results, axis=0)

        x_axis = list(range(1, len(avg_labels) + 1))

        ax.plot(x_axis, max_labels, linestyle="-", color=c, alpha=0.1)
        ax.plot(x_axis, min_labels, linestyle="-", color=c, alpha=0.1)

        ax.fill_between(
            x_axis,
            min_labels,
            max_labels,
            alpha=0.05,
            color=c,
        )

        ax.plot(
            x_axis,
            avg_labels,
            linestyle="--",
            marker="+",
            markevery=5,
            lw=1,
            label=simulation,
            color=c,
        )

    ax.set_xlabel("Active learning iterations")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, len(avg_labels) + 1)

    ax.set_ylabel("Annotations ratio")

    # ax.set_title("Average annotated tokens to all tokens in a document ratio")
    ax.legend(loc="lower right")


def visualize(
    f1: dict[str, Any],
    documents_stats: dict[str, Any],
    labels_stats: dict[str, Any],
) -> None:
    # f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, sharey=False)

    # colors = ["b", "orange", "g", "r"]
    # plot_evolutions(ax1, f1, colors)
    # plot_documents_to_read(ax2, documents_stats, colors)
    # plot_labels_ratio_per_doc(ax3, labels_stats, colors)

    # f, (ax1, ax2) = plt.subplots(1, 2, sharex=False, sharey=False)

    # colors = ["b", "orange", "g", "r"]
    # plot_evolutions(ax1, f1, colors)
    # plot_documents_to_read(ax1, documents_stats, colors)
    # plot_labels_ratio_per_doc(ax2, labels_stats, colors)
    # f.suptitle("Document metrics for 200 documents of interest")

    f = plt.figure(layout="constrained")
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)
    ax2 = plt.subplot(
        gs[0, :2],
    )
    ax3 = plt.subplot(gs[0, 2:])
    ax1 = plt.subplot(gs[1, 1:3])

    colors = ["b", "orange", "g", "r"]
    plot_evolutions(ax1, f1, colors)
    plot_documents_to_read(ax2, documents_stats, colors)
    plot_labels_ratio_per_doc(ax3, labels_stats, colors)
    f.suptitle("Document metrics and LEA scores")
    plt.show()


def get_state_given_key(
    simulation_results: dict[str, list[Any]],
    key_func: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    return {
        simulation: np.asarray(
            [
                [res for _, res in g]
                for _, g in groupby(
                    sorted(
                        [(record["loop"], key_func(record)) for record in results],
                        key=lambda x: x[0],  # type: ignore
                    ),
                    key=lambda x: x[0],  # type: ignore
                )
            ]
        )
        for simulation, results in simulation_results.items()
    }


def print_table_to_console(data: dict[str, Any]):
    f1s = [["al_iters"] + [5, 10, 20, 30, 40, 50]]
    f1s_std = [["al_iters"] + [5, 10, 20, 30, 40, 50]]

    for simulation, results in data.items():
        avg_f1 = np.median(results, axis=0)
        std_f1 = np.std(results, axis=0)

        f1s.append([simulation] + avg_f1[[4, 9, 19, 29, 39, 49]].tolist())
        f1s_std.append([simulation] + std_f1[[4, 9, 19, 29, 39, 49]].tolist())

    print(tabulate(f1s))
    print(tabulate(f1s_std))


if __name__ == "__main__":
    args = get_args()
    simulation_results = get_data(Path(args.data))

    f1 = get_state_given_key(simulation_results, lambda x: x["evaluation"]["f1_lea"])
    print_table_to_console(f1)

    docs_of_interest = get_state_given_key(simulation_results, lambda x: x["documents"])
    print_table_to_console(docs_of_interest)

    labels_stats = get_state_given_key(
        simulation_results, lambda x: x["avg_labels_ratio_per_doc"]
    )
    print_table_to_console(labels_stats)

    visualize(f1, docs_of_interest, labels_stats)
