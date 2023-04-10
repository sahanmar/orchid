import argparse
import numpy as np
import json
from pathlib import Path
from typing import Any
from itertools import groupby

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter


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
            # "docs_of_interest": int(row["docs_of_interest"]),
            # "acquisition_function": row["acquisition_function"],
            # "coref_model": row["coref_model"]
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
                                {**al_simulation, **{"eval_metrics": eval_metrics}}
                            )
                            eval_metrics = []

                        al_simulation = json_line["al_simulation"]

                    elif "eval_metrics" in json_line:
                        eval_metrics.append(json_line["eval_metrics"])

                data.append({**al_simulation, **{"eval_metrics": eval_metrics}})

            if file.stem in combined_data:
                raise KeyError("Date will be overwritten... Its bad...")
            combined_data[file.stem] = format_data(data)

    return combined_data


def plot_evolutions(f1_data: dict[str, np.ndarray]) -> None:

    ax = plt.subplot()
    ax.grid(alpha=0.2)

    for simulation, results in f1_data.items():
        avg_f1 = np.mean(results, axis=0)

        ax.plot(
            list(range(1, len(avg_f1) + 1)),
            avg_f1,
            linestyle="--",
            marker="*",
            lw=1,
            label=simulation,
        )

    ax.set_xlabel("Active learning iterations")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, len(avg_f1) + 1)

    ax.set_ylabel("F1 Lea")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.set_title("Coreference resolution AL F1 evaluation, 4 training epochs")
    ax.legend(loc="lower right")

    plt.show()


if __name__ == "__main__":
    args = get_args()
    simulation_results = get_data(Path(args.data))

    f1 = {
        simulation: np.asarray(
            [
                [res for _, res in g]
                for _, g in groupby(
                    sorted(
                        [
                            (record["loop"], record["evaluation"]["f1_lea"])
                            for record in results
                        ],
                        key=lambda x: x[0],
                    ),
                    key=lambda x: x[0],
                )
            ]
        )
        for simulation, results in simulation_results.items()
    }

    plot_evolutions(f1)
    # import IPython

    # IPython.embed()
