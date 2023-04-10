import argparse
import numpy as np
import json
from pathlib import Path
from typing import Any


def get_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data")
    return argparser.parse_args()


def get_data(path: Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    al_simulation: dict[str, Any] = {}
    eval_metrics: list[dict[str, Any]] = []
    with open(path, "r") as f:
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

    return [
        {
            "loop": int(row["loop"]) + 1,
            "round": int(row["round"]),
            "documents": int(row["documents"]),
            "tokens": int(row["tokens"]),
            "docs_of_interest": int(row["docs_of_interest"]),
            "acquisition_function": row["acquisition_function"],
            "training": {
                "loss": [float(m["loss"]) for m in row["eval_metrics"][:-1]],
                "f1_lea": [
                    float(m["f1_lea"]) for m in row["eval_metrics"][:-1]
                ],
                "precision_lea": [
                    float(m["precision_lea"]) for m in row["eval_metrics"][:-1]
                ],
                "recall_lea": [
                    float(m["recall_lea"]) for m in row["eval_metrics"][:-1]
                ],
                "epoch": [
                    int(m["epoch"]) + 1 for m in row["eval_metrics"][:-1]
                ],
            },
            "evaluation": {
                "loss": float(row["eval_metrics"][-1]["loss"]),
                "f1_lea": float(row["eval_metrics"][-1]["f1_lea"]),
                "precision_lea": float(
                    row["eval_metrics"][-1]["precision_lea"]
                ),
                "recall_lea": float(row["eval_metrics"][-1]["recall_lea"]),
                "epoch": int(row["eval_metrics"][-1]["epoch"]) + 1,
            },
        }
        for row in data
    ]


# def plot_evolutions(f1_data: np.ndarray) -> None:
#     ...


if __name__ == "__main__":
    args = get_args()
    data = get_data(args.data)

    # import IPython

    # IPython.embed()
