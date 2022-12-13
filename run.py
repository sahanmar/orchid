""" Runs experiments with CorefModel.

Try 'python run.py -h' for more details.
"""

import argparse
from contextlib import contextmanager
import datetime
import random
import sys
import time
from typing import Iterator

import numpy as np  # type: ignore
import torch  # type: ignore

from coref.models import load_coref_model
from config import Config
from coref.data_utils import get_docs, DataType
from active_learning.active_learning_simulation import run_simulation


@contextmanager
def output_running_time() -> Iterator[None]:
    """Prints the time elapsed in the context"""
    start = int(time.time())
    try:
        yield
    finally:
        end = int(time.time())
        delta = datetime.timedelta(seconds=end - start)
        print(f"Total running time: {delta}")


def seed(value: int) -> None:
    """Seed random number generators to get reproducible results"""
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "mode", choices=("train", "eval", "metrics", "simulation")
    )
    argparser.add_argument("experiment")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument(
        "--batch-size",
        type=int,
        help="Adjust to override the config value if you're"
        " experiencing out-of-memory issues",
    )
    argparser.add_argument(
        "--warm-start",
        action="store_true",
        help="If set, the training will resume from the"
        " last checkpoint saved if any. Ignored in"
        " evaluation modes."
        " Incompatible with '--weights'.",
    )
    argparser.add_argument(
        "--weights",
        help="Path to file with weights to load."
        " If not supplied, in 'eval' mode the latest"
        " weights of the experiment will be loaded;"
        " in 'train' mode no weights will be loaded.",
    )
    argparser.add_argument(
        "--word-level",
        action="store_true",
        help="If set, output word-level conll-formatted"
        " files in evaluation modes. Ignored in"
        " 'train' mode.",
    )
    args = argparser.parse_args()

    if args.warm_start and args.weights is not None:
        print(
            "The following options are incompatible:"
            " '--warm_start' and '--weights'",
            file=sys.stderr,
        )
        sys.exit(1)

    # seed(2020)

    # Load config
    config = Config.load_config(args.config_file, args.experiment)
    model = load_coref_model(config)

    # Load data
    train_data = get_docs(DataType.train, config=config)
    test_data = get_docs(DataType.test, config=config)
    dev_data = get_docs(DataType.dev, config=config)

    # TODO must be also in config
    if args.batch_size:
        model.config.model_params.a_scoring_batch_size = args.batch_size

    if args.mode == "train":
        if args.weights is not None or args.warm_start:
            model.load_weights(
                path=args.weights,
                map_location="cpu",
                noexception=args.warm_start,
            )
        with output_running_time():
            model.train(docs=train_data, docs_dev=dev_data)
    elif args.mode == "eval":
        model.load_weights(
            path=args.weights,
            map_location="cpu",
            ignore={
                "bert_optimizer",
                "general_optimizer",
                "bert_scheduler",
                "general_scheduler",
            },
        )
        model.evaluate(test_data, word_level_conll=args.word_level)
    elif args.mode == "simulation":
        run_simulation(model, train_data, test_data, dev_data)
    else:
        model.load_weights(
            path=args.weights,
            map_location="cpu",
            ignore={
                "bert_optimizer",
                "general_optimizer",
                "bert_scheduler",
                "general_scheduler",
            },
        )
        model.get_uncertainty_metrics(test_data)
