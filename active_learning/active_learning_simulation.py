import torch
from coref.models import GeneralCorefModel
from coref.const import Doc, SampledData
from coref.models import load_coref_model, GeneralCorefModel
from config import Config
from typing import Tuple
from copy import deepcopy
from random import shuffle

DEFAULT_CFG = "config.toml"


def get_training_iteration_docs(
    docs: list[Doc], sampled_docs: SampledData
) -> Tuple[list[Doc], list[Doc]]:
    for i, doc in zip(sampled_docs.indices, sampled_docs.instances):
        docs[i].simulation_token_annotations = doc.simulation_token_annotations
    return [
        doc for doc in docs if doc.simulation_token_annotations.tokens
    ], docs


def train_split(
    model: GeneralCorefModel, docs: list[Doc]
) -> Tuple[list[Doc], list[Doc]]:
    shuffle(docs)
    sampled_data = model.sample_unlabeled_data(docs)
    return get_training_iteration_docs(deepcopy(docs), sampled_data)


def run_training(
    model: GeneralCorefModel,
    training_data: list[Doc],
    dev_docs: list[Doc],
    test_data: list[Doc],
) -> None:
    model._logger.info("training")
    model.train(docs=training_data, docs_dev=dev_docs)
    model._logger.info("evaluation")
    model.evaluate(docs=test_data)


def get_logging_info(
    model: GeneralCorefModel,
    training_data: list[Doc],
    round: int,
) -> None:
    tokens = sum(
        len(doc.simulation_token_annotations.tokens) for doc in training_data
    )
    model._logger.info(
        {
            "al_simulation": {
                "round": round,
                "documents": len(training_data),
                "tokens": tokens,
            }
        }
    )


def run_simulation(
    model: GeneralCorefModel,
    config: Config,
    train_docs: list[Doc],
    test_data: list[Doc],
    dev_docs: list[Doc],
) -> None:
    al_config = config.active_learning
    coref_model = config.model_params.coref_model

    # Information about the run
    model._logger.info(
        {
            "metadata": {
                "type": "Active Learning simulation",
                "docs_of_interest": al_config.sampling_strategy.docs_of_interest,
                "acquisition_function": al_config.instance_sampling.value,
                "batch_size": al_config.sampling_strategy.batch_size,
                "train_epochs": config.training_params.train_epochs,
                "coref_model": coref_model,
                "bert_model": config.model_params.bert_model,
                "bert_finetune": config.training_params.bert_finetune,
            }
        }
    )

    for al_round in range(
        al_config.sampling_strategy.total_number_of_iterations
    ):
        training_data, train_docs = train_split(model, train_docs)
        del model
        model = load_coref_model(config)
        get_logging_info(
            model,
            training_data,
            round=al_round,
        )
        run_training(model, training_data, dev_docs, test_data)
