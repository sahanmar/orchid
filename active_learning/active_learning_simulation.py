from coref.models import GeneralCorefModel
from coref.const import Doc, SampledData
from coref.models import load_coref_model, GeneralCorefModel
from config import Config
from typing import Tuple
from copy import deepcopy
from random import shuffle


def get_training_iteration_docs(
    docs: list[Doc], sampled_docs: SampledData
) -> Tuple[list[Doc], list[Doc]]:
    for i, doc in zip(sampled_docs.indices, sampled_docs.instances):
        docs[i].simulation_token_annotations = doc.simulation_token_annotations
    return [
        doc for doc in docs if doc.simulation_token_annotations.tokens
    ], docs


def train_split(
    model: GeneralCorefModel, train_docs: list[Doc]
) -> Tuple[list[Doc], list[Doc]]:
    shuffle(train_docs)
    sampled_data = model.sample_unlabeled_data(train_docs)
    return get_training_iteration_docs(deepcopy(train_docs), sampled_data)


def run_training(
    model: GeneralCorefModel,
    training_data: list[Doc],
    dev_docs: list[Doc],
    test_data: list[Doc],
) -> None:
    model._logger.info("Training\n")
    model.train(docs=training_data, docs_dev=dev_docs)
    model._logger.info("Evaluation\n")
    model.evaluate(docs=test_data)


def get_logging_info(
    model: GeneralCorefModel, training_data: list[Doc], round: int
) -> None:
    tokens = sum(
        len(doc.simulation_token_annotations.tokens) for doc in training_data
    )
    model._logger.info(
        f" AL SIMULATION | round: {round}, Documents: {len(training_data)}, Tokens: {tokens}\n"
    )


def run_simulation(
    model: GeneralCorefModel,
    config: Config,
    train_docs: list[Doc],
    test_data: list[Doc],
    dev_docs: list[Doc],
) -> None:

    al_config = config.active_learning

    for i in range(al_config.sampling_strategy.total_number_of_iterations):
        training_data, train_docs = train_split(model, train_docs)
        model = load_coref_model(config)
        get_logging_info(model, training_data, round=i)
        run_training(model, training_data, dev_docs, test_data)
