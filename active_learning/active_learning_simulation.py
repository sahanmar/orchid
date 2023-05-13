import numpy as np
import os
import time
from coref.models import GeneralCorefModel
from coref.const import Doc, SampledData
from coref.models import load_coref_model, GeneralCorefModel
from config import Config
from typing import Tuple
from copy import deepcopy
from random import shuffle
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class DocumentsStats:
    avg_labels_per_doc: float
    max_labels_per_doc: float
    min_labels_per_doc: float
    labels_per_doc_std: float

    avg_labels_ratio_per_doc: float
    min_labels_ratio_per_doc: float
    max_labels_ratio_per_doc: float
    labels_ratio_per_doc_std: float

    avg_distance_from_labels: float
    max_distance_from_labels: float
    min_distance_from_labels: float
    distance_from_labels_std: float

    avg_furthest_labels: float
    max_furthest_labels: float
    min_furthest_labels: float
    furthest_labels_std: float

    @staticmethod
    def calculate(docs: list[Doc]) -> "DocumentsStats":
        (
            avg_labels_per_doc,
            max_labels_per_doc,
            min_labels_per_doc,
            labels_per_doc_std,
        ) = get_decriminative_stats(
            [
                len(doc.simulation_token_annotations.tokens)
                for doc in docs
                if doc.simulation_token_annotations.tokens
            ]
        )

        (
            avg_labels_ratio_per_doc,
            max_labels_ratio_per_doc,
            min_labels_ratio_per_doc,
            labels_ratio_per_doc_std,
        ) = get_decriminative_stats(
            [
                len(doc.simulation_token_annotations.tokens)
                / len(doc.cased_words)
                for doc in docs
                if doc.simulation_token_annotations.tokens
            ]
        )

        labels_distance_diff: list[float] = []
        furthest_labels: list[float] = []
        for doc in docs:
            sorted_sampled_tokens = sorted(
                doc.simulation_token_annotations.tokens
            )
            if len(sorted_sampled_tokens) > 0:
                labels_distance_diff.extend(
                    [
                        float(
                            sorted_sampled_tokens[i + 1]
                            - sorted_sampled_tokens[i]
                        )
                        for i in range(len(sorted_sampled_tokens) - 1)
                    ]
                )
                furthest_labels.append(
                    float(sorted_sampled_tokens[-1] - sorted_sampled_tokens[0])
                )
        (
            avg_distance_from_labels,
            max_distance_from_labels,
            min_distance_from_labels,
            distance_from_labels_std,
        ) = get_decriminative_stats(labels_distance_diff)

        (
            avg_furthest_labels,
            max_furthest_labels,
            min_furthest_labels,
            furthest_labels_std,
        ) = get_decriminative_stats(furthest_labels)

        return DocumentsStats(
            avg_labels_per_doc,
            max_labels_per_doc,
            min_labels_per_doc,
            labels_per_doc_std,
            avg_labels_ratio_per_doc,
            min_labels_ratio_per_doc,
            max_labels_ratio_per_doc,
            labels_ratio_per_doc_std,
            avg_distance_from_labels,
            max_distance_from_labels,
            min_distance_from_labels,
            distance_from_labels_std,
            avg_furthest_labels,
            max_furthest_labels,
            min_furthest_labels,
            furthest_labels_std,
        )


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
    loop: int,
) -> None:
    tokens = sum(
        len(doc.simulation_token_annotations.tokens) for doc in training_data
    )
    model._logger.info(
        {
            "al_simulation": {
                "loop": loop,
                "round": round,
                "documents": len(training_data),
                "tokens": tokens,
            }
        }
    )


def get_decriminative_stats(
    values: list[float],
) -> Tuple[float, float, float, float]:
    return (
        round(np.mean(values), 3),
        round(max(values), 3),
        round(min(values), 3),
        round(np.std(values), 3),
    )


def get_documents_stats(model: GeneralCorefModel, docs: list[Doc]) -> None:
    stats = DocumentsStats.calculate(docs)
    model._logger.info(
        {
            "sampling_stats": {
                "avg_labels_per_doc": stats.avg_labels_per_doc,
                "max_labels_per_doc": stats.max_labels_per_doc,
                "min_labels_per_doc": stats.min_labels_per_doc,
                "labels_per_doc_std": stats.labels_per_doc_std,
                "avg_labels_ratio_per_doc": stats.avg_labels_ratio_per_doc,
                "max_labels_ratio_per_doc": stats.max_labels_ratio_per_doc,
                "min_labels_ratio_per_doc": stats.min_labels_ratio_per_doc,
                "labels_ratio_per_doc_std": stats.labels_ratio_per_doc_std,
                "avg_distance_from_labels": stats.avg_distance_from_labels,
                "max_distance_from_labels": stats.max_distance_from_labels,
                "min_distance_from_labels": stats.min_distance_from_labels,
                "distance_from_labels_std": stats.distance_from_labels_std,
                "avg_furthest_labels": stats.avg_furthest_labels,
                "max_furthest_labels": stats.max_furthest_labels,
                "min_furthest_labels": stats.min_furthest_labels,
                "furthest_labels_std": stats.furthest_labels_std,
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

    # Article stats
    (
        avg_doc_tokens,
        max_doc_tokens,
        min_doc_tokens,
        doc_tokens_std,
    ) = get_decriminative_stats([len(doc.cased_words) for doc in train_docs])

    model._logger.info(
        {
            "article_stats": {
                "avg_doc_tokens": avg_doc_tokens,
                "max_doc_tokens": max_doc_tokens,
                "min_doc_tokens": min_doc_tokens,
                "doc_tokens_std": doc_tokens_std,
            }
        }
    )

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
                "active_learning_loops": al_config.simulation.active_learning_loops,
            }
        }
    )

    timestamp = int(time.time())
    path = os.path.join(config.data.data_dir, f"al_init_weights_{timestamp}.pt")
    model.save_weights(path=path)
    for loop in range(al_config.simulation.active_learning_loops):
        simulation_training_docs = deepcopy(train_docs)
        model = load_coref_model(config)
        model.load_weights(path=path)

        for al_round in range(
            al_config.sampling_strategy.total_number_of_iterations
        ):
            training_data, simulation_training_docs = train_split(
                model, simulation_training_docs
            )

            if al_config.cold_start or al_round == 0:
                model = load_coref_model(config)
                model.load_weights(path=path)

            get_logging_info(model, training_data, round=al_round, loop=loop)
            get_documents_stats(model, training_data)
            run_training(model, training_data, dev_docs, test_data)
