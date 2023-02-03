from coref.models import GeneralCorefModel
from coref.const import Doc, SampledData
from coref.models import load_coref_model, GeneralCorefModel
from config import Config


def get_training_iteration_docs(
    docs: list[Doc], sampled_docs: SampledData
) -> list[Doc]:
    for i, pseudo_doc in zip(sampled_docs.indices, sampled_docs.instances):
        docs[
            i
        ].simulation_token_annotations = pseudo_doc.simulation_token_annotations
    return sampled_docs.instances + [
        doc
        for i, doc in enumerate(docs)
        if i not in sampled_docs.indices
        and doc.simulation_token_annotations.tokens
    ]


def run_simulation(
    model: GeneralCorefModel,
    config: Config,
    train_docs: list[Doc],
    test_data: list[Doc],
    dev_docs: list[Doc],
) -> None:

    al_config = config.active_learning
    # Training split
    sampled_data = model.sample_unlabled_data(train_docs)
    training_data = get_training_iteration_docs(train_docs, sampled_data)

    # First training
    model.train(docs=training_data, docs_dev=dev_docs)
    model.evaluate(docs=test_data)

    model._logger.info("Yo Yo! Initial training iteration is done...\n")

    for i in range(al_config.simulation.active_learning_steps):
        model._logger.info(f" AL SIMULATION | round: {i}\n")
        # Prepare the data
        sampled_data = model.sample_unlabled_data(train_docs)
        training_data = get_training_iteration_docs(train_docs, sampled_data)

        # Train the model
        model = load_coref_model(config)
        model._logger.info("Training\n")
        model.train(docs=training_data, docs_dev=dev_docs)
        model._logger.info("Evaluation\n")
        model.evaluate(docs=test_data)
