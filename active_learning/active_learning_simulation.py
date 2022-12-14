from coref.models import GeneralCorefModel
from coref.const import Doc, SampledData
from typing import Optional


def run_simulation(
    model: GeneralCorefModel,
    train_docs: list[Doc],
    test_data: list[Doc],
    dev_docs: list[Doc],
) -> None:
    al_config = model.config.active_learning

    # Training split
    training_data = train_docs[: al_config.simulation.initial_sample_size]
    complementary_data = train_docs[al_config.simulation.initial_sample_size :]
    # First training
    model.train(docs=training_data, docs_dev=dev_docs)
    model.evaluate(docs=test_data)

    print("Yo Yo! Initial training iteration is done...\n")

    for i in range(al_config.simulation.active_learning_steps):
        print(f"Active Learning round {i}...\n")
        # Prepare the data
        batch: SampledData = model.sample_unlabled_data(complementary_data)
        training_data.extend(batch.instances)
        complementary_data = [
            doc
            for i, doc in enumerate(complementary_data)
            if i not in batch.indices
        ]

        # Train the model
        model._build_model()
        print("Training\n")
        model.train(docs=training_data, docs_dev=dev_docs)
        print("Evaluation\n")
        model.evaluate(docs=test_data)
