import argparse
from typing import List, Tuple

import jsonlines
import torch
from tqdm import tqdm

from coref.models import CorefModel
from coref.tokenizer_customization import *


def build_doc(doc: dict, model: CorefModel) -> dict:
    filter_func = TOKENIZER_FILTERS.get(
        model.config.model_params.bert_model, lambda _: True
    )
    token_map = TOKENIZER_MAPS.get(model.config.model_params.bert_model, {})

    word2subword: List[Tuple[int, int]] = []
    subwords: List[int] = []
    word_id: List[int] = []
    for i, word in enumerate(doc["cased_words"]):
        tokenized_word = (
            token_map[word]
            if word in token_map
            else model.tokenizer.tokenize(word)
        )
        tokenized_word = list(filter(filter_func, tokenized_word))
        word2subword.append(
            (len(subwords), len(subwords) + len(tokenized_word))
        )
        subwords.extend(tokenized_word)
        word_id.extend([i] * len(tokenized_word))
    doc["word2subword"] = word2subword
    doc["subwords"] = subwords
    doc["word_id"] = word_id

    doc["head2span"] = []
    if "speaker" not in doc:
        doc["speaker"] = ["_" for _ in doc["cased_words"]]
    doc["word_clusters"] = []
    doc["span_clusters"] = []

    return doc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("experiment")
    argparser.add_argument("input_file")
    argparser.add_argument("output_file")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument(
        "--batch-size",
        type=int,
        help="Adjust to override the config value if you're"
        " experiencing out-of-memory issues",
    )
    argparser.add_argument(
        "--weights",
        help="Path to file with weights to load."
        " If not supplied, in the latest"
        " weights of the experiment will be loaded;"
        " if there aren't any, an error is raised.",
    )
    args = argparser.parse_args()

    model = CorefModel(args.config_file, args.experiment)

    if args.batch_size:
        model.config.model_params.a_scoring_batch_size = args.batch_size

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
    model.training = False

    with jsonlines.open(args.input_file, mode="r") as input_data:
        docs = [build_doc(doc, model) for doc in input_data]

    with torch.no_grad():
        for doc in tqdm(docs, unit="docs"):
            result = model.run(doc)
            doc["span_clusters"] = result.span_clusters
            doc["word_clusters"] = result.word_clusters

            for key in ("word2subword", "subwords", "word_id", "head2span"):
                del doc[key]

    with jsonlines.open(args.output_file, mode="w") as output_data:
        output_data.write_all(docs)
