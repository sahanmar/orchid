"""
This module is responsible for reading json-line data and its tokenization.
"""

import pickle
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import List, Tuple

import jsonlines

from config import Config
from coref.const import Doc
from coref.tokenizer_customization import TOKENIZER_FILTERS, TOKENIZER_MAPS


class DataType(Enum):
    train = "train_data"
    test = "test_data"
    dev = "dev_data"


def tokenize_docs(path: Path, config: Config) -> List[Doc]:
    print(f"Tokenizing documents at {path}...", flush=True)
    out: List[Doc] = []
    filter_func = TOKENIZER_FILTERS.get(
        config.model_params.bert_model, lambda _: True
    )
    token_map = TOKENIZER_MAPS.get(config.model_params.bert_model, {})
    with jsonlines.open(path, mode="r") as data_f:
        for doc in data_f:
            doc["span_clusters"] = [
                [tuple(mention) for mention in cluster]
                for cluster in doc["span_clusters"]
            ]
            word2subword: List[Tuple[int, int]] = []
            subwords: List[int] = []
            word_id: List[int] = []
            for i, word in enumerate(doc["cased_words"]):
                tokenized_word = (
                    token_map[word]
                    if word in token_map
                    else config.model_bank.tokenizer.tokenize(word)
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
            out.append(doc)
    print("Tokenization OK", flush=True)
    return out


def get_docs(data_type: DataType, config: Config) -> List[Doc]:
    path = asdict(config.data)[data_type.value]
    model_name = config.model_params.bert_model.replace("/", "_")
    cache_filename = Path(f"{model_name}_{path.name}.pickle")
    if cache_filename.exists():
        with open(cache_filename, mode="rb") as cache_f:
            docs = pickle.load(cache_f)
    else:
        docs = tokenize_docs(path, config)
        with open(cache_filename, mode="wb") as cache_f:
            pickle.dump(docs, cache_f)
    return docs
