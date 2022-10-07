""" Describes Config, a simple namespace for config values.

For description of all config values, refer to config.toml.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import toml  # type: ignore[import]


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """Contains values needed to set up the coreference model."""

    section: str

    data_dir: str

    train_data: str
    dev_data: str
    test_data: str
    # TODO This doesnt look like a good fit. Maybe config override for test purposes?
    pipeline_test_data: str

    num_of_training_docs: int = field(init=False)

    device: str

    bert_model: str
    bert_window_size: int

    embedding_size: int
    sp_embedding_size: int
    a_scoring_batch_size: int
    hidden_size: int
    n_hidden_layers: int

    max_span_len: int

    rough_k: int

    bert_finetune: bool
    dropout_rate: float
    learning_rate: float
    bert_learning_rate: float
    train_epochs: int
    bce_loss_weight: float

    tokenizer_kwargs: Dict[str, dict]
    conll_log_dir: str

    def __post_init__(self):
        with open(Path(self.train_data), "r") as f:
            self.num_of_training_docs = sum(1 for _ in f)

    @staticmethod
    def load_config(config_path: str, section: str = "roberta") -> "Config":
        config = toml.load(config_path)
        default_section = config["DEFAULT"]
        current_section = config[section]
        unknown_keys = set(current_section.keys()) - set(default_section.keys())
        if unknown_keys:
            raise ValueError(f"Unexpected config keys: {unknown_keys}")
        return Config(section, **{**default_section, **current_section})

    @staticmethod
    def load_default_config(section: Optional[str]) -> "Config":
        if section is not None:
            return Config.load_config("config.toml", section)
        return Config.load_config("config.toml")
