"""
Describes Config, a simple namespace for config values.
For description of all config values, refer to config.toml.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import toml
from transformers import AutoTokenizer, AutoModel

from coref import bert


def overwrite_config(dataclass_2_create):
    def load_overwritten_config(
        config: Dict[str, Any], overwrite: Dict[str, Any]
    ):
        unknown_keys = set(overwrite.keys()) - set(config.keys())
        if unknown_keys:
            raise ValueError(f"Unexpected config keys: {unknown_keys}")
        return dataclass_2_create(
            **{key: overwrite.get(key, val) for key, val in config.items()}
        )

    return load_overwritten_config


@dataclass
class ModelBank:
    encoder: AutoModel
    tokenizer: AutoTokenizer


@dataclass
class Data:
    data_dir: str

    train_data: Path
    dev_data: Path
    test_data: Path

    num_of_training_docs: int = field(init=False)

    def __post_init__(self):
        with open(self.train_data, "r") as f:
            self.num_of_training_docs = sum(1 for _ in f)

    @staticmethod
    def load_config(
        config: Dict[str, Any], overwrite: Dict[str, Any]
    ) -> "Data":
        unknown_keys = set(overwrite.keys()) - set(config.keys())
        if unknown_keys:
            raise ValueError(f"Unexpected config keys: {unknown_keys}")
        return Data(
            **{  # type: ignore[arg-type]
                key: Path(overwrite.get(key, val))
                for key, val in config.items()
            }
        )


@overwrite_config
@dataclass
class ModelParams:
    bert_model: str
    bert_window_size: int

    embedding_size: int
    sp_embedding_size: int
    a_scoring_batch_size: int
    hidden_size: int
    n_hidden_layers: int

    max_span_len: int

    rough_k: int


@overwrite_config
@dataclass
class TrainingParams:
    device: str
    bert_finetune: bool
    dropout_rate: float
    learning_rate: float
    bert_learning_rate: float
    train_epochs: int
    bce_loss_weight: float
    conll_log_dir: str


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """Contains values needed to set up the coreference model."""

    section: str

    data: Data
    model_params: ModelParams
    training_params: TrainingParams

    tokenizer_kwargs: Dict[str, dict]

    model_bank: ModelBank = field(init=False)

    def __post_init__(self):
        encoder, tokenizer = bert.load_bert(self)
        self.model_bank = ModelBank(encoder, tokenizer)

    @staticmethod
    def load_config(config_path: str, section: str = "roberta") -> "Config":
        config = toml.load(config_path)

        default_conf = config["DEFAULT"]
        overwrite_conf = config[section]

        data = Data.load_config(
            default_conf["data"], overwrite_conf.get("data", {})
        )
        model_params = ModelParams(  # type: ignore[call-arg]
            default_conf["model_params"], overwrite_conf.get("model_params", {})
        )
        training_params = TrainingParams(  # type: ignore[call-arg]
            default_conf["training_params"],
            overwrite_conf.get("training_params", {}),
        )

        tokenizer_kwards = default_conf["tokenizer_kwargs"]

        return Config(
            section, data, model_params, training_params, tokenizer_kwards
        )

    @staticmethod
    def load_default_config(section: Optional[str]) -> "Config":
        if section is not None:
            return Config.load_config("config.toml", section)
        return Config.load_config("config.toml")


def strs_2_paths(
    default_dict: Dict[str, Any], current_dict: Dict[str, Any]
) -> Dict[str, Path]:
    return {
        "train_data": Path(
            current_dict.get("train_data", default_dict["train_data"])
        ),
        "dev_data": Path(
            current_dict.get("dev_data", default_dict["dev_data"])
        ),
        "test_data": Path(
            current_dict.get("test_data", default_dict["test_data"])
        ),
    }
