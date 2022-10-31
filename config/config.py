"""
Describes Config, a simple namespace for config values.
For description of all config values, refer to config.toml.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List

import toml
from transformers import AutoTokenizer, AutoModel

from active_learning.exploration import GreedySampling
from config.config_utils import overwrite_config
from coref.bert import load_bert


@overwrite_config
@dataclass
class ActiveLearning:
    parameters_samples: int


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

    def __post_init__(self) -> None:
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

    # TODO Change to enum in load config for this class
    coref_model: str

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


# region Manifold Learning Configuration
@overwrite_config
@dataclass
class ManifoldLearningParams:
    loss_name: str
    # For separate (non-CR) use cases
    batch_size: int
    shuffle: bool
    learning_rate: float
    epochs: int
    input_dimensionality: Optional[int] = None
    output_dimensionality: Optional[int] = None
    verbose_outputs: List[str] = field(default_factory=list)


# endregion


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """Contains values needed to set up the coreference model."""

    section: str

    data: Data
    model_params: ModelParams
    training_params: TrainingParams

    tokenizer_kwargs: Dict[str, str]

    # AL sampling_strategy. Will be added to AL section later
    sampling_strategy: GreedySampling

    active_learning: ActiveLearning

    # Manifold Learning
    manifold: ManifoldLearningParams

    model_bank: ModelBank = field(init=False)

    def __post_init__(self) -> None:
        encoder, tokenizer = load_bert(
            self.model_params.bert_model,
            self.tokenizer_kwargs,
            self.training_params.device,
        )
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

        tokenizer_kwargs = default_conf["tokenizer_kwargs"]

        sampling_strategy = GreedySampling.load_config(
            default_conf["sampling_strategy"],
            overwrite_conf.get("sampling_strategy", {}),
        )

        # noinspection PyArgumentList
        active_learning = ActiveLearning(  # type: ignore[call-arg]
            default_conf["active_learning"],
            overwrite_conf.get("active_learning", {}),
        )

        # Loading the manifold learning configuration
        # noinspection PyArgumentList
        manifold_learning = ManifoldLearningParams(  # type: ignore[call-arg]
            config=default_conf["manifold_learning"],
            overwrite=overwrite_conf.get("manifold_learning", {}),
        )

        return Config(
            section=section,
            data=data,
            model_params=model_params,
            training_params=training_params,
            tokenizer_kwargs=tokenizer_kwargs,
            sampling_strategy=sampling_strategy,
            active_learning=active_learning,
            manifold=manifold_learning,
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
