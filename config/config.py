"""
Describes Config, a simple namespace for config values.
For description of all config values, refer to config.toml.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List

import toml  # type: ignore
from transformers import AutoTokenizer, AutoModel

from config.active_learning import ActiveLearning
from config.config_utils import overwrite_config, get_overwrite_value
from config.metrics import Metrics
from coref.bert import load_bert


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
@dataclass
class ManifoldLearningParamsStandalone:
    # For separate (non-CR) use cases
    batch_size: int
    shuffle: bool
    learning_rate: float
    epochs: int
    input_dimensionality: Optional[int] = None
    output_dimensionality: Optional[int] = None


@dataclass
class ManifoldLearningParams:
    enable: bool
    loss_name: str
    loss_alpha: float
    reduction_ratio: float
    standalone: ManifoldLearningParamsStandalone
    verbose_outputs: List[str] = field(default_factory=list)

    @staticmethod
    @overwrite_config
    def from_config(
        **kwargs: Any,
    ) -> "ManifoldLearningParams":
        _manifold_standalone_dict = kwargs.pop("standalone")
        _manifold_standalone = ManifoldLearningParamsStandalone(
            **_manifold_standalone_dict,
        )
        return ManifoldLearningParams(
            standalone=_manifold_standalone,
            **kwargs,
        )

    def __post_init__(self) -> None:
        assert 0.0 < self.reduction_ratio < 1.0


# endregion


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """Contains values needed to set up the coreference model."""

    section: str

    data: Data
    model_params: ModelParams
    training_params: TrainingParams

    tokenizer_kwargs: Dict[str, str]

    active_learning: ActiveLearning

    # Manifold Learning
    manifold: ManifoldLearningParams

    model_bank: ModelBank = field(init=False)

    metrics: Metrics

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
            default_conf["data"], get_overwrite_value(overwrite_conf, "data")
        )
        model_params = ModelParams(
            default_conf["model_params"],
            get_overwrite_value(overwrite_conf, "model_params"),  # type: ignore
        )
        training_params = TrainingParams(
            default_conf["training_params"],
            get_overwrite_value(overwrite_conf, "training_params"),  # type: ignore
        )

        tokenizer_kwargs = default_conf["tokenizer_kwargs"]

        active_learning = ActiveLearning.load_config(
            default_conf["active_learning"],
            get_overwrite_value(overwrite_conf, "active_learning"),  # type: ignore
        )

        metrics = Metrics.load_config(
            default_conf["metrics"],
            get_overwrite_value(overwrite_conf, "metrics"),  # type: ignore
        )

        # Loading the manifold learning configuration
        manifold_learning = ManifoldLearningParams.from_config(  # type: ignore[call-arg]
            config=default_conf["manifold_learning"],
            overwrite=get_overwrite_value(overwrite_conf, "manifold_learning"),
        )

        return Config(
            section=section,
            data=data,
            model_params=model_params,
            training_params=training_params,
            tokenizer_kwargs=tokenizer_kwargs,
            active_learning=active_learning,
            metrics=metrics,
            manifold=manifold_learning,
        )

    @staticmethod
    def load_default_config(section: Optional[str]) -> "Config":
        if section is not None:
            return Config.load_config("config.toml", section)
        return Config.load_config("config.toml")
