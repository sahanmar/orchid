""" Describes a model to extract coreferential spans from a list of tokens.

  Usage example:

  model = CorefModel("config.toml", "debug")
  model.evaluate("dev")
"""

from config.config import Config

from coref.models.mc_dropout_coref import MCDropoutCorefModel
from coref.models.coref_model import CorefModel
from coref.models.general_coref_model import GeneralCorefModel


# When we will have more coref models, will define
# them in config and resolve it here. Slaaaaaaaay!


def load_coref_model(config: Config) -> GeneralCorefModel:
    if config.model_params.coref_model == "base":
        return CorefModel(config)
    elif config.model_params.coref_model == "mc_dropout":
        return MCDropoutCorefModel(config)
    raise ValueError(
        "Amigo, bad news... {config.model_params.coref_model} is not on the table..."
    )
