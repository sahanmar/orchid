""" Describes a model to extract coreferential spans from a list of tokens.

  Usage example:

  model = CorefModel("config.toml", "debug")
  model.evaluate("dev")
"""

from config.config import Config

from coref.models.mc_dropout_coref import MCDropoutCorefModel
from coref.models.coref_model import CorefModel
from coref.models.general_coref_model import GeneralCorefModel
from .exceptions import InvalidModelName


# When we will have more coref models, will define
# them in config and resolve it here. Slaaaaaaaay!


def load_coref_model(config: Config) -> GeneralCorefModel:
    _model_id = config.model_params.coref_model
    print(f"Attempting to load model: {_model_id}")
    if _model_id == "base":
        return CorefModel(config)
    elif _model_id == "mc_dropout":
        return MCDropoutCorefModel(config)
    raise InvalidModelName(model_id=_model_id)
