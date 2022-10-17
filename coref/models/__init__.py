""" Describes a model to extract coreferential spans from a list of tokens.

  Usage example:

  model = CorefModel("config.toml", "debug")
  model.evaluate("dev")
"""

from coref.models.coref_model import CorefModel


# When we will have more coref models, will define
# them in config and resolve it here. Slaaaaaaaay!
__all__ = ["CorefModel"]
