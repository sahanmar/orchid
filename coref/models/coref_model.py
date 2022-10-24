from coref.models.general_coref_model import GeneralCorefModel
from config import Config


class CorefModel(GeneralCorefModel):
    def __init__(self, config: Config, epochs_trained: int = 0):
        super().__init__(config, epochs_trained)
