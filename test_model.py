from coref.models import load_coref_model
from config import Config
from run import output_running_time
from coref.data_utils import get_docs, DataType

CONFIG = Config.load_default_config(section="debug")
DATA = get_docs(DataType.test, CONFIG)


def test_pipeline() -> None:
    word_level = False

    model = load_coref_model(CONFIG)
    # no weights are loaded. Random init to test forward step
    with output_running_time():
        model.evaluate(DATA, word_level_conll=word_level)


def test_mc_dropout_pipeline() -> None:
    word_level = False

    CONFIG.model_params.coref_model = "mc_dropout"
    model = load_coref_model(CONFIG)
    # no weights are loaded. Random init to test forward step
    with output_running_time():
        model.evaluate(DATA, word_level_conll=word_level)
