from coref.models import load_coref_model
from config import Config
from run import output_running_time
from copy import deepcopy
from coref.const import Doc


def test_pipeline(config: Config, dev_data: list[Doc]) -> None:
    word_level = False

    model = load_coref_model(config)
    # no weights are loaded. Random init to test forward step
    with output_running_time():
        model.evaluate(dev_data, word_level_conll=word_level)


def test_mc_dropout_pipeline(config: Config, dev_data: list[Doc]) -> None:
    word_level = False

    mc_dropout_conf = deepcopy(config)
    mc_dropout_conf.model_params.coref_model = "mc_dropout"
    model = load_coref_model(mc_dropout_conf)
    # no weights are loaded. Random init to test forward step
    with output_running_time():
        model.evaluate(dev_data, word_level_conll=word_level)


def test_al_pipeline(al_config: Config, dev_data: list[Doc]) -> None:
    word_level = False

    mc_dropout_conf = deepcopy(al_config)
    mc_dropout_conf.model_params.coref_model = "mc_dropout"
    model = load_coref_model(al_config)
    # no weights are loaded. Random init to test forward step
    with output_running_time():
        model.sample_unlabeled_data(dev_data)
