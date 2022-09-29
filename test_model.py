from coref import CorefModel
from coref.config import Config
from run import output_running_time


def test_pipeline():
    data_split = "pipeline_test"
    word_level = False

    config = Config.load_default_config(section="debug")

    model = CorefModel(config)
    # no weights are loaded. Random init to test forward step
    with output_running_time():
        model.evaluate(data_split=data_split, word_level_conll=word_level)
