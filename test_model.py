from coref import CorefModel
from run import output_running_time


def test_pipeline():
    config_path = "config.toml"
    section = "roberta"
    data_split = "pipeline_test"
    word_level = False
    # default batch_size is used

    config = CorefModel._load_config(config_path, section)
    config.device = "cpu"

    model = CorefModel(config, section)
    # no weights are loaded. Random init to test forward step
    with output_running_time():
        model.evaluate(data_split=data_split, word_level_conll=word_level)
