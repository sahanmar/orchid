from coref.models import CorefModel
from config import Config
from run import output_running_time
from coref.data_utils import get_docs, DataType


def test_pipeline() -> None:
    word_level = False

    config = Config.load_default_config(section="debug")
    data = get_docs(DataType.test, config)

    model = CorefModel(config)
    # no weights are loaded. Random init to test forward step
    with output_running_time():
        model.evaluate(data, word_level_conll=word_level)
