from typing import List

from config import Config
from coref.data_utils import get_docs, DataType
from coref.const import Doc
from active_learning.acquisition_functions import random_sampling

BATCH_SIZE = 7


def mock_input() -> List[Doc]:
    config = Config.load_default_config(section="debug")
    # Unfortunately we have only one debug doc
    data = get_docs(DataType.test, config)
    return 10 * data  # lest make a list of 10 same docs


def test_random_sampling() -> None:
    instances = mock_input()
    sampled_data = random_sampling(instances, BATCH_SIZE)
    assert len(sampled_data.indices) == BATCH_SIZE
    assert sampled_data.instances[0]["document_id"] == "bc/cctv/00/cctv_0000"
