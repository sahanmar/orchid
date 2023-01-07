from coref.const import Doc
from active_learning.acquisition_functions import random_sampling

BATCH_SIZE = 7
TOO_LARGE_BATCH_SIZE = 11


def test_random_sampling(dev_data: list[Doc]) -> None:
    instances = 10 * dev_data  # we have only one debug doc, so make 10
    sampled_data = random_sampling(instances, BATCH_SIZE)
    assert len(sampled_data.indices) == BATCH_SIZE
    assert sampled_data.instances[0]["document_id"] == "bc/cctv/00/cctv_0000"


def test_random_sampling_w_too_large_batch_size(dev_data: list[Doc]) -> None:
    sampled_data = random_sampling(dev_data, TOO_LARGE_BATCH_SIZE)
    assert len(sampled_data.indices) == len(dev_data)
    assert sampled_data.instances[0]["document_id"] == "bc/cctv/00/cctv_0000"
