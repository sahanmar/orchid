from coref.const import Doc
from active_learning.acquisition_functions import random_sampling, span_sampling
from itertools import chain

BATCH_SIZE = 7
TOO_LARGE_BATCH_SIZE = 11


def test_random_sampling(dev_data: list[Doc]) -> None:
    instances = 10 * dev_data  # we have only one debug doc, so make 10
    sampled_data = random_sampling(instances, BATCH_SIZE)
    assert len(sampled_data.indices) == BATCH_SIZE
    assert sampled_data.instances[0].document_id == "bc/cctv/00/cctv_0000"


def test_random_sampling_w_too_large_batch_size(dev_data: list[Doc]) -> None:
    sampled_data = random_sampling(dev_data, TOO_LARGE_BATCH_SIZE)
    assert len(sampled_data.indices) == len(dev_data)
    assert sampled_data.instances[0].document_id == "bc/cctv/00/cctv_0000"


def test_span_sampling(dev_data: list[Doc]) -> None:
    assert (
        len(
            span_sampling(dev_data, BATCH_SIZE)
            .instances[0]
            .simulation_span_annotations.spans
        )
        >= BATCH_SIZE
    )

    assert len(
        span_sampling(dev_data, 23).instances[0].simulation_span_annotations.spans
    ) == len(set(chain.from_iterable(dev_data[0].span_clusters)))
