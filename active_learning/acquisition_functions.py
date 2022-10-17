from typing import List
from random import sample

from coref.const import Doc, SampledData


def random_sampling(instances: List[Doc], batch_size: int) -> SampledData:
    """
    Provides random sampling from given instances
    """
    indices = sample(list(range(0, len(instances))), batch_size)
    return SampledData(indices, [instances[i] for i in indices])
