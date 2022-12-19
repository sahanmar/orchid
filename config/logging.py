from dataclasses import dataclass
from logging import (
    ERROR,
    WARNING,
    INFO,
    DEBUG,
)


@dataclass(init=False, frozen=True)
class LoggingConfig:
    verbosity_mapping = {0: ERROR, 1: WARNING, 2: INFO, 3: DEBUG}
    verbosity: int = 3
    stream_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    datetime_format: str = "%Y-%m-%dT%H:%M:%S%z"
