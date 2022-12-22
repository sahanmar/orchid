from dataclasses import dataclass
from logging import (
    ERROR,
    WARNING,
    INFO,
    DEBUG,
)

from enum import Enum
from pathlib import Path
from config.config_utils import overwrite_config
from typing import Any


class LogVerbosityMapping(Enum):
    error = ERROR
    warning = WARNING
    info = INFO
    debug = DEBUG


@dataclass
class Logging:
    logger_name: str
    verbosity: LogVerbosityMapping
    stream_format: str
    datetime_format: str
    log_file: Path

    @staticmethod
    @overwrite_config
    def from_config(
        logger_name: str,
        verbosity: str,
        stream_format: str,
        datetime_format: str,
        log_file: str,
    ) -> "Logging":
        return Logging(
            # The code has to fail if the config is bad
            logger_name=logger_name,
            verbosity=LogVerbosityMapping[verbosity],
            stream_format=stream_format,
            datetime_format=datetime_format,
            log_file=Path(log_file),
        )
