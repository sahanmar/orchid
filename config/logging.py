import time
from dataclasses import dataclass
from enum import Enum
from logging import (
    ERROR,
    WARNING,
    INFO,
    DEBUG,
)
from pathlib import Path

from config.config_utils import overwrite_config


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
    jsonl_format: dict[str, str]
    datetime_format: str
    log_folder: Path
    timestamp: int

    @staticmethod
    @overwrite_config
    def from_config(
        logger_name: str,
        verbosity: str,
        stream_format: str,
        jsonl_format: dict[str, str],
        datetime_format: str,
        log_folder: str,
    ) -> "Logging":
        return Logging(
            # The code has to fail if the config is bad
            logger_name=logger_name,
            verbosity=LogVerbosityMapping[verbosity],
            stream_format=stream_format,
            jsonl_format=jsonl_format,
            datetime_format=datetime_format,
            log_folder=Path(log_folder),
            timestamp=int(time.time()),
        )
