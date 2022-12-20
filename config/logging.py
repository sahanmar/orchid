from dataclasses import dataclass
from logging import (
    ERROR,
    WARNING,
    INFO,
    DEBUG,
)

from enum import Enum
from pathlib import Path


class LogVerbosityMapping(Enum):
    error = ERROR
    warning = WARNING
    info = INFO
    debug = DEBUG


class LoggingConfig:
    verbosity: LogVerbosityMapping = LogVerbosityMapping.debug
    stream_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    datetime_format: str = "%Y-%m-%dT%H:%M:%S%z"
    file: Path = Path("coref_model_logs.log")
