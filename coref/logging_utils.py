import logging
import warnings
from logging import Logger
from typing import Optional, Any

from config.logging import LoggingConfig


def get_logging_level(verbosity: int = LoggingConfig.verbosity) -> int:
    try:
        verbosity_mapped: int = LoggingConfig.verbosity_mapping[verbosity]
        return verbosity_mapped
    except (KeyError, NameError):
        # Since these are logging utilities, we do not want to be dependent
        #   on logging that can happen to be not set up;
        # Using the UserWarning instead
        warnings.warn(
            f"verbosity={verbosity} could not be mapped to any logging level; "
            f"returning the DEBUG level",
        )
        return logging.DEBUG


def get_stream_logger(
    name: str,
    verbosity: int = LoggingConfig.verbosity,
    stream_format: Optional[str] = LoggingConfig.stream_format,
    datetime_format: Optional[str] = LoggingConfig.datetime_format,
    **stream_handler_kw: Any,
) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(get_logging_level(verbosity=verbosity))
    logger.handlers.clear()
    logger.propagate = False

    # Define handlers
    sh = logging.StreamHandler(**stream_handler_kw)
    assert isinstance(stream_format, str) and len(
        stream_format,
    ), f"Invalid stream_format: {stream_format}"
    formatter = logging.Formatter(
        stream_format,
        datefmt=datetime_format,
        style="%",
    )
    sh.setFormatter(formatter)

    # Add stream handler to the logger
    logger.addHandler(sh)
    return logger
