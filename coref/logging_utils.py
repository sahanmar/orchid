import logging
from logging import Logger
from typing import Any

from config.logging import LoggingConfig


def get_stream_logger(
    name: str,
    **stream_handler_kw: Any,
) -> Logger:

    logging_conf = LoggingConfig()

    logger = logging.getLogger(name)
    logger.setLevel(logging_conf.verbosity.value)
    logger.handlers.clear()
    logger.propagate = False

    # Define handlers
    logger.addHandler(logging.FileHandler(logging_conf.file))
    sh = logging.StreamHandler(**stream_handler_kw)
    assert isinstance(logging_conf.stream_format, str) and len(
        logging_conf.stream_format,
    ), f"Invalid stream_format: {logging_conf.stream_format}"
    formatter = logging.Formatter(
        logging_conf.stream_format,
        datefmt=logging_conf.datetime_format,
        style="%",
    )
    sh.setFormatter(formatter)

    # Add stream handler to the logger
    logger.addHandler(sh)
    return logger
