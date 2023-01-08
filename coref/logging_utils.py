import logging
from logging import Logger
from typing import Any

from config.logging import Logging


def get_stream_logger(
    logging_conf: Logging,
    experiment: str,
    **stream_handler_kw: Any,
) -> Logger:

    logger = logging.getLogger(logging_conf.logger_name)
    logger.setLevel(logging_conf.verbosity.value)
    logger.handlers.clear()
    logger.propagate = False

    if not logging_conf.log_folder.parent.is_dir():
        raise FileNotFoundError(
            "The data folder does not exist. Are you sure you are in the root dir?..."
        )
    if not logging_conf.log_folder.is_dir():
        logging_conf.log_folder.mkdir()

    log_file = (
        logging_conf.log_folder / f"{experiment}_{logging_conf.timestamp}"
    )

    # Define handlers
    logger.addHandler(logging.FileHandler(log_file))
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
