import json
import logging
from logging import Logger, LogRecord
from typing import Any, Optional, Dict, cast, Union

from config.logging import Logging


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.

    @param fmt_dict: dict;
        key-value logging format attribute pairs.
        Defaults to {"message": "message"}.
    @param time_format: str; time.strftime() format string. Default: "%Y-%m-%dT%H:%M:%S"
    """

    def __init__(
        self,
        fmt_dict: Optional[Dict[str, str]] = None,
        time_format: str = "%Y-%m-%dT%H:%M:%S",
    ) -> None:
        super().__init__(
            # Filler values
            fmt=None,
            datefmt=None,
            style="%",
            validate=False,
        )
        self.fmt_dict = (
            fmt_dict if fmt_dict is not None else {"message": "message"}
        )
        self.default_time_format = time_format
        self.default_msec_format = ""

    def usesTime(self) -> bool:
        """
        Overwritten to look for the attribute in the format dict values instead of the fmt string.
        """
        return "asctime" in self.fmt_dict.values()

    def formatMessage(self, record: LogRecord) -> Dict[str, str]:  # type: ignore[override]
        """
        Overwritten to return a dictionary of the relevant LogRecord attributes instead of a string.
        KeyError is raised if an unknown attribute is provided in the fmt_dict.
        """
        return {
            fmt_key: record.__dict__[fmt_val]
            for fmt_key, fmt_val in self.fmt_dict.items()
        }

    def format(self, record: LogRecord) -> str:
        """
        Mostly the same as the parent's class method,
        the difference being that a dict is manipulated and dumped as JSON
        instead of a string.
        """
        msg_raw = record.msg
        if isinstance(msg_raw, str):
            message = record.getMessage()
        else:
            # Does not supply user-supplied arguments
            # The below statement is reachable, e.g., in cases when a dictionary
            # is passed to the log message
            message = msg_raw  # type: ignore[unreachable]
        record.message = message

        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        message_dict = self.formatMessage(record)

        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            message_dict["exc_info"] = record.exc_text

        if record.stack_info:
            message_dict["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(message_dict, default=str)


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

    log_file_unformatted = logging_conf.log_folder.joinpath(
        f"{experiment}_{logging_conf.timestamp}.log"
    )

    # Define handlers
    # Standard file handler
    file_handler_unformatted = logging.FileHandler(log_file_unformatted)
    logger.addHandler(file_handler_unformatted)

    # Jsonlines file handler
    file_handler_jsonl = logging.FileHandler(
        log_file_unformatted.with_suffix(".jsonl"),
        mode="a",
    )
    formatter_jsonl = JsonFormatter(
        logging_conf.jsonl_format,
        time_format=logging_conf.datetime_format,
    )
    file_handler_jsonl.setFormatter(formatter_jsonl)
    logger.addHandler(file_handler_jsonl)

    # Formatted stream handler
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
