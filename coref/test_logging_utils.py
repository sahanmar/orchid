from coref.logging_utils import get_stream_logger
from config.logging import LoggingConfig


def test_get_stream_logger() -> None:
    text = "Test file is created"
    logger = get_stream_logger(f"coref-model")
    logger.info(text)
    assert LoggingConfig.file.is_file()
    with open(LoggingConfig.file, "r") as f:
        log_text = [l.strip() for l in f.readlines()]
    assert text in log_text
    LoggingConfig.file.unlink()
    assert not LoggingConfig.file.is_file()
