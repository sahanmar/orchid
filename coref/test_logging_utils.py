from coref.logging_utils import get_stream_logger
from config.logging import Logging
from config import Config


def test_get_stream_logger(config: Config) -> None:
    text = "Test file is created"
    logger = get_stream_logger(f"test-logging", logging_conf=config.logging)
    logger.info(text)
    assert config.logging.log_file.is_file()
    with open(config.logging.log_file, "r") as f:
        log_text = [l.strip() for l in f.readlines()]
    assert text in log_text
    config.logging.log_file.unlink()
    assert not config.logging.log_file.is_file()
