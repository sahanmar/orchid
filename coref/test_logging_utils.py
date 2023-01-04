from coref.logging_utils import get_stream_logger
from config import Config


def test_get_stream_logger(config: Config) -> None:
    text = "Test file is created"
    logger = get_stream_logger(
        logging_conf=config.logging, experiment="test_run"
    )
    logger.info(text)
    assert config.logging.log_folder.is_dir()
    for log_file in config.logging.log_folder.iterdir():
        with open(log_file, "r") as f:
            log_text = [l.strip() for l in f.readlines()]
        assert text in log_text
        log_file.unlink()
        assert not log_file.is_file()
