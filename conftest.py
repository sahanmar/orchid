from pytest import fixture
from config import Config
from coref.const import Doc
from coref.data_utils import get_docs, DataType


@fixture(scope="session")
def config() -> Config:
    return Config.load_default_config(section="debug")


@fixture(scope="session")
def dev_data(config: Config) -> list[Doc]:
    return get_docs(DataType.test, config)
