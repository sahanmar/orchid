from coref.config import Config
from coref.data_utils import get_docs, DataType


def test_get_doc():
    config = Config.load_default_config(section="debug")
    data = get_docs(DataType.test, config)
    assert len(data) == 1
    assert data[0]["document_id"] == "bc/cctv/00/cctv_0000"
