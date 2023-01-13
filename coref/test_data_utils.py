from coref.const import Doc


def test_get_doc(dev_data: list[Doc]) -> None:
    assert len(dev_data) == 1
    assert dev_data[0].document_id == "bc/cctv/00/cctv_0000"
