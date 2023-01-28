from coref.const import Doc


def test_create_simulation_pseudodoc(dev_data: list[Doc]) -> None:
    # We can use only one doc
    # Create the pseudo doc from an existing one
    doc = dev_data[0]
    pseudo_doc = doc.create_simulation_pseudodoc()
    assert doc.document_id == pseudo_doc.document_id
    assert doc.cased_words == pseudo_doc.cased_words
    assert doc.sent_id == pseudo_doc.sent_id
    assert doc.part_id == pseudo_doc.part_id
    assert doc.speaker == pseudo_doc.speaker
    assert doc.pos == pseudo_doc.pos
    assert doc.deprel == pseudo_doc.deprel
    assert doc.head2span == pseudo_doc.head2span
    assert doc.span_clusters == pseudo_doc.span_clusters
    assert doc.word2subword == pseudo_doc.word2subword
    assert doc.subwords == pseudo_doc.subwords

    # Create a pseudo doc from
    doc.simulation_token_annotations.tokens = {
        55,
        70,
        377,
        378,
        379,
        411,
        412,
        413,
    }
    pseudo_doc = doc.create_simulation_pseudodoc()
    assert doc.document_id == pseudo_doc.document_id
    assert pseudo_doc.cased_words == [
        "Disney",
        "Disney",
        "a",
        "security",
        "guard",
        "the",
        "security",
        "guard",
    ]
    assert pseudo_doc.head2span == [[0, 0, 1], [1, 1, 2], [4, 2, 5], [7, 5, 8]]
    assert pseudo_doc.span_clusters == [[(0, 1), (1, 2)], [(2, 5), (5, 8)]]
    assert pseudo_doc.word2subword == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
    ]
