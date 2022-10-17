"""
This file defines functions used to modify the default behavior
of transformers.AutoTokenizer. These changes are necessary, because some
tokenizers are meant to be used with raw text, while the OntoNotes documents
have already been split into words.
"""


# Filters out unwanted tokens produced by the tokenizer
TOKENIZER_FILTERS = {
    "albert-xxlarge-v2": (lambda token: token != "▁"),  # U+2581, not just "_"
    "albert-large-v2": (lambda token: token != "▁"),
}

# Maps some words to tokens directly, without a tokenizer
TOKENIZER_MAPS = {
    "roberta-large": {
        ".": ["."],
        ",": [","],
        "!": ["!"],
        "?": ["?"],
        ":": [":"],
        ";": [";"],
        "'s": ["'s"],
    }
}
