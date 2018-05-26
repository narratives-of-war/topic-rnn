import json

from nltk import word_tokenize


def extract_tokens_from_conflict_json(path):
    """
    Tokenizes a Conflict JSON Wikipedia article and returns a list
    of its tokens..

    If a file does not have "title" and "sections" field, this
    function returns an empty list.
    :param path: The path to a training document.
    """
    parsed_document = json.load(open(path, 'r'))

    if "title" not in parsed_document or "sections" not in parsed_document:
        return

    # Collect the content sections.
    sections = parsed_document["sections"]

    # Collect tokens from every section of the paper except for references.
    exclude = ["References"]
    document_raw = ' '.join([section["text"] for section in sections
                             if "heading" in section and
                             section["heading"] not in exclude])

    return word_tokenize(document_raw)
