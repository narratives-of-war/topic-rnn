# Adapted from PyTorch examples:
# https://github.com/pytorch/examples/blob/master/word_language_model/data.py

import json
from nltk import word_tokenize
import os

import torch

UNKNOWN = "<UNKNOWN>"


class Dictionary(object):
    def __init__(self):
        self.word_to_index = {UNKNOWN: 0}
        self.index_to_word = [UNKNOWN]

    def add_word(self, word):
        if word not in self.word_to_index:
            self.index_to_word.append(word)
            self.word_to_index[word] = len(self.index_to_word) - 1

        return self.word_to_index[word]

    def __len__(self):
        return len(self.index_to_word)


class Corpus(object):
    """
    Corpus class with sequence encoding functionality.

    Use 'tokenize' to both update the vocabulary as well as produce a sequence
    tensor for the document passed.
    """

    def __init__(self, vocabulary):
        self.dictionary = Dictionary()
        self.documents = []
        self.vocabulary = vocabulary

        for word in vocabulary:
            self.dictionary.add_word(word)

    def add_document(self, path):
        """
        Tokenizes a Conflict JSON Wikipedia article and adds it's sequence
        tensor to the corpus.

        If a file being added does not have "title" and "sections" field, this
        function does nothing.
        :param path: The path to a training document.
        """
        parsed_document = json.load(open(path, 'r'))

        if "title" not in parsed_document or "sections" not in parsed_document:
            return

        # Collect the publication title and content sections.
        title = parsed_document["title"]
        sections = parsed_document["sections"]

        section_tensors = []

        # Vectorize every section of the paper except for references.
        exclude = ["References"]
        for section in sections:
            if "heading" in section and section["heading"] not in exclude:
                section_tensor = self.tokenize_from_text(section["text"])

                # Handle empty section case.
                if section_tensor is not None:
                    section_tensors.append(section_tensor)

        document_object = {
            "title": title,
            "sections": section_tensors
        }
        self.documents.append(document_object)

    def tokenize_from_text(self, text):
        words = word_tokenize(text)

        # Some sections may be empty; return None in this case.
        if len(words) == 0:
            return None

        # Construct a sequence tensor for the text.
        ids = torch.LongTensor(len(words))
        for i, word in enumerate(words):
            if word in self.dictionary.word_to_index:
                ids[i] = self.dictionary.word_to_index[word]
            else:
                ids[i] = self.dictionary.word_to_index[UNKNOWN]

        return ids
