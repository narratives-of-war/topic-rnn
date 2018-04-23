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

    def __init__(self, vocabulary, stops):
        self.dictionary = Dictionary()
        self.documents = []
        self.vocabulary = vocabulary
        self.stop_encodings = set()

        for word in vocabulary:
            self.dictionary.add_word(word)

        # Populate the set containing stopword encodings.
        # What the actual words are doesn't matter as long as we can
        # lookup whether a part of a seq tensor is a stop or not!
        for word in stops:
            self.stop_encodings.add(self.dictionary.word_to_index[word])

        # For the neural network that encodes word frequencies.
        vocabulary_no_stops = vocabulary - stops
        self.dictionary_no_stops = Dictionary()
        self.vocabulary_no_stops = vocabulary_no_stops
        self.vocab_size_no_stops = len(vocabulary_no_stops)

        # STOPLESS PARK! STOPLESS PARK!
        for word in self.vocabulary_no_stops:
            self.dictionary_no_stops.add_word(word)

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

    def compute_term_frequencies(self, seq_tensor):
        """
        Computes the term-frequency vector of 'seq_tensor' in the
        stop-less space.
        :param seq_tensor: A LongTensor containing word vectors.
        :return: A new vector in stopless space containing the counts of
                the words normalized by the size of the seq_tensor.
        """
        frequencies = torch.FloatTensor(self.vocab_size_no_stops)
        normalizer = torch.sum(seq_tensor)
        for word in seq_tensor.long():  # Precaution: Only long
            frequencies[word] += 1 / normalizer

        return frequencies

    def get_stop_indicators(self, seq_tensor):
        """
        Computes a boolean vector where 1 means the word in 'seq_tensor'
        in that place is a stopword and 0 means otherwise.
        """
        seq_length = seq_tensor.size(0)
        stop_indicators = torch.zeros(seq_length).long()
        for i, word in enumerate(seq_tensor):  # Precaution: Only long
            word_as_str = self.dictionary.index_to_word[word]
            if word_as_str in self.dictionary_no_stops.word_to_index:
                stop_indicators[i] = 1

        return stop_indicators

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
