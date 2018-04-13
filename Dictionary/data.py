# Adapted from PyTorch examples:
# https://github.com/pytorch/examples/blob/master/word_language_model/data.py

from nltk import word_tokenize
import os

import torch


class Dictionary(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = []

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

    def __init__(self):
        self.dictionary = Dictionary()
        self.examples = []

    def add_example(self, path):
        """
        Tokenizes a text file and adds it's sequence tensor to the corpus.
        :param path: The path to a training document.
        """
        sequence_tensor = self.tokenize(path)
        self.examples.append(sequence_tensor)

    def tokenize(self, path):
        """
        Tokenize a text file into a sequence tensor.
        :param path: The path to a training document.
        :return A sequence tensor of the document of dimensions
            (Length of document,) s.t. the ith column is the integer
            representation of the ith word in the document.

            Indices are consistent with all other documents used for this
            corpus.
        """
        assert(os.path.exists(path))

        lines = []
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = word_tokenize(line) + ['<EOS>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

                lines.append(words)

        # Convert the document into its own sequence tensor.
        ids = torch.LongTensor(tokens)
        tokens = 0
        for line in lines:
            for word in line:
                ids[tokens] = self.dictionary.word_to_index[word]
                tokens += 1

        return ids

