import torch

PAD = "<PAD>"
UNKNOWN = "<UNKNOWN>"


class Dictionary(object):
    def __init__(self):
        self.word_to_index = {PAD: 0, UNKNOWN: 1}
        self.index_to_word = [PAD, UNKNOWN]
        self.padding_index = 0
        self.unknown_index = 1

    def add_word(self, word):
        if word not in self.word_to_index:
            self.index_to_word.append(word)
            self.word_to_index[word] = len(self.index_to_word) - 1

        return self.word_to_index[word]

    def __len__(self):
        return len(self.index_to_word)


class Vocabulary(object):
    """
    Vocabulary class with sequence encoding functionality.

    Given a vocabulary and a list of stop words, creates two
    word spaces: a full vocabulary and a stopless vocabulary.
    """
    def __init__(self,
                 vocabulary: set,
                 frequencies: dict,
                 stops: set):
        self.stop_size = len(stops)

        # Not all stops are in the vocabulary! Add them for less hassle later.
        full = vocabulary.union(stops)
        self.vocabulary = {
            "full": Dictionary(),
            "stopless": Dictionary()
        }

        for word in full:
            self.vocabulary["full"].add_word(word)

        for word in vocabulary:
            if word not in stops:
                self.vocabulary["stopless"].add_word(word)

        # Sanity checking
        for word in full:
            assert(word in self.vocabulary["full"].word_to_index)
        for word in stops:
            assert(word not in self.vocabulary["stopless"].word_to_index)

        self.vocab_size = len(self.vocabulary["full"])
        self.stopless_vocab_size = len(self.vocabulary["stopless"])

        # Compute the global background frequency under this vocabulary.
        self.background_frequency = torch.zeros(self.vocab_size)
        for term, frequency in frequencies.items():
            term_index = self.get_index(term)
            self.background_frequency[term_index] = frequency

    def compute_term_frequencies(self, seq_tensor):
        """
        Computes the term-frequency vector of 'seq_tensor' in the
        stopless space.

        Expects that the sequence's first dimension is the length of words.
        """
        frequencies = torch.zeros(self.stopless_vocab_size).float()
        for word_index in seq_tensor.long():
            # Convert the word into a stopless space vector.
            # If the query to the stopless namespace is None, the word is a
            # stop and should not have a frequency.
            word = self.get_stopless_word(word_index)
            if word is not UNKNOWN:
                word_stopless_index = self.get_stopless_index(word)
                frequencies[word_stopless_index] += 1

        return frequencies / torch.sum(frequencies)

    def encode_from_text(self, text):
        """
        Given a string of text, returns its encoded representation
        given this vocabulary.

        Assumes text is already tokenized and separated by whitespace.
        """
        words = [word.lower() for word in text.split()]

        # Some sections may be empty; return None in this case.
        if len(words) == 0:
            return None

        # Construct a sequence tensor for the text.
        ids = torch.Tensor(len(words)).long()
        for i, word in enumerate(words):
            ids[i] = self.get_index(word)

        return ids

    def text_from_encoding(self, sequence_tensor, remove_padding=True):
        """
        Given an encoded representation of a string, return a list of
        the words given this vocabulary.

        Assumes padding only ever appears toward the end of a sequence.
        """
        res = []
        for i in sequence_tensor:
            res.append(self.vocabulary["full"].index_to_word[i])

        pad_idx = self.vocabulary["full"].padding_index = 0
        if remove_padding:
            res = [word for word in res
                   if self.vocabulary["full"].word_to_index[word] != pad_idx]

        return res

    def get_word(self, index):
        if index < len(self.vocabulary["full"].index_to_word):
            return self.vocabulary["full"].index_to_word[index]
        return UNKNOWN

    def get_index(self, word):
        if word in self.vocabulary["full"].index_to_word:
            return self.vocabulary["full"].word_to_index[word]
        return 1

    def get_stopless_word(self, index):
        if index < len(self.vocabulary["stopless"].index_to_word):
            return self.vocabulary["stopless"].index_to_word[index]
        return UNKNOWN

    def get_stopless_index(self, word):
        if word in self.vocabulary["stopless"].index_to_word:
            return self.vocabulary["stopless"].word_to_index[word]
        return 1
