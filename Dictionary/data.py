import random

import en_core_web_sm
import json
from nltk import word_tokenize

import torch

PAD = "<PAD>"
UNKNOWN = "<UNKNOWN>"


class Dictionary(object):
    def __init__(self):
        self.word_to_index = {UNKNOWN: 0, PAD: 1}
        self.index_to_word = [UNKNOWN, PAD]

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

    Use 'add_document' to both update the vocabulary as well as produce
    sequence tensors for the document passed.
    """

    def __init__(self, vocabulary, stops):
        self.dictionary = Dictionary()
        self.documents = []
        self.nlp = en_core_web_sm.load()
        self.stop_encodings = set()
        self.stop_size = len(stops)

        # Not all stops are in the vocabulary! Add them for less hassle later.
        vocabulary = vocabulary.union(stops)
        self.vocabulary = vocabulary
        for word in self.vocabulary:
            self.dictionary.add_word(word)

        self.vocab_size = len(self.dictionary)

        # Populate the set containing stopword encodings.
        # What the actual words are doesn't matter as long as we can
        # lookup whether a part of a seq tensor is a stop or not!
        for word in stops:
            self.stop_encodings.add(self.dictionary.word_to_index[word])

        # For the neural network that encodes word frequencies.
        vocabulary_no_stops = vocabulary - stops
        self.dictionary_no_stops = Dictionary()

        # STOPLESS PARK! STOPLESS PARK!
        for word in vocabulary_no_stops:
            self.dictionary_no_stops.add_word(word)

        self.vocab_size_no_stops = len(self.dictionary_no_stops)

    def add_document(self, path):
        """
        Tokenizes a Conflict JSON Wikipedia article and adds it's sequence
        tensor to the corpus.

        If a file being added does not have "title" and "sections" field, this
        function does nothing.
        :param path: The path to a JSON training document.
        """
        parsed_document = json.load(open(path, 'r'))

        if "title" not in parsed_document or "sections" not in parsed_document:
            return

        # Collect the publication title and content sections.
        title = parsed_document["title"]
        sections = parsed_document["sections"]

        # Vectorize every section of the paper except for references.
        exclude = ["References"]
        document_raw = ""
        for section in sections:
            # Collect only semantically significant sections.
            if "heading" in section and section["heading"] not in exclude:
                document_raw += ("\n" + section["text"])

        # Vectorize all words in the document.
        parsed_document = self.nlp(document_raw)
        encoded_sentences = []
        for s in parsed_document.sents:
            sentence = str(s).strip()
            encoded_sentence = self.tokenize_from_text(sentence)
            encoded_sentences.append(encoded_sentence)

        document_object = {
            "title": title,
            "sentences": encoded_sentences
        }
        self.documents.append(document_object)

    def compute_term_frequencies(self, seq_tensor):
        """
        Computes the term-frequency vector of 'seq_tensor' in the
        stopless space.
        :param seq_tensor: A LongTensor containing word vectors.
        :return: A new vector in stopless space containing the counts of
                the words normalized by the size of the seq_tensor.
        """
        frequencies = torch.zeros(self.vocab_size_no_stops).float()
        normalizer = seq_tensor.size(0)
        for word_as_idx in seq_tensor.long():
            # Convert the word into a stopless space vector
            word_as_str = self.dictionary.index_to_word[word_as_idx]
            if word_as_str in self.dictionary_no_stops.word_to_index:
                word_as_idx_stopless = self.dictionary_no_stops.word_to_index[word_as_str]
                frequencies[word_as_idx_stopless] += (1 / normalizer)

            # Do nothing for stop words!

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


class ConflictLoader(object):
    def __init__(self, corpus, random_seed=1):
        self.corpus = corpus

        # Separate train/dev via an 80:20 split.
        # Use the provided seed for reproducibility.
        num_docs = len(self.corpus.documents)
        docs = random.Random(random_seed).sample(corpus.documents, num_docs)

        self.training = docs[0:int(num_docs * 0.8)]
        self.development = docs[int(num_docs * 0.8):]

    @staticmethod
    def data_loader(batch_size, examples):
        """
        Returns a generator for 'examples'.

        The final batch may be have fewer than 'batch_size' documents.
        It is up to the user to decide whether to salvage or discard this batch.

        :param batch_size: int
            Partitions between data.
        :param examples: list(JSON)
            A list of documents in which to partition.
        :return: A generator that produces 'batch_size' documents at a time.
        """

        for i in range(0, len(examples), batch_size):
            yield examples[i:i + batch_size]

    def training_loader(self, batch_size):
        return self.data_loader(batch_size, self.training)

    def validation_loader(self, batch_size):
        return self.data_loader(batch_size, self.training)

    def test_loader(self, batch_size, test_data):
        return self.data_loader(batch_size, test_data)