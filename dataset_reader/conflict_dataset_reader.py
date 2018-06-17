import os
import random
import time

import ujson
from tqdm import tqdm

from dataset_reader.data import Vocabulary


class ConflictDatasetReader(object):
    def __init__(self,
                 vocab: Vocabulary,
                 batch_size=64,
                 bptt_limit=30,
                 exclude_sections=None):

        self.batch_size = batch_size
        self.bptt_limit = bptt_limit
        self.vocabulary = vocab
        self.exclude_sections = exclude_sections or ["References", "Bibliography", "See also"]
        self.examples = []

        # It's too expensive for each example to have it's own copy of the term
        # frequency for the document they belong to.
        self.id_to_encoding = {}
        self.id_to_term_frequency = {}
        self.id_to_title = {}

    # TODO: Map examples to titles, and titles to term-frequency vectors.
    def _read(self, file_path, id):
        """
        Given a Conflict Wikipedia JSON, produces a training example.
        Expected format:
        {  sections: [ { heading: string, text: string }   }
        Returns:
            A List of {  title: string, term_frequency: tensor, index: int   }
            Updates the vocabulary count vector.
        """
        parsed_document = ujson.load(open(file_path, 'r'))
        text = parsed_document["text"]
        title = parsed_document["title"]
        encoding = self.vocabulary.encode_from_text(text)
        if title is None or encoding is None:
            return []

        self.id_to_title[id] = title
        self.id_to_encoding[id] = encoding

        term_frequency = self.vocabulary.compute_term_frequencies(encoding)
        self.id_to_term_frequency[id] = term_frequency

        examples = []
        for i in range(len(encoding) - self.bptt_limit - 1):
            examples.append({
                "id": id,
                "index": i,
                "target": i + 1,
                "length": self.bptt_limit
            })
        return examples

    def load(self, data_path):
        """
        Loads each document into this dataset reader.

        :param data_path: list(JSON)
            A list of documents in which to read and collect examples from.
        :return: A list of objects containing
            titles: A list of titles in the order they appear from top to bottom
                in the batch
            portion: A bptt_limit-length tensor
            term_frequencies: A list containing frequencies for each
                tensor in 'sequence_tensors', (batch x stopless_vocab_dim) each
            index: The place in the document in which the portion is found
        """
        file_paths = os.listdir(data_path)
        absolute_paths = [os.path.join(data_path, document)
                          for document in file_paths]

        for i, document in tqdm(enumerate(absolute_paths[0:10])):
            self.examples += self._read(document, i)

    def data_loader(self, shuffle=True):
        """
        Returns a generator for batching of a single epoch.

        The final batch may be have fewer than 'batch_size' documents. Such
        batches will be discarded.

        Batches returned will consist of {
            "title": str
            "id": int
            "input": tensor
            "target": tensor
            "term_frequency": tensor
        }
        """
        examples = self.examples
        if shuffle:
            examples = self.examples.copy()
            random.shuffle(examples)

        rounded_by_batch = (len(examples) // self.batch_size) * self.batch_size
        for i in range(0, rounded_by_batch - self.batch_size, self.batch_size):
            sample = examples[i:i + self.batch_size]

            # Perform mappings to term frequency and titles to complete the batch.
            batch = []
            start = time.clock()
            for ex in sample:
                example_id = ex["id"]
                input_index = ex["index"]
                target_index = ex["target"]
                title = self.id_to_title[example_id]
                input = self.id_to_encoding[example_id][input_index: input_index + self.bptt_limit]
                target = self.id_to_encoding[example_id][target_index: target_index+ self.bptt_limit]
                term_frequency = self.id_to_term_frequency[example_id]
                batch.append({
                    "title": title,
                    "input": input,
                    "target": target,
                    "term_frequency": term_frequency
                })
            print("\nTime yielding batch:", time.clock() - start, "\n")
            yield batch
