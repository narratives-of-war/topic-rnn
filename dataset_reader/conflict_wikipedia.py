import ujson
import os

import torch

from dataset_reader.data import Vocabulary


class ConflictWikipediaDatasetReader(object):
    def __init__(self,
                 vocab: Vocabulary,
                 batch_size=64,
                 bptt_limit=30,
                 exclude_sections=None):

        self.batch_size = batch_size
        self.bptt_limit = bptt_limit
        self.vocabulary = vocab
        self.exclude_sections = exclude_sections or ["References", "Bibliography", "See also"]

    def _read(self, file_path):
        """
        Given a Conflict Wikipedia JSON, produces a training example.
        Expected format:
        {  sections: [ { heading: string, text: string }   }
        Returns:
        {   title: string, encoding: tensor, length: int   }
        """
        parsed_document = ujson.load(open(file_path, 'r'))
        text = parsed_document["text"]
        document_encoding = self.vocabulary.encode_from_text(text)
        return {
            "title": parsed_document["title"],
            "encoding": document_encoding,
            "length": document_encoding.size(0)
        }

    def data_loader(self, training_path, document_basis=True):
        """
        Returns a generator for batching across file paths.

        The final batch may be have fewer than 'batch_size' documents. Such
        batches will be discarded.

        :param training_path: list(JSON)
            A list of documents in which to partition.
        :return: A batch of objects containing
            titles: A list of titles in the order they appear from top to bottom
                in the batch
            sequence_tensors: A list of (batch size x bptt_limit) tensors
            term_frequencies: A list containing frequencies for each
                tensor in 'sequence_tensors', (batch x stopless_vocab_dim) each
            stop_indicators: A list containing stop-word indicators for each
                tensor in 'sequence_tensors' with the same dimensions,
                1 everywhere the word is a stop and 0 elsewhere.
        """
        file_paths = os.listdir(training_path)
        absolute_paths = [os.path.join(training_path, training_file)
                          for training_file in file_paths]
        if document_basis:
            for path in absolute_paths:
                yield self.configure_document(path, self.batch_size)
        else:
            for i in range(0, len(file_paths), self.batch_size):
                batch_full_paths = absolute_paths[i:i + self.batch_size]
                yield self.configure_batch(batch_full_paths)

    def configure_batch(self, file_paths):
        examples = []
        for file in file_paths:
            instance = self._read(file)
            if instance is not None:
                examples.append(instance)

        # 0. Sort and pad the instances.
        examples = sorted(examples, key=lambda x: x["length"], reverse=True)
        max_document_length = examples[0]["encoding"].size(0)

        # 1. Populate sequence tensor
        sequence_tensor = torch.zeros(len(examples), max_document_length).long()
        for i, ex in enumerate(examples):
            encoded_document = ex["encoding"]
            sequence_tensor[i][:encoded_document.size(0)] = encoded_document

        # 2. Split into backpropagation-through-time-limit portions
        bptt_portions = list(torch.split(sequence_tensor, self.bptt_limit, 1))
        if bptt_portions[-1].size(1) < self.bptt_limit:
            bptt_portions = bptt_portions[:-1]

        # 3. Compute term frequencies.
        term_frequencies = [None]
        for portion in bptt_portions[:-1]:
            frequencies = torch.zeros(self.batch_size,
                                      self.vocabulary.stopless_vocab_size)
            for i, sequence in enumerate(portion):
                frequencies[i] = self.vocabulary.compute_term_frequencies(sequence)

        # 4. Compute stopword indicators.
        stop_indicators = []
        for portion in bptt_portions:
            indicators = portion.clone()
            for i, sequence in enumerate(portion):
                indicators[i] = self.vocabulary.get_stop_indicators_from_tensor(sequence)

        return {
            "titles": [ex["title"] for ex in examples],
            "sequence_tensors": bptt_portions,
            "term_frequencies": term_frequencies,
            "stop_indicators": stop_indicators,
            "length": len(bptt_portions)
        }

    def configure_document(self, file, batch_size):
        example = self._read(file)

        # 0. Populate sequence tensor (cleave off remainder first)
        batch_rounded = (example["length"] // batch_size) * batch_size
        sequence_tensor = example["encoding"][:batch_rounded].view(batch_size, -1)

        # 2. Split into backpropagation-through-time-limit portions
        bptt_portions = list(torch.split(sequence_tensor, self.bptt_limit, 1))
        if bptt_portions[-1].size(1) < self.bptt_limit:
            bptt_portions = bptt_portions[:-1]

        # Reattach for RNN training
        sequence_tensor = torch.cat(bptt_portions, dim=1)

        # 3. Compute term frequencies.
        term_frequencies = [torch.rand(self.vocabulary.stopless_vocab_size)]
        for portion in bptt_portions[:-1]:
            frequencies = torch.zeros(self.vocabulary.stopless_vocab_size)
            for i, sequence in enumerate(portion):
                frequencies += self.vocabulary.compute_term_frequencies(sequence)

            term_frequencies.append(frequencies)

        # 4. Compute stopword indicators.
        stop_indicators = []
        for portion in bptt_portions:
            indicators = torch.zeros(portion.size())
            for i, sequence in enumerate(portion):
                indicators[i] = self.vocabulary.get_stop_indicators_from_tensor(sequence)

            stop_indicators.append(indicators)

        return {
            "title": example["title"],
            "sequence_tensors": sequence_tensor,
            "term_frequencies": term_frequencies,
            "stop_indicators": stop_indicators,
            "length": len(bptt_portions)
        }
