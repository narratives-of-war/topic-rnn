from collections import Counter
import dill
import os
import re
import sys

import ujson
import torch
from torch.nn.functional import normalize
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def create_embeddings_from_vocab(vocab, glove_path):
    """
    Given a vocabulary and a path to a glove file, construct the embedding
    matrix.

    Assigns words that do not appear in glove with a random vector.
    """
    print("Reading glove embeddings...")
    with open(glove_path, 'r') as f:
        glove_entries = f.readlines()

    embedding_dimension = len(glove_entries[0].split()) - 1

    def map_word_to_vector(entry):
        entry = entry.split()
        return entry[0], torch.Tensor([float(vi) for vi in entry[1:]])

    word_to_vectors = dict(map_word_to_vector(entry) for entry in glove_entries)

    # Ensure entries not from glove are normalized.
    embedding_weights = normalize(torch.rand(vocab.vocab_size, embedding_dimension),
                                  p=1, dim=-1)

    print("Constructing embedding matrix:")
    existing_mappings = vocab.vocabulary['full'].word_to_index

    contained = 0
    for word, index in tqdm(existing_mappings.items()):
        if word in word_to_vectors:
            vector = word_to_vectors[word]
            embedding_weights[index] = vector
            contained += 1

    print(contained, "words in the vocabulary had an embedding!")
    return embedding_weights, embedding_dimension


def sieve_vocabulary(training_path, belligerents_path, min_token_count,
                     normalize_vocabulary=True):
    print("Loading documents...")
    training_files = os.listdir(training_path)
    tokens = []
    for file in tqdm(training_files):
        file_path = os.path.join(training_path, file)
        with open(file_path, 'r') as f:
            file_json = ujson.load(f)
            tokens += file_json["text"].split()

    # Map words to the number of times they occur in the corpus.
    word_frequencies = dict(Counter(tokens))

    # Sieve the initial vocabulary by excluding all words that appear
    # fewer than min_token_count times.
    vocab = set([w.lower() if normalize_vocabulary else w
                 for w, f in word_frequencies.items()
                 if f >= min_token_count])

    if belligerents_path:
        print("Loading belligerents...")
        belligerents_files = os.listdir(belligerents_path)
        for file in tqdm(belligerents_files):
            file_path = os.path.join(belligerents_path, file)
            with open(file_path, 'r') as f:
                belligerents_tokens = word_tokenize(f.read())

            # Keep only words that are alphabetical.
            belligerents_tokens = [token.lower() if normalize_vocabulary else token
                                   for token in belligerents_tokens
                                   if re.match(r'^[a-zA-Z]+$', token)]

            tokens += belligerents_tokens
            vocab.update(belligerents_tokens)

    # Return the final counts of all tokens kept.
    tokens = [token for token in tokens if token in vocab]
    final_frequencies = dict(Counter(tokens))

    return vocab, final_frequencies


def preserve_pickle(obj, out):
    with open(out, 'wb') as f:
        dill.dump(obj, f, protocol=dill.HIGHEST_PROTOCOL)


def collect_pickle(out):
    with open(out, 'rb') as f:
        return dill.load(f)


def print_headline(headline, length=80):
    print("+" + "-" * (length - 2) + "+")
    print(" ", headline)
    print("+" + "-" * (length - 2) + "+")


def print_progress_in_place(*args):
    print("\r", *args, end="")
    sys.stdout.flush()
