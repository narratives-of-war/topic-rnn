
import torch


def word_vector_from_seq(sequence_tensor, i):
    """
    Collect the vector representation of the ith word in a sequence tensor.
    :param sequence_tensor: The document in which to collect from.
    :param i: The ith index of the document.
    :return: A `torch.LongTensor()` containing the document's ith word's
        encoding relative to this corpus.
    """

    word = torch.LongTensor(1)
    word[0] = sequence_tensor[i]
    return word
