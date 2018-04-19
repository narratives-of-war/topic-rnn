

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax
import torch.nn as nn


class TopicRNN(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, stop_size,
                 batch_size, layers=1, dropout=0.5, vae_hidden_size=128,
                 topic_dim=50):

        """
        RNN Language Model with latently learned topics for capturing global
        semantics: https://arxiv.org/abs/1611.01702

        When predicting each word, uses likelihood of it being a stopword to
        throttle the inclusion of topics in inference.

        Definitions of Constants
        ------------------------
        C - Vocabulary size including stopwords
        H - RNN hidden size
        K - Number of topic dimensions
        E - Normal Distribution parameters inference network's hidden dimension

        Model Parameters and Dimensions
        -------------------------------
        U := Projects xt into hidden space.
        W := RNN f_w weights for calculating h_t (H x H)
        V := RNN weights for inference on y_t (H x C)
        beta := Topic distributions over words (row-major) (K x C)
        theta := Topic vector (K)
        W_1 := Weights for affine calculating mu (E)
        W_2 := Weights for affine calculating sigma (E)

        Python/Torch Parameters:
        -----------

        :param vocab_size: int
            The size of the vocabulary.

        :param hidden_size: int
            The hidden size of the RNN.

        :param stop_size: int
            The number of stop words.

        :param batch_size: int
            Number of examples to observe per forward pass.

        :param vae_hidden_size: int
            The hidden size of the inference network that approximates
            the normal distribution for VAE.

        :param topic_dim: int
            The number of topics the model will learn.
        """

        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        super(TopicRNN, self).__init__()

        self.vocab_size = vocab_size  # V
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size  # H
        self.vae_hidden_size = vae_hidden_size  # E
        self.stop_size = stop_size
        self.batch_size = batch_size
        self.layers = layers
        self.topic_dim = topic_dim  # K

        """ TopicRNN-specific Parameters """

        # Topic proportions randomly initialized (uniform dist).
        topic_proportions = torch.rand(topic_dim)
        topic_proportions /= torch.sum(topic_proportions)
        self.theta = nn.Parameter(topic_proportions)

        # Topic distributions over words
        self.beta = nn.Parameter(torch.rand(topic_dim, vocab_size))

        # Parameters for the VAE that approximates a normal dist.
        self.g = G(vocab_size - stop_size, vae_hidden_size, topic_dim)
        self.w1 = nn.Parameter(torch.rand(vae_hidden_size))
        self.w2 = nn.Parameter(torch.rand(vae_hidden_size))

        # Weight matrix to extract word proportions from hidden states.
        self.v = torch.rand(vocab_size, hidden_size)

        """ Generic RNN Parameters """

        # Learned word embeddings (vocab_size x embedding_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Elman RNN, accepts vectors of length 'embedding_size'.
        self.rnn = nn.RNN(embedding_size, hidden_size, layers,
                          dropout=dropout,
                          batch_first=True)

        # Decode from hidden state space to vocab space.
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self):
        """
        Produce a new, initialized hidden state variable where all values
        are zero.
        :return: A torch Tensor.
        """

        weight = next(self.parameters()).data
        return Variable(weight.new(self.layers, self.batch_size, self.hidden_size).zero_())

    def forward(self, input, hidden, stop_indicators):
        # Embed the passage.
        # Shape: (batch, length (single word), embedding_size)
        embedded_passage = self.embedding(input).view(self.batch_size, 1, -1)

        # Forward pass through the RNN to compute the hidden state.
        # Shape (output): (1, hidden_size)
        # Shape (hidden): (layers, batch, hidden_size)
        output, hidden = self.rnn(embedded_passage, hidden)

        # Extract word proportions (stop words included).
        proportions_included_stops = torch.matmul(self.v.T, hidden)
        proportions_no_stops = torch.matmul((1 - stop_indicators),
                                            torch.matmul(self.beta.T, self.theta))

        return softmax(proportions_included_stops + proportions_no_stops), hidden


class G(nn.Module):
    """
    The feedforward network that projects term-frequencies into
    K-dimensional latent space.

    Used for calculating mu and sigma for the approximated normal.

    Parameters:
    -----------
    :param vc_dim: int
        The size of the vocabulary excluding stop words.

    :param hidden_size: int
        The hidden size of the inference network.

    :param topic_dim: int
        The latent space in which to project term-frequencies onto.
    """
    def __init__(self, vc_dim, hidden_size, topic_dim):
        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        super(G, self).__init__()

        self.vc_dim = vc_dim
        self.hidden_size = hidden_size
        self.topic_dim = topic_dim

        self.model = nn.Sequential(
            nn.Linear(vc_dim, hidden_size * topic_dim),
            nn.ReLU(),
            nn.Linear(hidden_size * topic_dim, hidden_size * topic_dim),
            nn.ReLU()
        )

    def forward(self, term_frequencies):
        output = self.model(term_frequencies)

        # Reshape to (E x K) space for calculation of mu and sigma.
        return output.view(self.hidden_size, self.topic_dim)
