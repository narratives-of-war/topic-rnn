
import torch
from torch.autograd import Variable
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, batch_size,
                 layers=1,
                 dropout=0.5):

        """
        RNN Language model: Choose between Elman, LSTM, and GRU
        RNN architectures.

        Expects single

        Parameters:
        -----------
        :param embedding_size: int
            The embedding size for embedding input words (space in which
            words are projected).

        :param hidden_size: int
            The hidden size of the RNN
        """
        # Save the construction arguments, useful for serialization
        self.init_arguments = locals()
        self.init_arguments.pop("self")
        self.init_arguments.pop("__class__")
        super(LSTM, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.layers = layers

        # Learned word embeddings (vocab_size x embedding_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # LSTM RNN, accepts vectors of length 'embedding_size'.
        self.lstm = nn.LSTM(embedding_size, hidden_size, layers,
                            dropout=dropout,
                            batch_first=True)

        # Decode from hidden state space to vocab space.
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def init_hidden(self):
        """
        Produce new, initialized cell state and hidden state variables
        where all values are zero.
        :return: A tuple of torch Tensors; cell and hidden states.
        """

        weight = next(self.parameters()).data
        return (Variable(weight.new(self.layers, self.batch_size, self.hidden_size).zero_()),
                Variable(weight.new(self.layers, self.batch_size, self.hidden_size).zero_()))

    def forward(self, input, hidden):

        # Embed the passage.
        # Shape: (batch, length (single word), embedding_size)
        embedded_passage = self.embedding(input).view(self.batch_size, 1, -1)

        # Forward pass.
        # Shape (output): (1, hidden_size)
        # Shape (hidden): 2x (layers, batch, hidden_size)
        # Expects a tuple of cell / hidden state pairs.
        output, hidden = self.lstm(embedded_passage, hidden)

        # Decode the final hidden state
        # Shape: (1, 1)
        decoded = self.decoder(output)

        return decoded, hidden
