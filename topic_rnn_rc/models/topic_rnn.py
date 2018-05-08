import torch
from torch.autograd import Variable
from torch.nn.functional import log_softmax, softmax
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
        self.embedding_size = embedding_size  # TODO: Pre-trained or not?
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
        self.theta = topic_proportions

        # Topic distributions over words
        self.beta = nn.Parameter(torch.rand(topic_dim, vocab_size))

        # Parameters for the VAE that approximates a normal dist.
        self.g = G(vocab_size - stop_size, vae_hidden_size, topic_dim)

        # mu
        self.w1 = nn.Parameter(torch.rand(vae_hidden_size))
        self.a1 = nn.Parameter(torch.rand(topic_dim))

        # sigma
        self.w2 = nn.Parameter(torch.rand(vae_hidden_size))
        self.a2 = nn.Parameter(torch.rand(topic_dim))

        """ Generic RNN Parameters """

        # Learned word embeddings (vocab_size x embedding_size)
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=0)

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
        return Variable(weight.new(self.layers, self.batch_size,
                                   self.hidden_size).zero_())

    def forward(self, input, hidden, stops):
        # Embed the passage.
        # Shape: (batch, length (single word), embedding_size)
        embedded_passage = self.embedding(input).view(self.batch_size, 1, -1)

        # Forward pass through the RNN to compute the hidden state.
        # Shape (output): (1, hidden_size)
        # Shape (hidden): (layers, batch, hidden_size)
        output, hidden = self.rnn(embedded_passage, hidden)

        # Extract word proportions (with and without stop words).
        # Disallow stopwords from having influence.
        with_stops = self.decoder(hidden).squeeze()
        no_stops = Variable(self.theta).matmul(self.beta)
        no_stops = Variable((stops != 0).float()).matmul(no_stops)
        return softmax(with_stops + no_stops, dim=1), hidden

    def likelihood(self, sequence_tensor, term_frequencies,
                   stop_indicators, cuda, num_samples=1):

        # TODO: stop_indicators should be (batch, max_sentence_length)

        # 1. Compute Kullback-Leibler Divergence
        mapped_term_frequencies = self.g(Variable(term_frequencies))

        # Compute Gaussian parameters
        # TODO: Swap E and (E x K)?
        mu = mapped_term_frequencies.matmul(self.w1) + self.a1
        log_sigma = mapped_term_frequencies.matmul(self.w2) + self.a2

        # A closed-form solution exists since we're assuming q
        # is drawn from a normal distribution.
        #
        # Sum along the batch dimension.
        neg_kl_div = 1 + 2 * log_sigma - (mu ** 2) - torch.exp(2 * log_sigma)
        neg_kl_div = torch.sum(neg_kl_div, 1) / 2

        # 2. Sample all words in the sequence

        def normal_noise():
            # Sample gaussian noise between steps and sample the words
            # Shape: (batch size, K)
            return torch.rand(sequence_tensor.size(0), self.topic_dim)[0]

        log_probabilities = 0
        for l in range(num_samples):
            hidden = self.init_hidden()
            for k in range(sequence_tensor.size(1) - 1):

                # TODO: Make word (batch size,)
                import pdb
                pdb.set_trace()
                word = sequence_tensor[:, k]
                epsilon = normal_noise()

                if cuda:
                    word = word.cuda()

                # Sample theta via mu + sigma (hadamard) epsilon.
                self.theta = mu.data + torch.exp(log_sigma).data * epsilon
                self.theta /= torch.sum(self.theta)  # TODO: Softmax?

                output, hidden = self.forward(Variable(word), hidden,
                                              stop_indicators[:, k])

                prediction_probabilities = log_softmax(output, 1)

                # Prevent padding from contributing.
                non_empty_words = Variable((word != 0).float().unsqueeze(1))
                prediction_probabilities *= non_empty_words

                # Index into probabilities of the actual words.
                word_index = Variable(word.unsqueeze(1))
                word_probabilities = prediction_probabilities.gather(1, word_index)

                # Update the Monte Carlo sample we have
                log_probabilities += word_probabilities.squeeze()

        # Likelihood of the sequence under the model
        # TODO: Scale KL-Div by size of block?
        # TODO: Print and figure out signs.
        return neg_kl_div + (log_probabilities / num_samples)


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

        # TODO: Make this simpler
        self.model = nn.Sequential(
            nn.Linear(vc_dim, hidden_size * topic_dim),
            nn.ReLU(),
            nn.Linear(hidden_size * topic_dim, hidden_size * topic_dim),
        )

    def forward(self, term_frequencies):
        output = self.model(term_frequencies)
        batch_size = term_frequencies.size(0)
        # Reshape to (batch size x K x E) space for calculation of mu and sigma.
        # Normalize along the topic dimension.
        return nn.Softmax(dim=1)(output.view(batch_size, self.topic_dim,
                                             self.hidden_size))
