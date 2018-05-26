import argparse
import logging
import os
import dill
from tabulate import tabulate
from tqdm import tqdm
import sys


import torch
from torch.autograd import Variable

from dataset_reader import ConflictWikipediaDatasetReader, Vocabulary
from topic_rnn_rc.models.topic_rnn import TopicRNN
from topic_rnn_rc.models.rnn import RNN
from topic_rnn_rc.models.lstm import LSTM
from utils import create_embeddings_from_vocab, sieve_vocabulary, preserve_pickle, collect_pickle

sys.path.append(os.path.join(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

MODEL_TYPES = {
    "vanilla": RNN,
    "topic": TopicRNN,
    "lstm": LSTM
}


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))

    """File/Directory paths"""
    parser.add_argument("--raw-train-path", type=str,
                        help="Path to the Conflict Wikipedia JSON"
                             "training data.")
    parser.add_argument("--belligerents-path", type=str,
                        help="Path to the Conflict Wikipedia JSON"
                             "belligerents meta-data.")
    parser.add_argument("--train-path", type=str,
                        default=os.path.join(
                            project_root, "data", "train"),
                        help="Path to the Conflict Wikipedia JSON"
                             "training data (cleaned).")
    parser.add_argument("--dev-path", type=str,
                        default=os.path.join(
                            project_root, "data", "validation"),
                        help="Path to the Conflict Wikipedia JSON"
                             "dev data.")
    parser.add_argument("--test-path", type=str,
                        default=os.path.join(
                            project_root, "data", "test"),
                        help="Path to the Conflict Wikipedia JSON"
                             "dev data.")

    # Vocabulary stuff.
    parser.add_argument("--built-vocab-path", type=str,
                        default=os.path.join(
                            project_root, "vocab.pkl"),
                        help="Path to a pre-constructed vocab.")
    parser.add_argument("--stopwords-path", type=str,
                        default=os.path.join(
                            project_root, "en.txt"),
                        help="Path to the list of stop words.")
    parser.add_argument("--glove-path", type=str,
                        help="Path to file containing glove embeddings.")
    parser.add_argument("--save-dir", type=str,
                        help=("Path to save model checkpoints and logs. "
                              "Required if not using --load-path. "
                              "May not be used with --load-path."))

    """Model"""
    parser.add_argument("--model-type", type=str, default="vanilla",
                        choices=["vanilla", "topic", "lstm"],
                        help="Model type to train.")

    """Hyperparameters"""
    parser.add_argument("--min-token-count", type=int, default=10,
                        help=("Number of times a token must be observed "
                              "in order to include it in the vocabulary."))
    parser.add_argument("--clip", type=int, default=0.5,
                        help="Gradient Clipping.")
    parser.add_argument("--bptt-limit", type=int, default=30,
                        help="Extent in which the model is allowed to"
                             "backpropagate in number of words.")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Batch size to use in training and evaluation.")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden size to use in RNN and TopicRNN models.")
    parser.add_argument("--embedding-size", type=int, default=200,
                        help="Embedding size to enocde words for the RNNs.")
    parser.add_argument("--topic-dim", type=int, default=10,
                        help="Number of latent topics.")
    parser.add_argument("--num-epochs", type=int, default=25,
                        help="Number of epochs to train for.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout proportion.")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="The learning rate to use.")
    parser.add_argument("--weight-decay", type=float, default=0.33,
                        help="L2 Penalty.")
    parser.add_argument("--train-embeddings", type=str,
                        default='False',
                        help="Flag as to whether embeddings should be learned.")

    """Logistical"""
    parser.add_argument("--log-period", type=int, default=50,
                        help=("Update training metrics every "
                              "log-period weight updates."))
    parser.add_argument("--validation-period", type=int, default=500,
                        help=("Calculate metrics on validation set every "
                              "validation-period weight updates."))
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed to use")
    parser.add_argument("--cuda", action="store_true",
                        help="Train or evaluate with GPU.")
    args = parser.parse_args()

    # Set random seed for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            logger.warning("\033[35mGPU available but not running with "
                           "CUDA (use --cuda to turn on.)\033[0m")
        else:
            torch.cuda.manual_seed(args.seed)

    if not args.train_path:
        raise ValueError("Training data directory required")

    if args.model_type not in MODEL_TYPES:
        raise ValueError("Please select a supported model.")

    print("Restricting vocabulary based on min token count",
          args.min_token_count)

    print("Collecting stopwords:")
    stopwords_file = open(args.stopwords_path, 'r')
    stops = set([s.strip() for s in stopwords_file.readlines()])

    print("Constructing War Wikipedia Dataset:")
    if not os.path.exists(args.built_vocab_path):
        vocab = sieve_vocabulary(args.train_path, args.belligerents_path,
                                 args.min_token_count)
        vocabulary = Vocabulary(vocab, stops)
        preserve_pickle(vocabulary, args.built_vocab_path)
    else:
        vocabulary = collect_pickle(args.built_vocab_path)
    print("Vocabulary Size:", vocabulary.vocab_size)
    print("Stop size:", vocabulary.stop_size)
    print("Vocab size no stops:", vocabulary.stopless_vocab_size)

    embedding_weights = None
    if args.glove_path:
        embedding_weights, embedding_size = create_embeddings_from_vocab(vocabulary,
                                                                         args.glove_path)
        args.embedding_size = embedding_size

    # Create Dataset Reader.
    conflict_reader = ConflictWikipediaDatasetReader(vocabulary,
                                                     batch_size=args.batch_size,
                                                     bptt_limit=args.bptt_limit)
    # Create model of the correct type.
    print("Building {}-RNN model ---------------------------------".format(args.model_type.upper()))
    logger.info("Building {} RNN model".format(args.model_type))

    # TopicRNN Construction
    model = TopicRNN(vocabulary.vocab_size, args.embedding_size, args.hidden_size, args.batch_size,
                     topic_dim=args.topic_dim,
                     # No support for booleans in argparse atm.
                     train_embeddings=args.train_embeddings.lower() == 'true',
                     embedding_matrix=embedding_weights)

    if args.cuda:
        model.cuda()

    logger.info(model)

    print("Model Parameters -------------------------------------------")
    for name, param in model.named_parameters():
        print(name, "Trainable:", param.requires_grad)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr)
    try:
        print("Training in progress ---------------------------------------")
        for _ in range(args.num_epochs):
            data_loader = conflict_reader.data_loader(args.train_path,
                                                      document_basis=True)
            train_epoch(model, vocabulary, data_loader, args.batch_size,
                        args.bptt_limit, args.clip, optimizer,
                        args.cuda)
    except KeyboardInterrupt:
        print("Stopping training early ------------------------------------")
        pass


def train_epoch(model, vocabulary, data_loader, batch_size,
                bptt_limit, clip, optimizer, cuda):
    """
    This model trains differently than baselines; it computes the likelihood
    of a portion of text under the model instead of doing cross entropy against
    the next word.

    We maintain processing one word at a time to allow previous sections to
    have direct influence on later ones (i.e. indifferent batching will
    cause issues).
    """

    original_topics = None
    last_topics = None

    # Set model to training mode (activates dropout and other things).
    model.train()
    for i, batch in enumerate(data_loader):
        sequence_tensors = batch["sequence_tensors"]
        hidden = model.init_hidden()
        term_frequencies = None
        for k in range(sequence_tensors.size(1) - bptt_limit - 1):
            feed = sequence_tensors[:, k:k+bptt_limit].contiguous()
            target = sequence_tensors[:, k + 1:k+bptt_limit + 1].contiguous()

            # Construct stop indicators
            stop_indicators = torch.zeros(feed.size())
            for r, row in enumerate(feed):
                stop_indicators[r] = vocabulary.get_stop_indicators_from_tensor(row)

            # Optimize on negative log likelihood.
            # output, hidden = model(sequence_tensors[:, k:k+bptt_limit], hidden, None)
            # loss = criterion(output.view(output.size(0) * output.size(1), -1),
            #                  Variable(sequence_tensors[:, k + 1:k+bptt_limit + 1].contiguous().view(-1,)))

            loss, hidden = model.likelihood(feed, hidden, term_frequencies, stop_indicators, target)

            # Perform backpropagation and update parameters.
            optimizer.zero_grad()
            loss.backward()

            # Helps with exploding/vanishing gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            hidden = hidden.detach()

            # Compute term frequencies (one set delay to prevent cheating)
            term_frequencies = vocabulary.compute_term_frequencies(feed.view(-1,))

            """ Progress checking """
            sanity_inference = sequence_tensors[0, k:k + bptt_limit]
            print("LOSS:", (loss.data.item()))
            print("Prediction:", ' '.join(predict(model, vocabulary, sanity_inference)))
            print("From:      ", ' '.join(vocabulary.text_from_encoding(sanity_inference)))
            print("Hidden state sum:", hidden.sum())

            # new_topics, new_beta = extract_topics(model, vocabulary, k=20)
            # if original_topics is None:
            #     original_topics = new_topics
            #
            # if last_topics != new_topics and last_topics is not None:
            #     print("CHANGE FROM LAST TIME!")
            #     print("O.G TOPICS ---------------------")
            #     print(tabulate(original_topics, headers=["Topic #", "Words"]))
            #     print('NEW ---------------------')
            #     print(tabulate(new_topics, headers=["Topic #", "Words"]))
            # last_topics = new_topics

            print("+--------------------------------------+")


def predict(model, vocab, sentence):
    """ Given an encoded sentence, make a prediction. """
    hidden = model.init_hidden(single_example=True)
    stops = vocab.get_stop_indicators_from_tensor(sentence).unsqueeze(0)
    output, hidden = model(Variable(sentence.unsqueeze(0)),
                           hidden, stops)
    values, indices = torch.max(output, dim=2)
    return vocab.text_from_encoding(indices.data.squeeze())


def extract_topics(model, vocabulary, k=20):
    """
    Given a model and corpus containing word encodings, print the
    top k topics present in the model.
    """
    beta = None
    for name, param in model.named_parameters():
        if name == 'beta':
            beta = param.clone()

    words = []
    for i in range(beta.size(1)):
        words.append(vocabulary.get_word(i))

    topics = []
    for i, row in enumerate(beta):
        row = [ri.item() for ri in row]
        word_strengths = list(zip(words, row))
        sorted_by_strength = sorted(word_strengths,
                                    key=lambda x: x[1],
                                    reverse=True)
        topic = [x[0] for x in sorted_by_strength][:k]
        topics.append((i, topic))

    return topics, beta


if __name__ == "__main__":
    main()
