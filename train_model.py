import argparse
from collections import Counter
import logging
import math
import os
import pickle
from tqdm import tqdm
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import cross_entropy, log_softmax

sys.path.append(os.path.join(os.path.dirname(__file__)))
from Dictionary import Corpus, word_vector_from_seq,\
    extract_tokens_from_conflict_json
from topic_rnn_rc.models.topic_rnn import TopicRNN
from topic_rnn_rc.models.rnn import RNN
from topic_rnn_rc.models.lstm import LSTM

logger = logging.getLogger(__name__)

"""
TODO:
    Training
    Evaluation
    Logging
    Define Evaluation Metrics
        - Sentiment Analysis?
        - Perplexity?
"""

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
    parser.add_argument("--conflicts-train-path", type=str,
                        default=os.path.join(
                            project_root, "data", "train"),
                        help="Path to the Conflict Wikipedia JSON"
                             "training data.")
    parser.add_argument("--conflicts-dev-path", type=str,
                        default=os.path.join(
                            project_root, "data", "validation"),
                        help="Path to the Conflict Wikipedia JSON"
                             "dev data.")
    parser.add_argument("--conflicts-test-path", type=str,
                        default=os.path.join(
                            project_root, "data", "test"),
                        help="Path to the Conflict Wikipedia JSON"
                             "dev data.")
    parser.add_argument("--built-corpus-path", type=str,
                        default=os.path.join(
                            project_root, "corpus.pkl"),
                        help="Path to a pre-constructed corpus.")
    parser.add_argument("--stopwords-path", type=str,
                        default=os.path.join(
                            project_root, "en.txt"),
                        help="Path to the list of stop words.")
    parser.add_argument("--save-dir", type=str,
                        help=("Path to save model checkpoints and logs. "
                              "Required if not using --load-path. "
                              "May not be used with --load-path."))
    parser.add_argument("--model-type", type=str, default="vanilla",
                        choices=["vanilla", "topic", "lstm"],
                        help="Model type to train.")
    parser.add_argument("--min-token-count", type=int, default=10,
                        help=("Number of times a token must be observed "
                              "in order to include it in the vocabulary."))
    parser.add_argument("--clip", type=int, default=0.33,
                        help="Gradient Clipping.")
    parser.add_argument("--bptt-limit", type=int, default=35,
                        help="Extent in which the model is allowed to"
                             "backpropagate.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to use in training and evaluation.")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden size to use in RNN and TopicRNN models.")
    parser.add_argument("--embedding-size", type=int, default=50,
                        help="Embedding size to enocde words for the RNNs.")
    parser.add_argument("--num-epochs", type=int, default=25,
                        help="Number of epochs to train for.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout proportion.")
    parser.add_argument("--lr", type=float, default=0.00,
                        help="The learning rate to use.")
    parser.add_argument("--log-period", type=int, default=50,
                        help=("Update training metrics every "
                              "log-period weight updates."))
    parser.add_argument("--validation-period", type=int, default=500,
                        help=("Calculate metrics on validation set every "
                              "validation-period weight updates."))
    parser.add_argument("--weight-decay", type=float, default=0.5,
                        help="L2 Penalty.")
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

    if not args.conflicts_train_path:
        raise ValueError("Training data directory required")

    if args.model_type not in MODEL_TYPES:
        raise ValueError("Please select a supported model.")

    print("Building corpus:")
    print("Restricting vocabulary based on min token count",
          args.min_token_count)

    print("Collecting stopwords:")
    stopwords_file = open(args.stopwords_path, 'r')
    stops = set([s.strip() for s in stopwords_file.readlines()])

    print("Collecting War Wikipedia JSONs:")
    if not os.path.exists(args.built_corpus_path):
        # Pickle the corpus for easy access
        corpus = init_corpus(args.conflicts_train_path,
                             args.min_token_count, stops)
        pickled_corpus = open(args.built_corpus_path, 'wb')
        pickle.dump(corpus, pickled_corpus)
    else:
        corpus = pickle.load(open(args.built_corpus_path, 'rb'))

    vocab_size = corpus.vocab_size
    print("Vocabulary Size:", vocab_size)
    print("Stop size:", corpus.stop_size)
    print("Vocab size no stops:", corpus.vocab_size_no_stops)

    # Create model of the correct type.
    print("Building {} RNN model ------------------".format(args.model_type))
    logger.info("Building {} RNN model".format(args.model_type))

    if args.model_type != "topic":
        # RNN / LSTM construction
        model = MODEL_TYPES[args.model_type](vocab_size, args.embedding_size,
                                             args.hidden_size, args.batch_size,
                                             layers=2, dropout=args.dropout)
    else:
        # TopicRNN Construction
        model = TopicRNN(vocab_size, args.embedding_size, args.hidden_size,
                         corpus.stop_size, args.batch_size)

        # Sanity check
        for name, param in model.named_parameters():
            if param.requires_grad:
                nn.init.uniform(param)

    if args.cuda:
        model.cuda()

    logger.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    if args.model_type != "topic":
        # Non-topic RNN models are trained on loss that compares
        # predicted and actual words.
        try:
            train_epoch(model, corpus, args.batch_size, args.bptt_limit, optimizer,
                        args.cuda)
        except KeyboardInterrupt:
            pass
    else:
        try:
            train_topic_rnn(model, corpus, args.bptt_limit, args.clip, optimizer,
                            args.cuda)
        except KeyboardInterrupt:
            pass

        print()  # Printing in-place progress flushes standard out.

    # Calculate perplexity.
    perplexity = evaluate_perplexity(model, corpus, args.batch_size,
                                     args.bptt_limit, args.cuda,
                                     model_type=args.model_type)

    print("\n\nFinal perplexity: {}".format(float(perplexity)))


def init_corpus(training_path, min_token_count, stops):
    training_files = os.listdir(training_path)
    tokens = []
    for file in tqdm(training_files):
        file_path = os.path.join(training_path, file)
        tokens += extract_tokens_from_conflict_json(file_path)

    # Map words to the number of times they occur in the corpus.
    word_frequencies = dict(Counter(tokens))

    # Sieve the dictionary by excluding all words that appear fewer
    # than min_token_count times.
    vocabulary = set([w for w, f in word_frequencies.items()
                      if f >= min_token_count])

    # Construct the corpus with the given vocabulary.
    corpus = Corpus(vocabulary, stops)

    print("Constructing corpus from JSON files:")
    for file in tqdm(training_files):
        # Corpus expects a full file path.
        corpus.add_document(os.path.join(training_path, file))

    return corpus


def train_topic_rnn(model, corpus, bptt_limit, clip, optimizer, cuda):
    """
    This model trains differently than baselines; it computes the likelihood
    of a portion of text under the model instead of doing cross entropy against
    the next word.

    We maintain processing one word at a time to allow previous sections to
    have direct influence on later ones (i.e. indifferent batching will
    cause issues).
    """

    def batchify_section(seq_tensor):
        # Partition the current sequence tensor into batches the same
        # length as our backpropagation-through-time limit.
        num_batches = seq_tensor.size(0) // bptt_limit

        # Not long enough for training!
        if num_batches == 0:
            return None, 0

        # Discard portions that don't fit evenly.
        seq_tensor = seq_tensor.narrow(0, 0, num_batches * bptt_limit)

        # (num_batches x length)
        seq_tensor = seq_tensor.view(num_batches, -1).contiguous()

        if cuda:
            seq_tensor = seq_tensor.cuda()

        return seq_tensor, num_batches

    # Set model to training mode (activates dropout and other things).
    model.train()
    print("Training in progress:")
    for i, document in enumerate(corpus.documents):
        # Incorporation of time requires feeding in by one word at
        # a time.
        #
        # Iterate through the words of the document, calculating loss between
        # the current word and the next, from first to penultimate.
        document_name = document["title"]
        for j, section in enumerate(document["sections"]):

            # Batchify the sequence tensor according to backpropagation limit.
            batched_section, batches = batchify_section(section)

            if batches == 0:
                continue

            for k, portion in enumerate(batched_section):
                # This uses an encoding from words to integers in a
                # space that excludes stop words.
                portion_frequencies = corpus.compute_term_frequencies(portion, corpus)
                stop_indicators = corpus.get_stop_indicators(portion)

                # Optimize on negative log likelihood.
                loss = -torch.log(model.likelihood(portion, portion_frequencies,
                                  stop_indicators, cuda))

                # Perform backpropagation and update parameters.
                optimizer.zero_grad()
                loss.backward(retain_graph=True)

                # Helps with exploding/vanishing gradient
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                optimizer.step()

                # Print progress
                print_progress_in_place("Document:", document_name,
                                        "Section:", j,
                                        "Portion:", k,
                                        "of", batches,
                                        "Normalized BPTT Loss:",
                                        loss.data[0] / bptt_limit)


def train_epoch(model, corpus, batch_size, bptt_limit, optimizer, cuda):
    """
    Train the model for one epoch.
    """

    # Set model to training mode (activates dropout and other things).
    model.train()
    print("Training in progress:")
    for i, document in enumerate(corpus.documents):
        # Incorporation of time requires feeding in by one word at
        # a time.
        #
        # Iterate through the words of the document, calculating loss between
        # the current word and the next, from first to penultimate.
        document_name = document["title"]
        for j, section in enumerate(document["sections"]):
            loss = 0
            hidden = model.init_hidden()

            # Training at the word level allows flexibility in inference.
            for k in range(section.size(0) - 1):
                current_word = word_vector_from_seq(section, k)
                next_word = word_vector_from_seq(section, k + 1)

                if cuda:
                    current_word = current_word.cuda()
                    next_word = next_word.cuda()

                output, hidden = model(Variable(current_word), hidden)

                # Calculate loss between prediction and what was anticipated.
                loss += cross_entropy(output.view(1, -1), Variable(next_word))

                # Perform backpropagation and update parameters.
                #
                # Detaches hidden state history to prevent bp all the way
                # back to the start of the section.
                if (k + 1) % bptt_limit == 0:
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    # Print progress
                    print_progress_in_place("Document:", document_name,
                                            "Section:", j,
                                            "Word:", k,
                                            "Normalized BPTT Loss:",
                                            loss.data[0] / bptt_limit)

                    loss = 0
                    if type(hidden) == tuple:
                        hidden = tuple(Variable(hidden[i].data)
                                       for i in range(len(hidden)))
                    else:
                        hidden = Variable(hidden.data)


def evaluate_perplexity(model, corpus, batch_size, bptt_limit, cuda, model_type="rnn"):
    """
    Calculate perplexity of the trained model for the given corpus..
    """

    M = 0  # Word count.
    log_prob_sum = 0  # Log Probability

    # Set model to evaluation mode (deactivates dropout).
    model.eval()
    print("Evaluation in progress: Perplexity")
    for i, document in enumerate(corpus.documents):
        # Iterate through the words of the document, calculating log
        # probability of the next word given the history at the time.
        for j, section in enumerate(document["sections"]):
            hidden = model.init_hidden()

            # Training at the word level allows flexibility in inference.
            stop_indicators = corpus.get_stop_indicators(section)
            for k in range(section.size(0) - 1):
                current_word = word_vector_from_seq(section, k)
                next_word = word_vector_from_seq(section, k + 1)

                if cuda:
                    current_word = current_word.cuda()
                    next_word = next_word.cuda()

                if model_type == "topic":
                    output, hidden = model(Variable(current_word), hidden, stop_indicators[k])
                else:
                    output, hidden = model(Variable(current_word), hidden)

                # Calculate probability of the next word given the model
                # in log space.
                # Reshape to (vocab size x 1) and perform log softmax over
                # the first dimension.
                prediction_probabilities = log_softmax(output.view(-1, 1), 0)

                # Extract next word's probability and update.
                prob_next_word = prediction_probabilities[next_word[0]]
                log_prob_sum += prob_next_word.data[0]
                M += 1

                # Detaches hidden state history at the same rate that is
                # done in training..
                if (k + 1) % bptt_limit == 0:
                    # Print progress
                    print_progress_in_place("Document #:", i,
                                            "Section:", j,
                                            "Word:", k,
                                            "M:", M,
                                            "Log Prob Sum:", log_prob_sum,
                                            "Normalized Perplexity thus far:",
                                            2 ** (-(log_prob_sum / M)))

                    if type(hidden) == tuple:
                        hidden = tuple(Variable(hidden[i].data)
                                       for i in range(len(hidden)))
                    else:
                        hidden = Variable(hidden.data)

        # Final model perplexity given the corpus.
        return 2 ** (-(log_prob_sum / M))


def print_progress_in_place(*args):
    print("\r", *args, end="")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
