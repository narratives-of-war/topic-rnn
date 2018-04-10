import argparse
import logging
import os
import shutil
import sys

import torch
from torch import optim

sys.path.append(os.path.join(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

"""
TODO:
    Training
    Evaluation
    Logging
    Define format of Data
        - Should be separated by paragraph for time
    Define Evaluation Metrics
        - Sentiment Analysis?
        - Perplexity?
"""


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))
    parser.add_argument("--conflict-train-path", type=str,
                        default=os.path.join(
                            project_root, "conflicts", "train"),
                        help="Path to the conflict training data.")
    parser.add_argument("--conflict-dev-path", type=str,
                        default=os.path.join(
                            project_root, "conflicts", "validation"),
                        help="Path to the conflicts dev data.")
    parser.add_argument("--squad-conflicts-path", type=str,
                        default=os.path.join(
                            project_root, "conflicts", "test"),
                        help="Path to the conflicts test data.")
    parser.add_argument("--glove-path", type=str,
                        default=os.path.join(project_root, "glove",
                                             "glove.6B.50d.txt"),
                        help="Path to word vectors in GloVe format.")
    parser.add_argument("--load-path", type=str,
                        help=("Path to load a saved model from and "
                              "evaluate on test data. May not be "
                              "used with --save-dir."))
    parser.add_argument("--save-dir", type=str,
                        help=("Path to save model checkpoints and logs. "
                              "Required if not using --load-path. "
                              "May not be used with --load-path."))
    parser.add_argument("--model-type", type=str, default="topic-rnn",
                        choices=["topic-rnn"],
                        help="Model type to train.")
    parser.add_argument("--min-token-count", type=int, default=10,
                        help=("Number of times a token must be observed "
                              "in order to include it in the vocabulary."))
    parser.add_argument("--max-passage-length", type=int, default=150,
                        help="Maximum number of words in the passage.")
    parser.add_argument("--max-question-length", type=int, default=15,
                        help="Maximum number of words in the question.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size to use in training and evaluation.")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden size to use in RNN and Attention models.")
    parser.add_argument("--num-epochs", type=int, default=25,
                        help="Number of epochs to train for.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout proportion.")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="The learning rate to use.")
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


def train_epoch(model):
    """
    Train the model for one epoch.
    """

    # Set model to training mode (activates dropout and other things).
    model.train()


if __name__ == "__main__":
    main()
