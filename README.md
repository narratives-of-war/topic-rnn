# TopicRNN
PyTorch implementation of Dieng et al.'s [TopicRNN](https://www.semanticscholar.org/paper/TopicRNN%3A-A-Recurrent-Neural-Network-with-Semantic-Dieng-Wang/412068c7e8e77b73add471789d58df3d2f3e08d8): a language model that combines local (syntatic) dependencies, with global (semantic) dependencies to produce a contextual RNN that can be trained end-to-end.

The RNN is responsible for capturing local, syntactic properties while the learned, latent topics are responsbile for capturing global semantics and coherence.
