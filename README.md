## Reimplementing Neural Tensor Networks for Knowledge Base Completion (KBC) in the TensorFlow framework
###### Alex LeNail, Clare Liu, Dustin Doss

[Reasoning with Neural Tensor Networks for Knowledeg Base Completion](http://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion.pdf) has become something of a classic paper in the short span of two years, cited by nearly every knowledge base completion (KBC) paper since it's publication in 2013. It was the first major successful foray into the field of KBC by Deep Learning approaches, and was unique by using deep learning "end to end" in a sense.

[TensorFlow](http://tensorflow.org/) is a tensor-operation library recently released by Google. It represents algorithms as directed graphs, nodes as operations and edges as schemas for tensors. It has a robust python API and has bindings for GPUs.

We were intrigued by a [Kaggle competiton](https://www.kaggle.com/c/the-allen-ai-science-challenge) called the AI Science Challenge, the goal of which is to answer 8th grade multiple-choice science questions with minimal training data, but access to Knowledge bases. With an obvious baseline of 25%, two months in, no one seems capable of beating 50%. We initially thought we might tackle the entire problem, but have since backed off to what we think is likely the key missing component in others' approaches: complete knowledge bases. We suspect that given a complete knowledge base, augmented by some inference algorithm to complete missing edges, we might be able to break the 50% threshhold.




====










