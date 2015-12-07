## Reimplementing Neural Tensor Networks for Knowledge Base Completion (KBC) in the TensorFlow framework
###### Alex LeNail, Clare Liu, Dustin Doss

[Reasoning with Neural Tensor Networks for Knowledge Base Completion](http://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion.pdf) has become something of a seminal paper in the short span of two years, cited by nearly every knowledge base completion (KBC) paper since it's publication in 2013. It was one of the first major successful forays into the field of Deep Learning approaches to knowledge base completion, and was unique for using deep learning "end to end".

[TensorFlow](http://tensorflow.org/) is a tensor-oriented numerical computation library recently released by Google. It represents algorithms as directed acyclic graphs (DAGs), nodes as operations and edges as schemas for tensors. It has a robust python API and bindings to GPUs.

We reimplemented Socher's algorithm in the TensorFlow framework, achieving similar accuracy results with an elegant implementation in a modern language.

We were initially intrigued by a [Kaggle competiton](https://www.kaggle.com/c/the-allen-ai-science-challenge) called the AI Science Challenge, the goal of which is to answer 8th grade multiple-choice science questions with minimal training data, but access to knowledge bases. With an obvious baseline of 25%, two months after the competition commenced, top scores hardly exceed 50%. We initially thought we might attempt to submit an entry to the challenge, but have since backed off to what we think is likely the key missing component in others' approaches: complete knowledge bases. We suspect that given a complete knowledge base, augmented by some inference algorithm to complete missing edges, we might be able to break the 50% threshhold.











