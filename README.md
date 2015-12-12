## Reimplementing Neural Tensor Networks for Knowledge Base Completion (KBC) in the TensorFlow framework
###### Alex LeNail, Clare Liu, Dustin Doss

### Abstract

[Reasoning with Neural Tensor Networks for Knowledge Base Completion](http://papers.nips.cc/paper/5028-reasoning-with-neural-tensor-networks-for-knowledge-base-completion.pdf) has become something of a seminal paper in the short span of two years, cited by nearly every knowledge base completion (KBC) paper since it's publication in 2013. It was one of the first major successful forays into the field of Deep Learning approaches to knowledge base completion, and was unique for using deep learning "end to end".

[TensorFlow](http://tensorflow.org/) is a tensor-oriented numerical computation library recently released by Google. It represents algorithms as directed acyclic graphs (DAGs), nodes as operations and edges as schemas for tensors. It has a robust python API and bindings to GPUs.

We reimplemented Socher's algorithm in the TensorFlow framework, achieving similar accuracy results with an elegant implementation in a modern language.


### Introduction

We were initially intrigued by a [Kaggle competiton](https://www.kaggle.com/c/the-allen-ai-science-challenge) called the Allen AI Science Challenge (posed by the Allen institute in Seattle), the goal of which is to answer 8th grade multiple-choice science questions with minimal training data, but access to knowledge bases. Two months after the competition commenced, (with an obvious baseline of 25%) top scores hardly exceeded 50%. We initially thought we might attempt to submit an entry to the challenge, but have since backed off to what we think is likely the key missing component in others' approaches: complete knowledge bases. We suspect that given a complete knowledge base, a knowledge base augmented by some inference algorithm to complete missing edges, we might be able to supersede the 50% threshhold.

A knowledge base is a representation of factual knowledge, traditionally in a graph-like structure. In a knowledge base, entities may be represented as nodes and relations as edges. There are a discrete number of relation types, usually a quite small set of them. Knowledge bases characterstically suffer from incompleteness, in the from of missing edges. If A is related to B, and B is related to C, oftentimes A is related to C, but knowledge bases often don't have these relations explicitly listed, because they're simply common sense, and can easily be inferred by a human. This is the "common sense" often missing in Artificially Intelligent systems, especially question answering systems which rely on knowledge bases heavily.

There are a variety of open knowledge bases available today, including but not limited to YAGO, Wordnet, and Freebase. These are often developed by hand, for example, Freebase was put together by contractors working for Google, which sought to improve search results with richer understandings of entities in search queries. Humans overlook facts we consider "obvious," for example the knowledge base may specify that "MIT" is "located in" "Cambridge" and "Cambridge" is "located in" "Massachussetts" but may neglect to draw another "located in" edge from "MIT" to "Massachussetts". This is the very simplest kind of missing edge we might encounter in a knowledge base, and we'd like to develop a method to predict the likely truth of new facts with more complicated structure, in effect, *reasoning* over known facts and inferring new ones.

### Neural Tensor Networks for Knowledge Base Completion

The goal is to predict whether two entities are related by relation R. For example, we'd like to develop a model that assigns a high likelihood to ("MIT", "located in", "Massachussetts") but a low likelihood of ("MIT", "located in", "Belize"), a model that recognizes that some facts hold purely because of other relations in the knowledge base. This is not dissimilar to the task of link prediction from other domains of machine learning.

Socher introduces "Neural Tensor Networks" which differ from regular neural networks due to the way they relate entities directly to one another with the bilinear tensor product, which is the core operation in a Neural Tensor Network.

Specifically, in a Neural Tensor Network the confidence in a directed edge of type R from e1 to e1 is defined as

g(e1, R, e2) = ...

where f is tanh applied element-wise, U and B are vectors in R^k, e1 and e2 are d-dimensional embeddings of entities, and W is a R^dxdxk tensor.

e1 * W * e2, the bilinear tensor product, computes a vector in R^k where k is the depth of the tensor. Each entry in that vector is e1 * W[k] * e2, a product which results in a scalar. V is a matrix in R^kx2d, and the product of V and [e1, e2] adds a linear neural-network part to the neural tensor network's output.

Visual NTN operation

We train with "contrastive max-margin" objective functions. The loss is defined as:

J(omega) = ...

where N is the number of true training triplets, C is the number of corrupt false examples generated for each true triplet, and lambda is a regularization parameter which modulates the L2 norm penalty for the size of the parameters. Intuitively, this objective function maximizes the gap between the confidence in the true triplet and the false triplet, for each falsification of a true triplet, for each true triplet.

False triplets are generated by replacing either e1 or e2 with a randomly drawn e3, for example, if the true triplet were "MIT", "located in", "Massachussetts" then a corrupt triplet might be "MIT", "located in", "Belize" or "Belize", "located in", "Massachussetts". Our objective function maximizes the margin between true and false, while keeping the size of the parameters in check.

### Entity Representation

The question arises of how to represent entities for a neural tensor network. One of the insights from Socher's paper is that entities can be represented as some function of their constituent words, which provides for the sharing of statistical strength between similar entities. For example, "African Elephant" and "Asian Elephant" might have much in common, but previous approaches oftentimes embedded each of these separately. We embed each work ("African", "Asian", and "Elephant") and then build representations for entities as the average of those entities' constituent word vectors.

Word embedding vectors may be initialized randomly, but the model benefits strongly from using pre-trained word embeddings (e.g. by word2vec) which already have some structure built into them. However, once we have vectors representing words, we pass gradients back to them, meaning that those embeddings are by no means static, or final.


### TensorFlow

Shortly before we picked our project, Google open-sourced TensorFlow, a numerical computing library similar to Theano and Torch, but slightly different in that although it is optimized for neural networks and has plenty of additional helper functions for computations in that domain, it was not only built for Machine Learning. Secondly, it has visual debugging tools (called TensorBoard) built in to it, to visualize learning and diagnose potential bottlenecks.















