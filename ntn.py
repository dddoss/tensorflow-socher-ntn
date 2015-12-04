import tensorflow as tf


# tf.tanh()


# tf.matmul()
# tf.batch_matmul()


# def g((e1, R, e2)):

# These are just notes and ideas about functions we use


contrastive_max_margin = max(0, 1-g(true_triplet)+g(corrupt_triplet)) # + regularization term


train_step = tf.train.AdagradOptimizer(0.01).minimize(contrastive_max_margin)

