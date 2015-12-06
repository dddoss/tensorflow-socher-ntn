import tensorflow as tf

# e1 and e2 are d-dimensional entity vectors. W is a dxdxk tensor.
def bilinear_tensor_product(e1, W, e2):
	return tf.matmul()

# tf.matmul()
# tf.batch_matmul()


W = tf.Variable([])
V = tf.Variable([])
b = tf.Variable([])

def g((e1, R, e2)):
	a = tf.add(bilinear_tensor_product(e1, W, e2), tf.matmul(V, tf.concat(0, [e1, e2])), b)
	b = tf.tanh(a)
	return tf.matmul(U, b)

# These are just notes and ideas about functions we use
contrastive_max_margin = max(0, 1-g(true_triplet)+g(corrupt_triplet)) # + regularization term
train_step = tf.train.AdagradOptimizer(0.01).minimize(contrastive_max_margin)
