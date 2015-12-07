import tensorflow as tf

# Inference
# Loss
# Training

class NTN:
	def __init__(self, parameters):

		# for r in relations:

			W[r] = tf.Variable([])
			V[r] = tf.Variable([])
			b[r] = tf.Variable([])	# each of these are dicts? could also be extra-D tensors
									# could also do "with tf.name_scope('hidden1') as scope:" scope each relation type

		# We may need this:

		E = # a 2D tensor with every single entity vector



	# e1 and e2 are d-dimensional entity vectors. W is a dxdxk tensor.
	def bilinearTensorProduct(self, e1, W, e2):

		e1 = tf.reshape(e1, [1, d])
		W = tf.reshape(W, [d, d*k])

		temp = tf.matmul(e1, W)

		temp = tf.reshape(temp, [k, d])
		e2 = tf.reshape(e2, [d, 1])

		temp = tf.matmul(temp, e2)

		return temp


	def g(self, (e1, R, e2)):

		temp1 = bilinearTensorProduct(e1, W, e2)

		temp2 = tf.matmul(V, tf.concat(0, [e1, e2]))

		temp = tf.add(temp1, temp2, b)

		temp = tf.tanh(temp)

		temp = tf.matmul(U, temp)

		return temp


	# LOSS

	def loss(self, batch):

		contrastive_max_margin = max(0, 1-g(true_triplet)+g(corrupt_triplet)) # + regularization term


	def train(self):

		train_step = tf.train.AdagradOptimizer(0.01).minimize(contrastive_max_margin)


