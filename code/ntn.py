import tensorflow as tf
import params

# Inference
# Loss
# Training

class NTN:
    def __init__(self, hyperparameters):
        # self.num_words           = hyperparameters['num_words']
        self.d = d          = hyperparameters['embedding_size']
        self.num_entities   = hyperparameters['num_entities']
        self.num_relations  = hyperparameters['num_relations']
        self.batch_size     = hyperparameters['batch_size']
        self.k = k          = hyperparameters['slice_size']
        # self.word_indices        = hyperparameters['word_indices']
        # self.activation_function = hyperparameters['activation_function']
        self.lambda          = hyperparameters['lambda']

        # a 2D tensor with entity vectors. Someone needs to make this
        # and somehow we need to be able to index into it
        self.E = Entities()

        for r in range(self.num_relations):
            W[r] = tf.Variable([])
            V[r] = tf.Variable(tf.zeros([2 * d, k]))
            b[r] = tf.Variable(tf.zeros([1, k]))
            U[r] = tf.Variable(tf.ones([k, 1]))
            W[i] = np.random.random([d, d, k]) * 2 * r - r

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
        train_step = tf.train.AdagradOptimizer(0.01).minimize(contrastive_max_margin)

    def train(self):
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)

if name=="__main__":
    hyperparameters = {"data_path":params.data_path,
            "num_iter":params.num_iter,
            "train_both":params.train_both,
            "batch_size":params.batch_size,
            "corrupt_size":params.corrupt_size,
            "embedding_size":params.embedding_size,
            "slice_size":params.slice_size,
            "lambda":params.reg_parameter,
            "activation_function":params.act_func,
            "activation_derivative":params.act_deriv,
            "in_tensor_keep_normal":params.in_tensor_keep_normal,
            "save_per_iter":params.save_per_iter,
            "gradient_checking":params.gradient_checking
        }
    ntn = NTN(hyperparameters) 
