import tensorflow as tf
import params
import ntn_input
import random

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
        self.corrupt_size   = hyperparameters['corrupt_size']
        # self.word_indices        = hyperparameters['word_indices']
        # self.activation_function = hyperparameters['activation_function']
        self.regularization = hyperparameters['lambda']

        # a 2D tensor with entity vectors. Someone needs to make this
        # and somehow we need to be able to index into it

        data_path = hyperparameters['data_path']
        entities_strings = ntn_input.load_entities(data_path)
        self.entity_to_index = {entities_strings[i] : i for i in len(entities_strings)}
        relation_strings = tf.constant(ntn_input.load_relations(data_path))
        self.relation_to_index = {relation_strings[i] : i for i in len(relation_strings)}

        (initEmbeds, entity_words) = ntn_input.load_init_embeds(data_path) 
        
        self.entity_words = entity_words #map from entity indices to list of word indices 
        self.W = W = dict()
        self.V = V = dict()
        self.b = b = dict()
        self.U = U = dict()
        self.E = E = tf.Variable(initEmbeds) #word embeddings

        self.examples = tf.placeholder(tf.float32, shape=(4, self.batch_size), name='batch')

        for r in self.relations:
            W[r] = tf.Variable(tf.truncated_normal([d, d, k])) # W[i] = np.random.random([d, d, k]) * 2 * r - r
            V[r] = tf.Variable(tf.zeros([2 * d, k]))
            b[r] = tf.Variable(tf.zeros([1, k]))
            U[r] = tf.Variable(tf.ones([k, 1]))


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

    def L2(self):
        term = 0
        #sqrt(sum(trainable tensors))
        return term

    #randomly returns a batch_size*4 matrix: e1, R, e2, e3
    #e3 is a random vector
    def get_batch(self, batch_size, data):
        random_indices = random.sample(range(len(data)), batch_size)
        #data[i][0] = e1, data[i][1] = r, data[i][2] = e2, random=e3 (corrupted)
        batch = [(data[i][0], data[i][1], data[i][2], random.randint(len(self.num_entities))) for i in random_indices for j in range(self.corrupt_size)]
        return batch



    # LOSS
    def loss(self, batch, flip=True):        
        contrastive_max_margin = max(0.0, 1.0-g(true_triplet)+g(corrupt_triplet)) # + regularization term
        train_step = tf.train.AdagradOptimizer(0.01).minimize(contrastive_max_margin)

    def train(self):
        training_strings = ntn_input.load_training_data()
        training_data = [(self.entity_to_index(entry[0]), self.relation_to_index(entry[1]), self.entity_to_index(entry[2])) for entry in training_strings]
        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            for i in range(iterations):
                batch = self.get_batch()
                sess.run(train_step, feed_dict={'batch':batch})

if __name__=="__main__":
    hyperparameters = {"data_path": params.data_path,
            "num_iter":             params.num_iter,
            "train_both":           params.train_both,
            "batch_size":           params.batch_size,
            "corrupt_size":         params.corrupt_size,
            "embedding_size":       params.embedding_size,
            "slice_size":           params.slice_size,
            "lambda":               params.reg_parameter,
            "in_tensor_keep_normal":params.in_tensor_keep_normal,
            "save_per_iter":        params.save_per_iter,
            "gradient_checking":    params.gradient_checking
        }
    ntn = NTN(hyperparameters)
