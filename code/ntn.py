import tensorflow as tf
import params
import ntn_input
import random

# Inference
# Loss
# Training

# E1 and E2 are matrices of b columns of d-dim entity vectors. W is a dxdxk tensor.
# def BATCH_bilinearTensorProduct(self, E1, W, E2):

#     e1 = tf.reshape(e1, [d, b, 1])
#     W = tf.reshape(W, [d*k, d])

#     temp = tf.matmul(e1, W)
#     temp = tf.reshape(temp, [k, d])
#     e2 = tf.reshape(e2, [d, 1])
#     temp = tf.matmul(temp, e2)
#     return temp

# def BATCH_g(self, (e1, R, e2)):

#     temp1 = bilinearTensorProduct(e1, W, e2)
#     temp2 = tf.matmul(V, tf.concat(0, [e1, e2]))
#     temp = tf.add(temp1, temp2, b)
#     temp = tf.tanh(temp)
#     temp = tf.matmul(U, temp)
#     return temp

# e1 and e2 are d-dimensional entity vectors. W is a dxdxk tensor.
def bilinearTensorProduct(e1, W, e2):
    e1 = tf.reshape(e1, [1, d])
    W = tf.reshape(W, [d, d*k])
    temp = tf.matmul(e1, W)
    temp = tf.reshape(temp, [k, d])
    e2 = tf.reshape(e2, [d, 1])
    temp = tf.matmul(temp, e2)
    return temp

def g((e1, R, e2)):
    temp1 = bilinearTensorProduct(e1, W, e2)
    temp2 = tf.matmul(V, tf.concat(0, [e1, e2]))
    temp = tf.add(temp1, temp2, b)
    temp = tf.tanh(temp)
    temp = tf.matmul(U, temp)
    return temp

#def L2():
#    term = 0
#    #sqrt(sum(trainable tensors))
#    return term

# LOSS OLD
#def loss(batch, flip=True):
#    contrastive_max_margin = max(0.0, 1.0-g(true_triplet)+g(corrupt_triplet)) # + regularization term
#    train_step = tf.train.AdagradOptimizer(0.01).minimize(contrastive_max_margin)

#returns a (batch_size*corrupt_size, 2) vector corresponding to [g(T^i), g(T_c^i)] for all i
def inference(batch_placeholder, corrupt_placeholder, init_word_embeds,\
        entity_to_wordvec, num_entities, num_relations, slice_size, batch_size):
    #TODO: We need to check the shapes and axes used here!
    d = 100 #embed_size
    k = slice_size
    num_words = len(init_word_embeds)
    E = tf.Variable(init_word_embeds, shape=(num_words, d)) #d=embed size
    W = [tf.Variable(tf.truncated_normal([d,d,k])) for r in range(len(num_relations))]
    V = [tf.Variable(tf.zeros([2 * d, k])) for r in range(len(num_relations))]
    b = [tf.Variable(tf.zeros([1, k])) for r in range(len(num_relations))]
    U = [tf.Variable(tf.ones([k, 1])) for r in range(len(num_relations))]

    ent2words = [[0]*num_words for i in range(num_entities)]
    for i in range(len(entity_to_wordvec)):
        for j in entity_to_wordvec[i]:
            ent2words[i][j] = 1.0/len(entity_to_wordvec[i])
    ent2words_tensor = tf.Constant(ent2words) #each row i cooresponds to entity i; e2w_tensor[i]*E=entity embedding

    e1, R, e2, e3 = tf.split(1, 4, batch_placeholder) #TODO: should the split dimension be 0 or 1?
    #convert entity word index reps to embeddings... how?
    e1v = tf.pack([tf.matmul(E, tf.split(0, batch_size, tf.gather(ent2words_tensor, e1v)) for i in range(batch_size)]
    e2v = tf.pack([tf.matmul(E, tf.split(0, batch_size, tf.gather(ent2words_tensor, e2v)) for i in range(batch_size)]
    e3v = tf.pack([tf.matmul(E, tf.split(0, batch_size, tf.gather(ent2words_tensor, e3v)) for i in range(batch_size)]
    
    #e1v, e2v, e3v should be (batch_size * 100) tensors by now
    for r in range(num_relations):
        #calc g(e1, R, e2) and g(e1, R, e3) for each relation


def loss(infer_results):
    pass

def training(loss_results):
    pass

def eval(infer_results):
    pass
