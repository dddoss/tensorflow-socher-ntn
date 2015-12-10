import tensorflow as tf
import params
import ntn_input
import random

# Inference
# Loss
# Training

#returns a (batch_size*corrupt_size, 2) vector corresponding to [g(T^i), g(T_c^i)] for all i
def inference(batch_placeholder, corrupt_placeholder, init_word_embeds,\
        entity_to_wordvec, num_entities, num_relations, slice_size, batch_size):
    print("Beginning building inference:")
    #TODO: We need to check the shapes and axes used here!
    print("Creating variables")
    d = 100 #embed_size
    k = slice_size
    ten_k = tf.constant([k])
    num_words = len(init_word_embeds)
    E = tf.Variable(init_word_embeds) #d=embed size
    W = [tf.Variable(tf.truncated_normal([d,d,k])) for r in range(num_relations)]
    V = [tf.Variable(tf.zeros([2 * d, k])) for r in range(num_relations)]
    b = [tf.Variable(tf.zeros([1, k])) for r in range(num_relations)]
    U = [tf.Variable(tf.ones([k, 1])) for r in range(num_relations)]

    print("Calcing ent2word")
    #python list of tf vectors: i -> list of word indices cooresponding to entity i
    ent2word = [tf.constant(entity_to_wordvec[i]) for i in range(num_entities)]
    #(num_entities, d) matrix where row i cooresponds to the entity embedding (word embedding average) of entity i
    print("Calcing entEmbed..."),
    entEmbed = tf.pack([tf.reduce_mean(tf.gather(E, ent2word[i]), 0) for i in range(num_entities)])
    print("Done")

    e1, R, e2, e3 = tf.split(1, 4, tf.cast(batch_placeholder, tf.int32)) #TODO: should the split dimension be 0 or 1?
    #convert entity word index reps to embeddings
    

    print("Gathering e1-e3 embeddings")
    e1v = tf.gather(entEmbed, e1)
    e2v = tf.gather(entEmbed, e2)
    e3v = tf.gather(entEmbed, e3)

    print("Partitioning e1v-e3v on r")
    e1r_pos = tf.dynamic_partition(e1v, R, num_relations)
    e2r_pos = tf.dynamic_partition(e2v, R, num_relations)
    e3r = tf.dynamic_partition(e3v, R, num_relations)

    e1r_neg = e1r_pos
    e2r_neg = e3r


    predictions = list()

    print("Beginning relations loop")
    #e1v, e2v, e3v should be (batch_size * 100) tensors by now
    for r in range(num_relations):
        print("Relations loop "+str(r))
        #calc g(e1, R, e2) and g(e1, R, e3) for each relation
        # predictions.append(tf.pack([g(e1r[r],  W[r], e2r[r]), g(e1r[r], W[r], e3r[r])]))
        num_rel_r = tf.shape(e1r_pos[r])
        preactivation_pos = list()
        preactivation_neg = list()

        print("Starting preactivation funcs")
        for slice in range(k):
            preactivation_pos.append(tf.reduce_sum(e1r_pos[r] * tf.matmul(W[r][:,:,slice], e2r_pos[r]), 0))
            preactivation_neg.append(tf.reduce_sum(e1r_neg[r] * tf.matmul(W[r][:,:,slice], e2r_neg[r]), 0))

        preactivation_pos = tf.pack(preactivation_pos)
        preactivation_neg = tf.pack(preactivation_neg)

        temp2_pos = tf.matmul(V[r], tf.concat(0, [e1r_pos[r], e2r_pos[r]]))
        temp2_neg = tf.matmul(V[r], tf.concat(0, [e1r_neg[r], e2r_neg[r]]))
        preactivation_pos = preactivation_pos+temp2_pos+b[r]
        preactivation_neg = preactivation_neg+temp2_neg+b[r]

        print("Starting activation funcs")
        activation_pos = tf.tanh(preactivation_pos)
        activation_neg = tf.tanh(preactivation_neg)

        score_pos = tf.reshape(tf.matmul(U[r], activation_pos), num_rel_r)
        score_neg = tf.reshape(tf.matmul(U[r], activation_neg), num_rel_r)
        tf.pack([score_pos, score_neg])

    print("Concating predictions")
    predictions = tf.concat(1, predictions)

    return predictions


def loss(predictions, regularization):

    print("Beginning building loss")
    temp1 = tf.max(tf.sub(predictions[:, 1], predictions[:, 0]) + 1, 0)
    temp1 = tf.sum(temp)

    temp2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))

    temp = temp1 + (regularization * temp2)

    return temp


def training(loss, learningRate):
    print("Beginning building training")

    return tf.train.AdagradOptimizer(learningRate).minimize(loss)


def eval(predictions):
    pass






