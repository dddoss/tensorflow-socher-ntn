import tensorflow as tf
import params
import ntn_input
import random

# Inference
# Loss
# Training

#returns a (batch_size*corrupt_size, 2) vector corresponding to [g(T^i), g(T_c^i)] for all i
def inference(batches_placeholder, corrupt_placeholder, init_word_embeds, entity_to_wordvec,\
        num_entities, num_relations, slice_size, batch_size, is_eval=False, label_placeholders=None):
    print("Beginning building inference:")
    #TODO: We need to check the shapes and axes used here!
    print("Creating variables")
    d = 100 #embed_size
    k = slice_size
    ten_k = tf.constant([k])
    num_words = len(init_word_embeds)
    E = tf.Variable(init_word_embeds) #d=embed size
    W = [tf.Variable(tf.truncated_normal([d,d,k])) for r in range(num_relations)]
    V = [tf.Variable(tf.zeros([k, 2*d])) for r in range(num_relations)]
    b = [tf.Variable(tf.zeros([k, 1])) for r in range(num_relations)]
    U = [tf.Variable(tf.ones([1, k])) for r in range(num_relations)]

    print("Calcing ent2word")
    #python list of tf vectors: i -> list of word indices cooresponding to entity i
    ent2word = [tf.constant(entity_to_wordvec[i]) for i in range(num_entities)]
    #(num_entities, d) matrix where row i cooresponds to the entity embedding (word embedding average) of entity i
    print("Calcing entEmbed...")
    entEmbed = tf.pack([tf.reduce_mean(tf.gather(E, ent2word[i]), 0) for i in range(num_entities)])
    #TEST: entEmbed = tf.truncated_normal([num_entities, d])

    predictions = list()
    print("Beginning relations loop")
    for r in range(num_relations):
        print("Relations loop "+str(r))
        e1, e2, e3 = tf.split(0, 3, tf.cast(batches_placeholder[r], tf.int32)) #TODO: should the split dimension be 0 or 1?
        e1v = tf.squeeze(tf.gather(entEmbed, e2),[0])
        e2v = tf.squeeze(tf.gather(entEmbed, e2),[0])
        e3v = tf.squeeze(tf.gather(entEmbed, e3),[0])
        e1v_pos = e1v
        e2v_pos = e2v
        e1v_neg = e1v
        e2v_neg = e3v
        num_rel_r = tf.expand_dims(tf.shape(e1v_pos)[0], 0)
        preactivation_pos = list()
        preactivation_neg = list()

        print("e1v_pos: "+str(e1v_pos.get_shape()))
        print("W[r][:,:,slice]: "+str(W[r][:,:,0].get_shape()))
        print("e2v_pos: "+str(e2v_pos.get_shape()))

        #print("Starting preactivation funcs")
        for slice in range(k):
            preactivation_pos.append(tf.reduce_sum(e1v_pos * tf.matmul(W[r][:,:,slice], e2v_pos), 0))
            preactivation_neg.append(tf.reduce_sum(e1v_neg * tf.matmul(W[r][:,:,slice], e2v_neg), 0))

        preactivation_pos = tf.pack(preactivation_pos)
        preactivation_neg = tf.pack(preactivation_neg)

        temp2_pos = tf.matmul(V[r], tf.concat(0, [e1v_pos, e2v_pos]))
        temp2_neg = tf.matmul(V[r], tf.concat(0, [e1v_neg, e2v_neg]))

        #print("   temp2_pos: "+str(temp2_pos.get_shape()))
        preactivation_pos = preactivation_pos+temp2_pos+b[r]
        preactivation_neg = preactivation_neg+temp2_neg+b[r]

        #print("Starting activation funcs")
        activation_pos = tf.tanh(preactivation_pos)
        activation_neg = tf.tanh(preactivation_neg)

        score_pos = tf.reshape(tf.matmul(U[r], activation_pos), num_rel_r)
        score_neg = tf.reshape(tf.matmul(U[r], activation_neg), num_rel_r)
        #print("score_pos: "+str(score_pos.get_shape()))
        if not is_Eval:
            predictions.append(tf.pack([score_pos, score_neg]))
        else:
            predictions.append(tf.pack([score_pos, label_placeholders[r]]))
        #print("score_pos_and_neg: "+str(predictions[r].get_shape()))


    #print("Concating predictions")
    predictions = tf.concat(1, predictions)
    #print(predictions.get_shape())

    return predictions


def loss(predictions, regularization):

    print("Beginning building loss")
    temp1 = tf.maximum(tf.sub(predictions[1, :], predictions[0, :]) + 1, 0)
    temp1 = tf.reduce_sum(temp1)

    temp2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))

    temp = temp1 + (regularization * temp2)

    return temp


def training(loss, learningRate):
    print("Beginning building training")

    return tf.train.AdagradOptimizer(learningRate).minimize(loss)


def eval(predictions):
    pass






