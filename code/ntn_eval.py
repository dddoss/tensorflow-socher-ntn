import ntn_input
import params

def getThresholds(W, V, b, U, word_vectors):
    dev_data = ntn_input.load_dev_data()
    entity_vectors = tf.Variable(tf.zeros([params.embedding_size, params.num_entities]))

    # Set entity vectors to the mean of the word vectors 
    for i in range(params.num_entities):
        entity_vectors[:, i] = tf.reduce_mean(word_vectors[:, i], 1)

    dev_scores = tf.Variable(tf.zeros(dev_data.shape[0]),1)
    for i in range(dev_data.shape[0]):
        rel = dev_data[i,1]
        e1 = tf.reshape(entity_vectors[:, dev_data[i,0]], [params.embedding_size, 1])
        e2 = tf.reshape(entity_vectors[:, dev_data[i,2]], [params.embedding_size, 1])

    entity_stack = tf.Variable(tf.concat(0, [e1, e2]))
    
    for k in range(params.slice_size):
        dev_scores[i, 0] += U[rel][k, 0] * tf.mul(tf.transpose(e1), tf.mul(W[rel][:, :, k], e2)) + tf.mul(tf.transpose(V[rel][:, k]), entity_stack) + b[rel][0, k])

    score_min = tf.reduce_min(dev_scores)
    score_max = tf.reduce_max(dev_scores)

    # initialize thresholds and accuracies
    best_thresholds = tf.zeros([self.num_relations, 1])
    best_accuracies = tf.zeros([self.num_relations, 1])

    for i in range(self.num_relations):
        best_thresholds[i, :] = score_min
        best_accuracies[i, :] = -1

    score_temp = score_min
    interval   = 0.01

    while(score_temp <= score_max):
        for i in range(self.num_relations):
            rel_i_list    = (dev_data[:, 1] == i)
            predictions   = (dev_scores[rel_i_list, 0] <= score_temp) * 2 - 1
            temp_accuracy = tf.reduce_mean((predictions == dev_labels[rel_i_list, 0]))

            # update threshold and accuracy
            if(temp_accuracy > best_accuracies[i, 0]):
                best_accuracies[i, 0] = temp_accuracy
                best_thresholds[i, 0] = score_temp

        score_temp += interval

    # store threshold values
    return best_thresholds


def getPredictions(W, V, b, U, word_vectors):
    best_thresholds = getThresholds(W, V, b, U, word_vectors)
    
    test_data = ntn_input.load_test_data()
    entity_vectors = tf.Variable(tf.zeros([params.embedding_size, params.num_entities]))

    # Set entity vectors to the mean of the word vectors 
    for entity in range(self.num_entities):
        entity_vectors[:, entity] = tf.reduce_mean(word_vectors[:, self.word_indices[entity]],1)

    predictions = tf.Variable(tf.zeros(test_data.shape[0],1))

    for i in range(test_data.shape[0]):
        rel = test_data[i, 1]
        e1  = tf.reshape(entity_vectors[:, test_data[i,0]], [params.embedding_size, 1])
        e2  = tf.reshape(entity_vectors[:, test_data[i,2]], [params.embedding_size, 1])

    entity_stack = tf.Variable(tf.concat(0, [e1, e2]))
    test_score   = 0

    # calculate prediction score for ith example
    for k in range(params.slice_size):
        test_score += U[rel][k, 0] * tf.mul(tf.transpose(e1), tf.mul(W[rel][:, :, k], e2)) + tf.mul(tf.transpose(V[rel][:, k]), entity_stack) + b[rel][0, k])

        # get labels from theshold scores
        if(test_score <= best_thresholds[rel, 0]):
            predictions[i, 0] = 1
        else:
            predictions[i, 0] = -1

    return predictions
