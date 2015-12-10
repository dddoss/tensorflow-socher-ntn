import ntn_input
import ntn_train
import params
import tensorflow as tf

def do_eval(sess, eval_correct, batch_placeholders, label_placeholders, corrupt_placeholder, test_batches, test_labels):
    true_count = 0.
    num_examples = len(dataset)

    feed_dict = fill_feed_dict(test_batches, test_labels, params.train_both, batch_placeholder, corrupt_placeholder)
    true_count = sess.run(eval_correct, feed_dict)
    precision = float(true_count) / float(num_examples)
    return precision

def fill_feed_dict(batches, labels, train_both, batch_placeholders, label_placeholders, corrupt_placeholder):
    feed_dict = {corrupt_placeholder: (train_both and np.random.random()>0.5)}
    for i in range(len(batch_placeholders)):
        feed_dict[batch_placeholders[i]] = batches[i]
    for i in range(len(label_placeholders)):
        feed_dict[label_placeholders[i]] = labels[i]

def evaluation(inference):
    # get number of correct labels for the logits (if prediction is top 1 closest to actual)
    correct = tf.nn.in_top_k(inference, labels, 1)
    # cast tensor to int and return number of correct labels
    return tf.reduce_sum(tf.cast(correct, tf.int32))

#dataset is in the form (e1, R, e2, label)
def data_to_relation_sets(dataset, num_relations):
    batches = [[] for i in range(num_relations)]
    labels = [[] for i in range(num_relations)]
    for e1,r,e2,label in data_batch:
        batches[r].append((e1,e2,e3))
        labels[r].append(label)
    return (batches, labels)

def run_evaluation():
    test_data = ntn_input.load_test_data(params.data_path)
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_entities(params.data_path)
    
    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)
    batches, labels = data_to_relation_sets(test_data, num_relations)
    
    with tf.Graph().as_default(),sess = tf.Session():
        batch_placeholders = [tf.placeholder(tf.float32, shape=(4, None) for i in range(num_relations)]
        label_placeholders = [tf.placeholder(tf.float32, shape=(1, None) for i in range(num_relations)]
        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1)) 
        inference = ntn.inference(batch_placeholder, corrupt_placeholder, init_word_embeds, entity_to_wordvec, \
                num_entities, num_relations, slice_size, is_eval=True, label_placeholders=label_placeholders) 
        eval_correct = ntn_eval.evaluation(inference)

        tf.train.saver.restore(sess, params.sess_path)
        print do_eval(sess, eval_correct, batch_placeholders, label_placeholders, corrupt_placeholder, test_batches, test_labels)
                                           
def getThresholds():
    dev_data = ntn_input.load_dev_data()
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_entities(params.data_path)
    
    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)

    batch_placeholder = tf.placeholder(tf.float32, shape=(4, batch_size))
    corrupt_placeholder = tf.placeholder(tf.bool, shape=(1)) #Which of e1 or e2 to corrupt?
    predictions_list = ntn.inference(batch_placeholder, corrupt_placeholder, init_word_embeds,entity_to_wordvec, num_entities, num_relations, slice_size, batch_size)

    min_score = tf.reduce_min(predictions_list)
    max_score = tf.reduce_max(predictions_list)

    # initialize thresholds and accuracies
    best_thresholds = tf.zeros([params.num_relations, 1])
    best_accuracies = tf.zeros([params.num_relations, 1])

    for i in range(params.num_relations):
        best_thresholds[i, :] = score_min
        best_accuracies[i, :] = -1

    score = min_score
    increment = 0.01

    while(score <= max_score):
        # iterate through relations list to find 
        for i in range(params.num_relations):
            current_relation_list = (dev_data[:, 1] == i)
            predictions = (predictions_list[current_relation_list, 0] <= score) * 2 - 1
            accuracy = tf.reduce_mean((predictions == dev_labels[current_relations_list, 0]))

            # update threshold and accuracy
            if(accuracy > best_accuracies[i, 0]):
                best_accuracies[i, 0] = accuracy
                best_thresholds[i, 0] = score

        score += increment

    # store threshold values
    return best_thresholds

def getPredictions():
    best_thresholds = getThresholds()
    test_data = ntn_input.load_test_data()
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_entities(params.data_path)
    
    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)

    batch_placeholder = tf.placeholder(tf.float32, shape=(4, batch_size))
    corrupt_placeholder = tf.placeholder(tf.bool, shape=(1)) #Which of e1 or e2 to corrupt?
    predictions_list = ntn.inference(batch_placeholder, corrupt_placeholder, init_word_embeds,entity_to_wordvec, num_entities, num_relations, slice_size, batch_size)

    predictions = tf.zeros((test_data.shape[0], 1))
    for i in range(test_data.shape[0]):
        # get relation
        rel = test_data[i, 1]

        # get labels based on predictions
        if(preictions_list[i, 0] <= self.best_thresholds[rel, 0]):
            predictions[i, 0] = 1
        else:
            predictions[i, 0] = -1

    return predictions
