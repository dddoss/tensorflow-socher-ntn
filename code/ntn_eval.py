import ntn_input
import ntn_train
import params
import tensorflow as tf

def do_eval(sess, eval_correct, batch_placeholder, corrupt_placeholder, batch):
    true_count = 0
    steps = batch.num_examples
    num_examples = steps * params.batch_size

    for step in range(steps):
        feed_dict = ntn_train.fill_feed_dict(batch, params.train_both, batch_placeholder, corrupt_placeholder)
        true_count += sess.run(eval_correct, feed_dict)
    precision = true_count / num_examples
    return precision

def evaluation(inference, labels):
    # get number of correct labels for the logits (if prediction is top 1 closest to actual)
    correct = tf.nn.in_top_k(inference, labels, 1)
    # cast tensor to int and return number of correct labels
    return tf.reduce_sum(tf.cast(correct, tf.int32))

def run_evaluation():
    test_data = ntn_input.load_test_data(params.data_path)
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_entities(params.data_path)
    
    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = params.slice_size
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)
    
    with tf.Graph().as_default():
        batch_placeholder = tf.placeholder(tf.float32, shape=(4, batch_size)
        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1)) 
        inference = ntn.inference(batch_placeholder, corrupt_placeholder, init_word_embeds, entity_to_wordvec, num_entities, num_relations, slice_size, batch_size) 

        eval_correct = ntn_eval.evaluation(inference, corrupt_placeholder)

        print do_eval(sess, eval_correct, batch_placeholder, corrupt_placeholder, test_data)
                                           
def getThresholds():`
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
