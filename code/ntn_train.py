import tensorflow as tf
import ntn_input
import ntn
import params
import numpy as np
import numpy.matlib

def data_to_indexed(data, entities, relations):
    entity_to_index = {entities[i] : i for i in range(len(entities))}
    relation_to_index = {relations[i] : i for i in range(len(relations))}
    indexed_data = [(entity_to_index[data[i][0]], relation_to_index[data[i][1]],\
            entity_to_index[data[i][2]]) for i in range(len(data))]
    return indexed_data

def get_batch(batch_size, data, num_entities, corrupt_size):
    random_indices = random.sample(range(len(data)), batch_size)
    #data[i][0] = e1, data[i][1] = r, data[i][2] = e2, random=e3 (corrupted)
    batch = [(data[i][0], data[i][1], data[i][2], random.randint(len(num_entities)))\
	for i in random_indices for j in range(corrupt_size)]
    return batch

def fill_feed_dict(batch, train_both, batch_placeholder, corrupt_placeholder):
    return {batch_placeholder: batch, corrupt_placeholder: (train_both and np.random.random()>0.5)}

def run_training():
    print("Begin!")
    #python list of (e1, R, e2) for entire training set in string form
    print("Load training data...")
    raw_training_data = ntn_input.load_training_data(params.data_path)
    print("Load entities and relations...")
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_relations(params.data_path)
    #python list of (e1, R, e2) for entire training set in index form
    indexed_training_data = data_to_indexed(raw_training_data, entities_list, relations_list)
    print("Load embeddings...")
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    num_iters = params.num_iter
    batch_size = params.batch_size
    corrupt_size = params.corrupt_size
    slice_size = params.slice_size

    with tf.Graph().as_default():
        print("Starting to build graph")
        batch_placeholder = tf.placeholder(tf.float32, shape=(4, batch_size))
        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1)) #Which of e1 or e2 to corrupt?
        inference = ntn.inference(batch_placeholder, corrupt_placeholder, init_word_embeds, entity_to_wordvec, \
                num_entities, num_relations, slice_size, batch_size)
        loss = ntn.loss(inference, params.regularization)
        training = ntn.training(loss, params.learning_rate)

	# Create a session for running Ops on the Graph.
	sess = tf.Session()

	# Run the Op to initialize the variables.
	init = tf.initialize_all_variables()
	sess.run(init)
    saver = tf.train.Saver()
        for i in range(num_iters):
            print("Starting iter "+str(i))
            data_batch = get_batch(batch_size, indexed_training_data, num_entities, corrupt_size)

            if i % 5 == 0 and i != 0: saver.save(sess, params.output_dir)

	    feed_dict = fill_feed_dict(data_batch, params.train_both, batch_placeholder, corrupt_placeholder)
            _, loss_value = sess.run([training, loss], feed_dict=feed_dict)

            #TODO: Eval against dev set?
            #TODO: Save model!

def main(argv):
    run_training()

if __name__=="__main__":
    tf.app.run()
