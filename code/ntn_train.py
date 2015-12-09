import tensorflow as tf
import ntn_input
#import ntn
import params
import numpy as np
import numpy.matlib

num_iters = params.num_iter
batch_size = params.batch_size
corrupt_size = params.corrupt_size

num_examples = 10
training_data = ntn_input.load_training_data()

for i in range(num_iters):
    batch_idx = np.random.randint(num_examples, size = batch_size)
    data_batch = {}
    data_batch['e1'] = numpy.matlib.repmat(training_data[batch_idx, 0], 1, corrupt_size).T
    data_batch['rel'] = numpy.matlib.repmat(training_data[batch_idx, 1], 1, corrupt_size).T
    data_batch['e2'] = numpy.matlib.repmat(training_data[batch_idx, 2], 1, corrupt_size).T

    if params.train_both and np.random.random() < 0.5:
        # replace this with correct function call from ntn
        #(params.theta, cost) = scipy.optimize.minimize(ntn.tensorNetCostFunc(data_batch, params, 0),params.theta, options))
        pass
    else:
        # replace this with correct function call from ntn
        #(params.theta, cost) = scipy.optimize.minimize(ntn.tensorNetCostFunc(data_batch, params, 1),params.theta, options))
        pass
    # update cost
    # params.cost(iter) = cost;
    # write cost to file for each iteration

  


