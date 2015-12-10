import math

data_number = 0 #0 - Wordnet, 1 - Freebase
if data_number == 0: data_path = '../data/Wordnet'
else: data_path = '../data/Freebase'

num_iter = 1
train_both = False
batch_size = 20000
corrupt_size = 10 # how many negative examples are given for each positive example?
embedding_size = 100
slice_size = 3 #depth of tensor for each relation
regularization = 0.0001 #parameter \lambda used in L2 normalization
in_tensor_keep_normal = False
save_per_iter = 5
learning_rate = 0.01

output_dir = ''

