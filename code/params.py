import math

data_number = 0 #0 - Wordnet, 1 - Freebase
if data_number == 0:
    data_path = '../data/Wordnet'
else:
    data_path = '../data/Freebase'

# init_num = 0 #TODO: What do?

num_iter = 500
train_both = False

batch_size = 20000
corrupt_size = 10 #how many negative examples are given for each positive example?
embedding_size = 100
slice_size = 3 #depth of tensor for each relation

reg_parameter = 0.0001 #parameter \lambda used in L2 normalization

in_tensor_keep_normal = False

save_per_iter=100
gradient_checking = True
