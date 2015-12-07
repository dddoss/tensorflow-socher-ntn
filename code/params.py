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
act_func_num = 0 # 0 - tanh, 1 - sigmoid, 2 - identity
if act_func_num==0:
    act_func = lambda x: math.tanh(x)
    act_deriv = lambda x: 1-x**2
elif act_fun_num==1:
    act_func = lambda x: 1 / (1 + math.exp(-x))
    act_deriv = lambda x: x*(1-x)
elif act_fun_num==2:
    act_func = lambda x: x
    act_deriv = lambda x: 1
in_tensor_keep_normal = 0

save_per_iter=100
gradient_checking = True
