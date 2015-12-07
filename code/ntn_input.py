# Load entity, relation data, precomputed entity vectors based on specified database
# Currently supports FreeBase and Wordnet data
# Author: Dustin Doss

import params
import scipy.io as sio

entities_string='/entities.txt'
relations_string='/relations.txt'
embeds_string='/initEmbed.mat'

#input: path of dataset to be used
#output: python list of entities in dataset
def load_entities(data_path=params.data_path):
    entities_file = open(data_path+entities_string)
    entities_list = entities_file.read().strip().split('\n')
    entities_file.close()
    return entities_list

#input: path of dataset to be used
#output: python list of relations in dataset
def load_relations(data_path=params.data_path):
    relations_file = open(data_path+relations_string)
    relations_list = relations_file.read().strip().split('\n')
    relations_file.close()
    return relations_list

#input: path of dataset to be used
#output: python dict from entity string->1x100 vector embedding of entity as precalculated
def load_init_embeds(data_path=params.data_path):
    embeds_path = data_path+embeds_string
    return load_embeds(embeds_path)

#input: Generic function to load embeddings from a .mat file
def load_embeds(file_path):
    mat_contents = sio.loadmat(file_path)
    words = mat_contents['words']
    we = mat_contents['we']
    word_vecs = {str(words[0][i][0]) : [we[j][i] for j in range(params.embedding_size)] for i in range(len(words[0]))}
    return word_vecs

def load_training_data(data_path=params.data_path):
    training_file = open(data_path+training_string)
    training_data = [line.split('\t') for line in training_file.read().strip().split('\n')]
    return training_data
