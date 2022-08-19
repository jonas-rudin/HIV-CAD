import json

import numpy as np


# TODO split into save/load general files and .npy
# saves reads and one hot encoded reads to files
def save(path_to_result_file_one_hot, one_hot_encoded_reads, path_to_result_file_reads, reads):
    print('saving files...')
    np.save(path_to_result_file_one_hot, one_hot_encoded_reads)
    with open(path_to_result_file_reads, "w") as fasta_file:
        fasta_file.write(reads)
    return


# loads reads and one hot encoded reads from files
def load(path_to_result_file_one_hot, path_to_result_file_reads):
    print('loading files...')
    one_hot_encoded_reads = np.load(path_to_result_file_one_hot + '.npy')
    with open(path_to_result_file_reads, "r") as json_file:
        reads = json.load(json_file)
    return one_hot_encoded_reads, reads
