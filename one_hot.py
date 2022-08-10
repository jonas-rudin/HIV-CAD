import json
from os.path import exists

import numpy as np
import tensorflow as tf
import yaml

with open('./config.yml', 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)


def switcher(base):
    return {
        'A': 0.,
        'C': 1.,
        'G': 2.,
        'T': 3.
    }.get(base, -1.)


# encrypt reads according to switcher
def encode_read(read, length):
    switched_read = []
    for base in read:
        switched_read.append(switcher(base))
    # encrypted_reads.append(encrypted_read)
    # adjust length
    switched_read.extend(-1. for _ in range(length - len(switched_read)))
    switched_read_tensor = tf.cast(switched_read, tf.int32)
    # one_hot encode 0 -> [1,0,0,0], 1 -> [0,1,0,0] ... -1 -> [0,0,0,0]
    one_hot_read = tf.one_hot(switched_read_tensor, depth=4)
    return one_hot_read


def encode():
    path_to_result_file_one_hot = './results/' + config['name'] + "_one_hot"
    path_to_result_file_reads = './results/' + config['name'] + "_reads.json"
    # check if file already exists and use it if that is the case
    if config['load'] and exists(path_to_result_file_one_hot + '.npy') and exists(path_to_result_file_reads):
        print('loading files...')
        one_hot_encoded_reads_tensor = np.load(path_to_result_file_one_hot + '.npy')
        with open(path_to_result_file_reads, "r") as json_file:
            reads = json.load(json_file)
        return one_hot_encoded_reads_tensor, reads

    # read reads
    is_read_line = False
    one_hot_encoded_reads = []

    # read file line by line and one hot encode reads
    print('reading and encoding data...')
    reads = []
    with open(config['reads_file_path']) as file:
        for line in file:
            if is_read_line:
                read = line[:-1]
                reads.append(read)
                one_hot_encoded_read = encode_read(read, config['max_read_length'])
                one_hot_encoded_reads.append(one_hot_encoded_read)
                is_read_line = False
            else:
                is_read_line = True
                continue

    # convert to tensor
    one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads), axis=3)

    # save
    if config['save']:
        print('saving files...')
        np.save(path_to_result_file_one_hot, one_hot_encoded_reads_tensor)
        with open(path_to_result_file_reads + ".json", "w") as json_file:
            json.dump(reads, json_file)
    # TODO return one_hot and reads
    return one_hot_encoded_reads_tensor, reads


def decode():
    return 'Not yet implemented'
