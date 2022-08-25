from os.path import exists

import tensorflow as tf

from helpers.IO import save_tensor_file
from helpers.config import get_config

config = get_config()


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
    # check if file already exists
    # if config['load'] and exists(config[config['data']]['one_hot_path'] + '_0.npy'):
    if config['load'] and exists(config[config['data']]['one_hot_path'] + '.npy'):
        print('reads are already one hot encoded')
        return

    # read reads
    one_hot_encoded_reads = []

    spot_index = 0
    # file_index = 0
    print('reading file: 454.fastq')
    # read file line by line and one hot encode reads
    with open(config[config['data']]['reads_path']) as file:
        for line in file:
            if spot_index % 4 == 1:
                read = line[:-1]
                one_hot_encoded_read = encode_read(read, config['max_read_length'])
                one_hot_encoded_reads.append(one_hot_encoded_read)
            spot_index += 1
            if spot_index % 4000 == 0:
                print(spot_index / 4)
                if spot_index == 12000:
                    break
            #     # convert to tensor
            #     one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads), axis=3)
            #     # save
            #     save_tensor_file(config[config['data']]['one_hot_path'], file_index, one_hot_encoded_reads_tensor)
            #     file_index += 1
            #     one_hot_encoded_reads = []
    # convert to tensor
    print(one_hot_encoded_reads[0])

    one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads), axis=3)
    print(one_hot_encoded_reads_tensor[0])

    # save
    save_tensor_file(config[config['data']]['one_hot_path'], one_hot_encoded_reads_tensor)
    print('reads are one hot encoded and saved.')
    # return file_index
# TODO create reads of length max length and return reads as done in create data
