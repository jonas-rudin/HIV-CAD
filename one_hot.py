from os.path import exists

import numpy as np
import tensorflow as tf
import yaml

with open("./config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)


def switcher(base):
    return {
        'A': 0.,
        'C': 1.,
        'G': 2.,
        'T': 3.
    }.get(base, -1.)


# encrypt reads according to switcher
def encode_read(read):
    switched_read = []
    for base in read:
        switched_read.append(switcher(base))
    # encrypted_reads.append(encrypted_read)
    encrypted_read = tf.cast(switched_read, tf.int32)
    one_hot_read = tf.one_hot(encrypted_read, depth=4)
    # TODO all the same size
    return one_hot_read


def encode():
    # check if file already exists and use it if that is the case
    if config['load'] and exists("./results/" + config['name'] + ".npy"):
        print('loading file...')
        return np.load("./results/" + config['name'] + ".npy")

    # read reads
    is_read_line = False
    one_hot_encoded_reads = []

    # counter = 0
    with open(config['reads_file_path']) as file:
        for line in file:
            # if counter == 1:
            #     break
            if is_read_line:
                read = line[:-1]
                one_hot_encoded_read = encode_read(read)
                one_hot_encoded_reads.append(one_hot_encoded_read)
                is_read_line = False
                # counter += 1
            else:
                is_read_line = True
                continue
    if config['save']:
        print('saving file...')
        np.save("./results/" + config['name'], one_hot_encoded_reads)
    return one_hot_encoded_reads


def decode():
    return "Not yet implemented"
