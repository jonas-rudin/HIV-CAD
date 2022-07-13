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


def encrypt_rna_reads(reads):
    encrypted_reads = []
    for read in reads:
        singleReadList = list(read)
        encrypted_read = []
        for base in singleReadList:
            encrypted_read.append(switcher(base))
        #             print(singleRead)
        # #             print(''.join(singleReadEncrypted))
        #             print(singleReadEncrypted)
        encrypted_reads.append(encrypted_read)

    return encrypted_reads


def encode_hot_one():
    # check if file already exists and use it if that is the case
    if config['load'] and exists("./results/" + config['name'] + ".npy"):
        print('loading file...')
        print(np.load("./results/" + config['name'] + ".npy"))
        return np.load("./results/" + config['name'] + ".npy")

    # read reads
    fasta_file = open(config['reads_file_path'], 'r')
    lines = fasta_file.readlines()
    is_read_line = False
    read_lines = []
    # counter = 0
    for line in lines:
        # if counter == 1:
        #     break
        if is_read_line:
            read_lines.append(line[:-1])
            is_read_line = False
            # counter += 1
        else:
            is_read_line = True
            continue
    # encrypt reads according to switcher
    encrypt_reads = encrypt_rna_reads(read_lines)
    # hot one encode reads
    hot_one_reads = []
    for read in encrypt_reads:
        read = tf.cast(read, tf.int32)
        hot_one_read = tf.one_hot(read, depth=4)
        hot_one_reads.append(hot_one_read)
    # save created tensor
    if config['save']:
        print('saving file...')
        np.save("./results/" + config['name'], hot_one_reads)
        np.savetxt("./results/" + config['name'] + "txt", hot_one_reads)
    return hot_one_reads


def decode_hot_one():
    return "Not yet implemented"
