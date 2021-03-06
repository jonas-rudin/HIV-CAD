from os.path import exists
from random import randint, shuffle, uniform, choice

import numpy as np
import tensorflow as tf
import yaml

import one_hot

with open("./config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

bases = ['A', 'C', 'G', 'T']


# TODO add sequencing error (randomly remove or change base) (according to 454 error)
# add in an error to the read error rate: 0.49% per base
# ignoring difference in homopolymeric and non-homopolymeric regions
def add_error(read):
    i = 0
    while i < (len(read)):
        if uniform(0, 1) < 0.0049:
            # if random base = base -> remove base from read
            replacement_base = choice(bases)
            if read[i] == replacement_base:
                read = read[:i] + read[i + 1:]
                continue
            else:
                read = read[:i] + replacement_base + read[i + 1:]
            i += 1
    return read


# cut original haplotype in reads of certain length, creating at least amount reads
def cut_into_reads(rna, length, amount):
    rna_reads = []
    while amount > len(rna_reads):
        first_index = 0
        last_index = randint(1, length - 1)
        prov_rna_reads = []
        # go through rna and cut it into reads
        while last_index < len(rna):
            prov_rna_reads.append(add_error(rna[first_index:last_index]))
            first_index = last_index
            last_index += length + int(np.random.normal(0, length * 0.2, 1)[0])
        prov_rna_reads.append(add_error(rna[first_index:len(rna)]))
        rna_reads.extend(prov_rna_reads)
    return rna_reads


# create reads from existing haplotype
def create_reads():
    number_of_strains = config['number_of_strains']
    min_number_of_reads_per_strain = config['min_number_of_reads_per_strain']
    read_length = config['read_length']
    path_to_results_file = './results/' + config['name'] + '_created_' + str(number_of_strains) + '_' + str(
        read_length) + '_' + str(min_number_of_reads_per_strain)

    # load if exists
    if config['load'] and exists(path_to_results_file + '.npy'):
        print('loading file...')
        return np.load(path_to_results_file + '.npy')

    # read original haplotype
    f = open(config['haplotype_file_path'], 'r')
    reads = []
    for _ in range(number_of_strains):
        print("Creating reads from strain:", (f.readline())[1:])
        # create reads
        reads.extend(cut_into_reads(f.readline()[:-1], read_length, min_number_of_reads_per_strain))
    shuffle(reads)
    # one hot encoding, convert list to tensor
    # and add dimension (needed for conv2D in tensorflow shape(x,y) -> shape(x,y,1))
    one_hot_encoded_reads = tf.expand_dims(
        tf.convert_to_tensor([one_hot.encode_read(read, config["max_created_read_length"]) for read in reads]), axis=3)

    # save
    if config['save']:
        print('saving file...')
        np.save(path_to_results_file, one_hot_encoded_reads)

    return one_hot_encoded_reads
