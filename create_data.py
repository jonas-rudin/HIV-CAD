from os.path import exists
from random import randint, shuffle, uniform, choice

import numpy as np
import tensorflow as tf
import yaml

import one_hot
from helpers.IO import save, load

with open("./config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

bases = ['A', 'C', 'G', 'T']


# TODO add sequencing error (randomly remove or change base) (according to 454 error)
# add in an error to the read error rate: 0.49% per base
# ignoring difference in homopolymeric and non-homopolymeric regions
def add_error(read):
    i = 0
    while i < (len(read)):
        if uniform(0, 1) < 0.00473:
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
def cut_into_reads(dna, length, amount, name_of_strain):
    dna_reads = []
    counter = 0
    max_length = config["max_created_read_length"]
    while amount > len(dna_reads):
        first_index = 0
        last_index = randint(1, length - 1)
        prov_rna_reads = []
        # go through rna and cut it into reads
        while last_index < len(dna):
            read = add_error(dna[first_index:last_index])
            # read += '-' * (max_length - len(read))
            # make read fastq
            dna_reads.append(
                '@' + str(counter) + ' ' + name_of_strain + read + '\n+\n' + ('+' * len(read)))

            counter += 1
            first_index = last_index
            last_index += length + int(np.random.normal(0, length * 0.2, 1)[0])
        read = add_error(dna[first_index:len(dna)])
        # read += '-' * (max_length - len(read))
        # make read fastq
        dna_reads.append('@' + str(counter) + ' ' + name_of_strain + read + '\n+\n' + ('+' * len(read)))
        counter += 1
    return dna_reads


# create reads from existing haplotype
def create_reads():
    number_of_strains = config['number_of_strains']
    min_number_of_reads_per_strain = config['min_number_of_reads_per_strain']
    read_length = config['read_length']
    path_to_result_file_one_hot = config['created']['one_hot_path']
    path_to_result_file_reads = config['created']['reads_path']

    # load if exists
    if config['load'] and exists(path_to_result_file_one_hot + '.npy') and exists(
            path_to_result_file_reads):
        load(path_to_result_file_one_hot, path_to_result_file_reads)

    # read original haplotype
    f = open(config['created']['ref_path'], 'r')
    reads = []
    for _ in range(number_of_strains):
        name_of_strain = (f.readline())[1:]
        print("Creating reads from strain:", name_of_strain)
        # create reads
        reads.extend(cut_into_reads(f.readline()[:-1], read_length, min_number_of_reads_per_strain, name_of_strain))
    shuffle(reads)
    reads_for_one_hot = [read.split('\n')[1] for read in reads]

    # one hot encoding, convert list to tensor
    one_hot_encoded_reads = tf.expand_dims(
        tf.convert_to_tensor(
            [one_hot.encode_read(read, config['max_created_read_length']) for read in reads_for_one_hot]), axis=3)

    # save
    if config['save']:
        save(path_to_result_file_one_hot, one_hot_encoded_reads, path_to_result_file_reads, '\n'.join(reads))

    return one_hot_encoded_reads, reads
