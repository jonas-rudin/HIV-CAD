from random import randint, shuffle

import numpy as np
import yaml

import one_hot

with open("./config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)


def cut_into_reads(rna, length, amount):
    rna_reads = []
    while amount > len(rna_reads):
        first_index = 0
        last_index = randint(1, length - 1)
        prov_rna_reads = []
        while last_index < len(rna):
            prov_rna_reads.append(rna[first_index:last_index])
            first_index = last_index
            last_index += length + int(np.random.normal(0, length * 0.2, 1)[0])
        prov_rna_reads.append(rna[first_index:len(rna)])
        rna_reads.extend(prov_rna_reads)
    return rna_reads


def create_reads():
    number_of_strains = config['number_of_strains']
    min_number_of_reads_per_strain = config['min_number_of_reads_per_strain']
    read_length = config['read_length']
    f = open(config['haplotype_file_path'], 'r')
    reads = []
    for _ in range(number_of_strains):
        print("Creating reads from strain:", (f.readline())[1:])
        rna = f.readline()[:-1]
        reads.extend(cut_into_reads(rna, read_length, min_number_of_reads_per_strain))
    print(reads)
    shuffle(reads)
    print(reads)
    one_hot_encoded_reads = [one_hot.encode_read(read) for read in reads]
    print(one_hot_encoded_reads)
    if config['save']:
        print('saving file...')
        np.save(
            './results/' + config['name'] + 'created_' + str(number_of_strains) + '_' + str(read_length) + '_' + str(
                len(reads)),
            one_hot_encoded_reads)
        print('file saved')
        
# TODO add sequencing error (randomly remove or change base) (according to 454 error)
