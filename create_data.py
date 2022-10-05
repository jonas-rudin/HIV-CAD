from random import randint, shuffle, uniform, choice

import numpy as np

from helpers.IO import save_text
from helpers.config import get_config

config = get_config()
data = config['data']

bases = ['A', 'C', 'G', 'T']


# ignoring difference in homopolymeric and non-homopolymeric regions
# ignoring insertions
# TODO add insertions?
def add_error_or_mutation(sequence, error):
    i = 0
    while i < (len(sequence)):
        if uniform(0, 1) < error:
            # if random base = base -> remove base from read
            replacement_base = choice(bases)
            if sequence[i] == replacement_base:
                sequence = sequence[:i] + sequence[i + 1:]
                continue
            else:
                sequence = sequence[:i] + replacement_base + sequence[i + 1:]
        i += 1
    return sequence


# cut original haplotype in reads of certain length, creating at least amount reads
def cut_into_reads(dna, length, amount, name_of_strain):
    error = 0.00473  # 454 error
    dna_reads = []
    counter = 0
    while amount > len(dna_reads):
        first_index = int(uniform(0, length))
        last_index = first_index + randint(1, length - 1)

        prov_rna_reads = []
        # go through rna and cut it into reads
        while last_index < len(dna):
            read = add_error_or_mutation(dna[first_index:last_index], error)
            # make read fastq
            dna_reads.append(
                '@' + name_of_strain + '\n' + read + '\n+\n' + ('+' * len(read)))
            counter += 1
            first_index = last_index
            last_index += length + int(np.random.normal(0, length * 0.2, 1)[0])
        read = add_error_or_mutation(dna[first_index:len(dna)], error)
        # make read fastq
        dna_reads.append('@' + name_of_strain + '\n' + read + '\n+\n' + ('+' * len(read)))
        counter += 1
    # shuffle and remove every second element to reduce pattern in produced reads
    shuffle(dna_reads)
    return dna_reads[::2]


def create_reference(length, number_of_strains):
    og_strain = ''
    for _ in range(length):
        og_strain += choice(bases)
    snp_frequency = 0.0778
    mutations = []
    for _ in range(number_of_strains):
        mutations.append(add_error_or_mutation(og_strain, snp_frequency))
    return og_strain, mutations


# create reads
if __name__ == '__main__':
    if data != 'created':
        print('Set data to created in config')
        exit(-1)

    haplotype_length = config[data]['haplotype_length']
    number_of_strains = config['number_of_strains']
    read_length = config['read_length']
    # double to remove half later  -> less pattern in produced reads
    min_number_of_reads_per_strain = 2 * config['min_number_of_reads_per_strain']

    og_strain, mutated_strains = create_reference(haplotype_length, number_of_strains)
    print(len(mutated_strains))
    fasta_encoded = ''
    for i in range(len(mutated_strains)):
        fasta_encoded += '>' + str(i) + '\n' + mutated_strains[i] + '\n'
    save_text(config[data]['ref_path'] + '.fasta', fasta_encoded[:-1])
    save_text(config[data]['og_path'], '>OG\n' + og_strain)
    reads = []
    f = open(config['created']['ref_path'] + '.fasta', 'r')
    for _ in range(number_of_strains):
        name_of_strain = (f.readline())[1:-1]
        print('Creating reads from strain:', name_of_strain)
        # create reads
        # add variation in amount of reads per strain
        min_number_of_reads = min_number_of_reads_per_strain + min_number_of_reads_per_strain * uniform(-0.2, 0.2)
        reads.extend(cut_into_reads(f.readline()[:-1], read_length, min_number_of_reads, name_of_strain))
    shuffle(reads)
    save_text(config[data]['reads_path'] + '.fastq', '\n'.join(reads))
    print('data created')
