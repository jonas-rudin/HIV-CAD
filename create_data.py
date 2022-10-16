from random import shuffle, uniform, choice

import numpy as np

from helpers.IO import save_text
from helpers.colors_coding import ColorCoding
from helpers.config import get_config

config = get_config()
data = config['data']

bases = ['A', 'C', 'G', 'T']


# ignoring difference in homopolymeric and non-homopolymeric regions
def add_error_or_mutation(sequence, error):
    i = 0
    insertions = 0
    while i < (len(sequence)):
        if uniform(0, 1) < error:
            # if random base = base -> remove base from read
            replacement_base = choice(bases + ['I'])
            # insertion
            if replacement_base == 'I':
                insertions += 1
                # if more insertions than deletions do deletion
                if insertions > 0:
                    sequence = sequence[:i] + sequence[i + 1:]
                    insertions -= 1
                    continue
                insert = choice(bases)
                sequence = sequence[:i + 1] + insert + sequence[i + 1:]
                i += 1

            # deletion
            elif sequence[i] == replacement_base:
                sequence = sequence[:i] + sequence[i + 1:]
                insertions -= 1
                continue
            # replacement
            else:
                sequence = sequence[:i] + replacement_base + sequence[i + 1:]
        i += 1
    return sequence


# cut original haplotype in reads of certain length, creating at least amount reads
def cut_into_reads(dna, length, coverage, name_of_strain):
    error = 0.00473  # 454 error
    dna_reads = []
    counter = 0
    for _ in range(2 * coverage):
        # while amount > len(dna_reads):
        first_index = 0
        last_index = int(uniform(0, length))

        prov_rna_reads = []
        # go through rna and cut it into reads
        while last_index < len(dna):
            read = add_error_or_mutation(dna[first_index:last_index], error)
            # make read fastq
            dna_reads.append(
                '@c' + name_of_strain + '\n' + read + '\n+\n' + ('+' * len(read)))
            counter += 1
            first_index = last_index
            last_index += length + int(np.random.normal(0, length * 0.05, 1)[0])
        read = add_error_or_mutation(dna[first_index:len(dna)], error)
        # make read fastq
        dna_reads.append('@c' + name_of_strain + '\n' + read + '\n+\n' + ('+' * len(read)))
        counter += 1
    # shuffle and remove every second element to reduce pattern in produced reads
    shuffle(dna_reads)
    return dna_reads[::2]


def create_reference(length, number_of_strains):
    og_strain = ''
    for _ in range(length):
        og_strain += choice(bases)
    snp_frequency = 0.0778 / 5
    mutations = []
    for _ in range(number_of_strains):
        mutations.append(add_error_or_mutation(og_strain, snp_frequency))  # , strain=True))
    return og_strain, mutations


# create reads
if __name__ == '__main__':
    if data != 'created':
        print(f'{ColorCoding.WARNING}Set data to created in config{ColorCoding.ENDC}')
        exit(-1)

    haplotype_length = config[data]['haplotype_length']
    number_of_strains = config[data]['number_of_strains']
    read_length = config[data]['read_length']
    # double to remove half later  -> less pattern in produced reads
    # min_number_of_reads_per_strain = 2 * config['min_number_of_reads_per_strain']

    og_strain, mutated_strains = create_reference(haplotype_length, number_of_strains)

    print(len(mutated_strains))
    longest = max(mutated_strains, key=len)
    index = mutated_strains.index(longest)
    print(longest)
    print(index)
    fasta_encoded = ''
    for i in range(len(mutated_strains)):
        fasta_encoded += '>' + str(i) + '\n' + mutated_strains[i] + '\n'
    save_text(config[data]['ref_path'] + '.fasta', fasta_encoded[:-1])

    save_text(config[data]['og_path'], '>OG\n' + og_strain)
    save_text(config[data]['longest_path'], '>c' + str(index) + '\n' + longest)

    coverage = config[data]['coverage']

    reads = []
    f = open(config['created']['ref_path'] + '.fasta', 'r')
    for _ in range(number_of_strains):
        name_of_strain = (f.readline())[1:-1]
        print('Creating reads from strain:', name_of_strain)
        # create reads
        # add variation in amount of reads per strain
        # min_number_of_reads = min_number_of_reads_per_strain + min_number_of_reads_per_strain * uniform(-0.05, 0.05)
        dna = (f.readline()[:-1])  # .replace('-', '')
        # reads.extend(
        #     cut_into_reads(dna, read_length, min_number_of_reads, name_of_strain))

        reads.extend(
            cut_into_reads(dna, read_length, coverage, name_of_strain))
    shuffle(reads)
    save_text(config[data]['reads_path'] + '.fastq', '\n'.join(reads))
    print('data created')
