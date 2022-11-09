from os.path import exists
from random import shuffle, uniform, choice

import numpy as np

from helpers.IO import save_text
from helpers.colors_coding import ColorCoding
from helpers.config import get_config

config = get_config()
data = config['data']

bases = ['A', 'C', 'G', 'T']


# ignoring difference in homopolymeric and non-homopolymeric regions
def add_error(sequence):
    error = config[data]["sequencing_error"]
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
            replacement_base = choice(bases)
            sequence = sequence[:i] + replacement_base + sequence[i + 1:]
        i += 1
    return sequence


def add_mutation(sequence, mutation_rate, longest=False):
    print(sequence)
    # TODO  -> safe correctly for data preparation
    i = 0
    insertions = 0
    snp_positions = []
    while i < (len(sequence)):

        if uniform(0, 1) < mutation_rate:

            snp_positions.append(i)
            bases_to_choose_from = []
            for base in bases:
                if base != sequence[i]:
                    bases_to_choose_from.append(base)
            # deletion
            if not longest and uniform(0, 1) < 0.0175:
                sequence = sequence[:i] + '-' + sequence[i + 1:]
                continue

            # replacement
            else:
                replacement_base = choice(bases_to_choose_from)
                sequence = sequence[:i] + replacement_base + sequence[i + 1:]
            # sequence = sequence[:i] + replacement_base + sequence[i + 1:]
        i += 1
    return sequence, snp_positions


# cut original haplotype in reads of certain length, creating at least amount reads
def cut_into_reads(dna, length, coverage, name_of_strain):
    dna_reads = []
    counter = 0
    for _ in range(2 * coverage):
        # while amount > len(dna_reads):
        first_index = 0
        last_index = int(uniform(0, length))

        prov_rna_reads = []
        # go through rna and cut it into reads
        while last_index < len(dna):
            read = add_error(dna[first_index:last_index])
            # make read fastq
            dna_reads.append(
                '@c' + name_of_strain + '\n' + read + '\n+\n' + ('+' * len(read)))
            counter += 1
            first_index = last_index
            last_index += length + int(np.random.normal(0, length * 0.05, 1)[0])
        read = add_error(dna[first_index:len(dna)])
        # make read fastq
        dna_reads.append('@c' + name_of_strain + '\n' + read + '\n+\n' + ('+' * len(read)))
        counter += 1
    # shuffle and remove every second element to reduce pattern in produced reads
    shuffle(dna_reads)
    return dna_reads[::2]


def create_reference(length, number_of_strains):
    if exists(config[data]['og_path']):
        og_file = open(config[data]['og_path'])
        lines = og_file.readlines()
        og_strain = lines[1]
        print(og_strain)
    else:
        og_strain = ''
        for _ in range(length):
            og_strain += choice(bases)
        save_text(config[data]['og_path'], '>OG\n' + og_strain)

    # snp_frequency = 0.0778 / 5
    snp_frequency = 0.1 / 5
    mutated_strains = []
    if exists(config[data]['ref_path'] + '_' + str(number_of_strains) + '.fasta'):
        print('using existing file:', config[data]['ref_path'] + '_' + str(number_of_strains) + '.fasta')
        ref_file = open(config[data]['ref_path'] + '_' + str(number_of_strains) + '.fasta')
        lines = ref_file.readlines()
        for line in lines:
            if line[0] != '>':
                mutated_strains.append(line)
        longest_file = open(config[data]['longest_path'] + '_' + str(number_of_strains) + '.fasta')
        lines = longest_file.readlines()
        longest = lines[1]
    else:
        aligned_mutated_strains = []
        longest, combined_snp_positions = add_mutation(og_strain, snp_frequency, True)
        for _ in range(number_of_strains - 1):
            sequence, snp_positions = add_mutation(og_strain, snp_frequency)
            mutated_strains.append(sequence.replace('-', ''))
            aligned_mutated_strains.append(sequence)
            combined_snp_positions.extend(snp_positions)
        all_mutated_strains = [longest] + mutated_strains
        print(*all_mutated_strains, sep="\n")
        print('-----------------------------------------------')
        all_aligned_mutated_strains = [longest] + aligned_mutated_strains
        print(*all_aligned_mutated_strains, sep="\n")
        # exit(-1)
        fasta_encoded = ''
        aligned_fasta_encoded = ''
        for i in range(len(all_mutated_strains)):
            fasta_encoded += '>' + str(i) + '\n' + all_mutated_strains[i] + '\n'
            aligned_fasta_encoded += '>' + str(i) + '\n' + all_aligned_mutated_strains[i] + '\n'
        # longest = max(mutated_strains, key=len)
        # index = mutated_strains.index(longest)
        save_text(config[data]['ref_path'] + '_' + str(number_of_strains) + '.fasta', fasta_encoded[:-1])
        save_text(config[data]['aligned_ref_path'] + '_' + str(number_of_strains) + '.fasta',
                  aligned_fasta_encoded[:-1])
        save_text(config[data]['longest_path'] + '_' + str(number_of_strains) + '.fasta',
                  '>c0\n' + longest)
        # remove duplicates
        combined_snp_positions = list(dict.fromkeys(combined_snp_positions))
        combined_snp_positions.sort()
        with open(config[data]['snp_positions'] + '_' + str(number_of_strains) + '.txt', 'w') as fp:
            for position in combined_snp_positions:
                fp.write("%s\n" % position)
            print(combined_snp_positions)
    return og_strain, mutated_strains, longest


# create reads
def create_data():
    if data != 'created':
        print(f'{ColorCoding.WARNING}Set data to created in config{ColorCoding.ENDC}')
        exit(-1)

    haplotype_length = config[data]['haplotype_length']
    number_of_strains = config[data]['n_clusters']
    read_length = config[data]['read_length']

    og_strain, mutated_strains, longest = create_reference(haplotype_length, number_of_strains)

    coverage = config[data]['coverage']

    reads = []
    if not exists(config['created']['reads_path'] + '_' + str(config[data]['n_clusters']) + '.fasta'):
        f = open(config['created']['ref_path'] + '_' + str(config[data]['n_clusters']) + '.fasta', 'r')
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
        save_text(config['created']['reads_path'] + '_' + str(config[data]['n_clusters']) + '_' + str(
            config[data]['coverage']) + '_' + str(config[data]['read_length']) + '_' + str(
            config[data]['sequencing_error']) + '.fastq', '\n'.join(reads))
    print('Data created')
