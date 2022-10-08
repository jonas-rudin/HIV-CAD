import itertools

import numpy as np

import one_hot
from helpers.IO import load_tensor_file
from helpers.config import get_config

config = get_config()
data = config['data']


def hamming_distance(read, consensus_sequence):
    index_of_non_zero = ([np.where(read != 0)][0][0])
    difference = read[index_of_non_zero] - consensus_sequence[index_of_non_zero]
    return np.count_nonzero(difference) / 2


def minimum_error_correction(reads, consensus_sequences):
    mec = 0
    for read in reads:
        hd = []
        for consensus_sequence in consensus_sequences:
            hd.append(hamming_distance(read, consensus_sequence))
        mec += np.min(hd)
    return mec


def correct_phasing_rate(consensus_sequences):
    reference_sequences = load_tensor_file(config[data]['ref_path'])
    # TODO should i do that?
    # consensus_sequences = [shift_start_of_sequence(sequence) for sequence in consensus_sequences]

    print('\nconsensus_sequences:\n', one_hot.decode(consensus_sequences))
    print('\nreference_sequences:\n', one_hot.decode(reference_sequences))

    if len(consensus_sequences) != len(reference_sequences):
        print(
            'number of consensus sequences and reference sequences mismatch, correction phasing rate can\'t be '
            'calculated')
        return
    distances = np.zeros((len(consensus_sequences), len(reference_sequences)))
    for cs_index, consensus_sequence in enumerate(consensus_sequences):
        for ref_index, reference_sequence in enumerate(reference_sequences):
            print('ref_index', ref_index)
            # TODO New
            hd_per_ref = []
            if np.sum(reference_sequence[-1]) != 0:
                hd_per_ref.append(hamming_distance(consensus_sequence, reference_sequence))
            else:
                for index in range(len(reference_sequence - 1), -1, -1):
                    reference_sequence = np.roll(reference_sequence, 1, axis=0)
                    if np.sum(reference_sequence[-1]) != 0:
                        break
                    hd_per_ref.append(hamming_distance(consensus_sequence, reference_sequence))
            print('hd_per_ref', hd_per_ref)
            print('min(hd_per_ref)', min(hd_per_ref))
            distances[cs_index][ref_index] = min(hd_per_ref)
            # TODO Old
            # distances[cs_index][ref_index] = hamming_distance(consensus_sequences[cs_index],
            #                                                   reference_sequences[ref_index])

    min_indexes = []
    min_sum = 0
    indexes = list(range(len(reference_sequences)))
    permutations = list(itertools.permutations(indexes))
    for permutation in permutations:
        tmp_sum = 0
        for i in range(len(consensus_sequences)):
            tmp_sum += distances[i][permutation[i]]
        if len(min_indexes) == 0 or tmp_sum < min_sum:
            min_indexes = permutation
            min_sum = tmp_sum

    print('min_sum', min_sum)

    # print(result)
    # # SORRY for that code...
    #
    # min_indexes = []
    # min_value = 0
    # for a in range(len(reference_sequences)):
    #     for b in range(len(reference_sequences)):
    #         for c in range(len(reference_sequences)):
    #             for d in range(len(reference_sequences)):
    #                 for e in range(len(reference_sequences)):
    #                     if len([a, b, c, d, e]) == len(set([a, b, c, d, e])):
    #                         tmp_value = distances[0][a] + distances[1][b] + distances[2][c] + distances[3][d] + \
    #                                     distances[4][e]
    #
    #                         if len(min_indexes) == 0 or tmp_value < min_value:
    #                             min_indexes = [a, b, c, d, e]
    #                             min_value = tmp_value
    # print('min_value', min_value)
    cpr = 1 - (min_sum / (len(consensus_sequences) * config[data]['haplotype_length']))
    print(cpr)
    return min_sum, min_indexes, cpr
