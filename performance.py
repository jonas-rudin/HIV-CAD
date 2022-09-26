import itertools

import numpy as np

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
    if len(consensus_sequences) != len(reference_sequences):
        print(
            'number of consensus sequences and reference sequences mismatch,' + 'correction phasing rate can\'t be calculated')
        return
    adjustment = int(config[data]['haplotype_length'] / 50)
    distances = np.zeros((len(consensus_sequences), len(reference_sequences)))
    for cs_index in range(len(consensus_sequences)):
        for ref_index in range(len(reference_sequences)):
            # TODO figure out with shift to adjust -> must have the same length

            distances[cs_index][ref_index] = hamming_distance(consensus_sequences[cs_index],
                                                              reference_sequences[ref_index])

    min_indexes = []
    min_sum = 0
    # TODO fix code below... with not dumb code or with permutaions from sympy
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
