import itertools

import numpy as np

import one_hot
from helpers.IO import load_tensor_file
from helpers.config import get_config

config = get_config()
data = config['data']


# old shape
# def hamming_distance(read, consensus_sequence):
#     index_of_non_zero = ([np.where(read != 0)][0][0])
#     difference = read[index_of_non_zero] - consensus_sequence[index_of_non_zero]
#     return np.count_nonzero(difference) / 2

def hamming_distance(read, consensus_sequence):
    difference = (consensus_sequence - read)[np.where(read != 0)]
    return np.count_nonzero(difference)


def minimum_error_correction(reads, consensus_sequences):
    mec = 0
    for read in reads:
        hd = []
        for consensus_sequence in consensus_sequences:
            hd.append(hamming_distance(read, consensus_sequence))
        mec += np.min(hd)
    return mec


# # old shape
# def correct_phasing_rate(consensus_sequences):
#     if data == 'created':
#         reference_sequences = load_tensor_file(
#             config[data]['aligned_ref_path'] + '_' + str(config[data]['n_clusters']))
#     else:
#         reference_sequences = load_tensor_file(config[data]['aligned_ref_path'])
#
#     print('\nreference_sequences:\n', one_hot.decode(reference_sequences, True))
#
#     print('\nconsensus_sequences:\n', one_hot.decode(consensus_sequences))
#
#     if len(consensus_sequences) != len(reference_sequences):
#         print(
#             'number of consensus sequences and reference sequences mismatch, correction phasing rate can\'t be '
#             'calculated')
#         return
#     distances = np.zeros((len(consensus_sequences), len(reference_sequences)))
#     if data == 'per_gene_454':
#         # TODO do per consensus sequence
#         start = config[data]['haplotype_start']
#         for cs_index, consensus_sequence in enumerate(consensus_sequences):
#             for ref_index, reference_sequence in enumerate(reference_sequences):
#                 hd_per_ref = []
#                 # window of index plus minus 100
#                 for start_index in range(start - 100, start + 100):
#                     end_index = start_index + len(consensus_sequence)
#                     hd_per_ref.append(
#                         hamming_distance(consensus_sequence, reference_sequence[start_index:end_index]))
#                     # reference_sequence = np.roll(reference_sequence, 1, axis=0)
#                 distances[cs_index][ref_index] = min(hd_per_ref)
#
#     else:
#         for cs_index, consensus_sequence in enumerate(consensus_sequences):
#             for ref_index, reference_sequence in enumerate(reference_sequences):
#                 hd_per_ref = []
#                 # if last element not part of sequence
#                 # -> move reference sequence to the right till last element is part of sequence
#                 if np.sum(reference_sequence[-1]) != 0:
#                     hd_per_ref.append(hamming_distance(consensus_sequence, reference_sequence))
#                 else:
#                     for index in range(len(reference_sequence - 1), -1, -1):
#                         if np.sum(reference_sequence[-1]) != 0:
#                             break
#                         hd_per_ref.append(hamming_distance(consensus_sequence, reference_sequence))
#                         reference_sequence = np.roll(reference_sequence, 1, axis=0)
#
#                 distances[cs_index][ref_index] = min(hd_per_ref)
#
#     min_indexes = []
#     min_sum = 0
#     indexes = list(range(len(reference_sequences)))
#     permutations = list(itertools.permutations(indexes))
#     for permutation in permutations:
#         tmp_sum = 0
#         for i in range(len(consensus_sequences)):
#             tmp_sum += distances[i][permutation[i]]
#         if len(min_indexes) == 0 or tmp_sum < min_sum:
#             min_indexes = permutation
#             min_sum = tmp_sum
#
#     # print('min_sum', min_sum)
#     cpr = 1 - (min_sum / (len(consensus_sequences) * reference_sequences.shape[1]))
#     # print(cpr)
#
#     return min_sum, min_indexes, cpr


def correct_phasing_rate(consensus_sequences):
    if data == 'created':
        reference_sequences = load_tensor_file(
            config[data]['aligned_ref_path'] + '_' + str(config[data]['n_clusters']))
    else:
        reference_sequences = load_tensor_file(config[data]['aligned_ref_path'])

    reference_sequences = np.transpose(reference_sequences, axes=(0, 2, 1, 3))
    consensus_sequences = np.transpose(consensus_sequences, axes=(0, 2, 1, 3))

    print('\nreference_sequences:\n', one_hot.decode(reference_sequences, True))
    print('\nconsensus_sequences:\n', one_hot.decode(consensus_sequences))

    if len(consensus_sequences) != len(reference_sequences):
        print(
            'number of consensus sequences and reference sequences mismatch, correction phasing rate can\'t be '
            'calculated')
        return
    distances = np.zeros((len(consensus_sequences), len(reference_sequences)))
    for cs_index, consensus_sequence in enumerate(consensus_sequences):
        for ref_index, reference_sequence in enumerate(reference_sequences):
            hd_per_ref = []
            # if last element not part of sequence
            # -> move reference sequence to the right till last element is part of sequence
            if np.sum(reference_sequence[-1]) != 0:
                hd_per_ref.append(hamming_distance(consensus_sequence, reference_sequence))
            else:
                for index in range(len(reference_sequence - 1), -1, -1):
                    if np.sum(reference_sequence[-1]) != 0:
                        break
                    hd_per_ref.append(hamming_distance(consensus_sequence, reference_sequence))
                    reference_sequence = np.roll(reference_sequence, 1, axis=0)

            distances[cs_index][ref_index] = min(hd_per_ref)

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
    print(reference_sequences.shape[1])
    print(len(consensus_sequences))
    cpr = 1 - (min_sum / (len(consensus_sequences) * reference_sequences.shape[1]))
    # print(cpr)
    return min_sum, min_indexes, cpr
