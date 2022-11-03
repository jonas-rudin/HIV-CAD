import itertools

import numpy as np

import one_hot
from helpers.IO import load_tensor_file
from helpers.config import get_config

config = get_config()
data = config['data']


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


def correct_phasing_rate(consensus_sequences, info='', reverse=False):
    if data == 'created':
        reference_sequences = load_tensor_file(
            config[data]['aligned_ref_path'] + '_' + str(config[data]['n_clusters']))
    else:
        reference_sequences = load_tensor_file(config[data]['aligned_ref_path'])

    if data == 'per_gene':
        prov = []
        for reference_sequence in reference_sequences:
            sequence = list(reference_sequence[config[data]['start_ref']:config[data]['end_ref']])
            if len(sequence) % 4 != 0:
                additional = 4 - len(sequence) % 4
                for i in range(additional):
                    sequence.append([[0], [0], [0], [0]])
            prov.append(np.array(sequence))
        reference_sequences = np.array(prov)

    print('\nreference_sequences:\n', one_hot.decode(reference_sequences, info, True))
    print('\nconsensus_sequences:\n', one_hot.decode(consensus_sequences, info))

    if len(consensus_sequences) != len(reference_sequences):
        print(
            'number of consensus sequences and reference sequences mismatch, correction phasing rate can\'t be '
            'calculated')
        return
    distances = np.zeros((len(consensus_sequences), len(reference_sequences)))
    for cs_index, consensus_sequence in enumerate(consensus_sequences):
        for ref_index, reference_sequence in enumerate(reference_sequences):
            if reverse:
                distances[cs_index][ref_index] = hamming_distance(reference_sequence, consensus_sequence)
            else:
                distances[cs_index][ref_index] = hamming_distance(consensus_sequence, reference_sequence, )

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

    if data == 'per_gene':
        cpr_per_gene_and_strain = np.zeros((len(reference_sequences)))
        name_of_reference_strains = ['HXB2', '89.6', 'JRCSF', 'NL43', 'YU2']
        print('CPR per gene:')
        for i in range(5):
            cpr_per_gene_and_strain[i] = 1 - (distances[i][min_indexes[i]] / len(reference_sequence))
            print(name_of_reference_strains[i], cpr_per_gene_and_strain[i])

        # TODO print percentage of reads per strain
    cpr = 1 - (min_sum / (len(consensus_sequences) * reference_sequences.shape[1]))
    return min_sum, min_indexes, cpr
