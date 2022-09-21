import numpy as np


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
    reference_sequences = load_tensor_file()
    return

    # res = 0
    #
    # for i in range(len(SNVmatrix)):
    #     dis = [hamming_distance(SNVmatrix[i, :], Recovered_Haplo[j, :]) for j in range(len(Recovered_Haplo))]
    #     res += min(dis)
    #
    # return res
