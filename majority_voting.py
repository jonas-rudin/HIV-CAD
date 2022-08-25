import random

import numpy as np


def ACGT_count(submatrix):
    out = np.zeros((submatrix.shape[0], 4))

    print(out.shape)
    print(out)
    print(submatrix.shape)
    print(submatrix[0])
    print(submatrix[0][0])

    # print(out)

    # for i in range(4):
    i = 0

    # out (submatrix == (i + 1)).sum(axis=1))
    # print(submatrix)
    # out[:, i] = (submatrix == (i + 1)).sum(axis=0)

    return out


def origin2haplotype(origins, SNVmatrix, n_cluster):
    V_major = np.zeros((n_cluster, SNVmatrix.shape[0]))  # majority voting result
    ACGTcount = ACGT_count(SNVmatrix)

    for i in range(n_cluster):
        reads_single = SNVmatrix[origins == i, :]  # all reads from one haplotypes
        single_sta = np.zeros((SNVmatrix.shape[0], 4))

        if len(reads_single) != 0:
            single_sta = ACGT_count(reads_single)  # ACGT statistics of a single nucleotide position
        V_major[i, :] = np.argmax(single_sta, axis=1) + 1

        uncov_pos = np.where(np.sum(single_sta, axis=1) == 0)[0]

        for j in range(len(uncov_pos)):
            if len(np.where(ACGTcount[uncov_pos[j], :] == max(ACGTcount[uncov_pos[j], :]))[
                       0]) != 1:  # if not covered, select the most doninant one based on 'ACGTcount'
                tem = np.where(ACGTcount[uncov_pos[j], :] == max(ACGTcount[uncov_pos[j], :]))[0]
                V_major[i, uncov_pos[j]] = tem[int(np.floor(random.random() * len(tem)))] + 1
            else:
                V_major[i, uncov_pos[j]] = np.argmax(ACGTcount[uncov_pos[j], :]) + 1

    return V_major

# def U2V(U, M_E, R=ploidy_candidate, hap_len=len_haplo):
#     min_index = np.argmax(U, axis=1)
#     V_major = np.zeros((R, hap_len))  # majority voting result
#     ACGTcount = ACGT_count(M_E)
#
#     for i in range(R):
#         reads_single = M_E[min_index == i, :]  # all reads from one haplotypes
#         single_sta = np.zeros((hap_len, 4))
#
#         if len(reads_single) != 0:
#             single_sta = ACGT_count(reads_single)  # ACGT statistics of a single nucleotide position
#         V_major[i, :] = np.argmax(single_sta, axis=1) + 1.0
#
#         uncov_pos = np.where(np.sum(single_sta, axis=1) == 0)[0]
#
#         for j in range(len(uncov_pos)):
#             if len(np.where(ACGTcount[uncov_pos[j], :] == max(ACGTcount[uncov_pos[j], :]))[
#                        0]) != 1:  # if not covered, select the most doninant one based on 'ACGTcount'
#                 tem = np.where(ACGTcount[uncov_pos[j], :] == max(ACGTcount[uncov_pos[j], :]))[0]
#                 V_major[i, uncov_pos[j]] = tem[int(np.floor(random.random() * len(tem)))] + 1
#             else:
#                 V_major[i, uncov_pos[j]] = np.argmax(ACGTcount[uncov_pos[j], :]) + 1
#
#     return V_major
