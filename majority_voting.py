from random import choice

import numpy as np

import performance


def last_majority_vote_alignment(clustered_reads, overall_reads_sum):
    clustered_reads_sum = clustered_reads.sum(axis=0)

    consensus_sequence = np.zeros((clustered_reads.shape[1], 4, 1), dtype=np.int)
    for i in range(clustered_reads_sum.shape[0]):
        if clustered_reads_sum[i].sum(axis=0) != 0:
            # print(clustered_reads.shape[1] / 16)
            # print(clustered_reads.shape[1] / 8)
            # if clustered_reads_sum[i].sum(axis=0) > clustered_reads.shape[1] / 8:
            consensus_sequence[i, np.argmax(clustered_reads_sum[i])] = 1
        elif 2 <= i < clustered_reads_sum.shape[0] - 2:
            sum_of_neighbours = 0
            for j in range(-2, 3):
                sum_of_neighbours += clustered_reads_sum[i + j].sum(axis=0)
            print(sum_of_neighbours)
            if sum_of_neighbours == 0:
                print('sum is zero')
                not_covered_neighbours = [i - 2, i - 1, i, i + 1, i + 2]
                for i in not_covered_neighbours:
                    if np.sum(overall_reads_sum[i][:]) != 0:
                        if len(np.where(overall_reads_sum[i][:] == np.max(overall_reads_sum[i][:]))[0]) == 1:
                            # if not set by reads -> choose max from all reads
                            consensus_sequence[i, np.argmax(overall_reads_sum[i])] = 1
                        else:
                            # if multiple are the same -> choose one of them at random
                            max_positions = np.where(overall_reads_sum[i][:] == max(overall_reads_sum[i][:]))[0]
                            consensus_sequence[i, choice(max_positions)] = 1
        # TODO check what if multiple the same count?
    return consensus_sequence


def majority_vote_alignment(clustered_reads, overall_reads_sum):
    clustered_reads_sum = clustered_reads.sum(axis=0)
    indexes_not_covered_by_reads = np.where(np.sum(clustered_reads_sum, axis=1) == 0)[0]
    consensus_sequence = np.zeros((clustered_reads.shape[1], 4, 1), dtype=np.int)
    for i in range(clustered_reads_sum.shape[0]):
        if clustered_reads_sum[i].sum(axis=0) != 0:
            # if clustered_reads_sum[i].sum(axis=0) > clustered_reads.shape[1] / 16:
            consensus_sequence[i, np.argmax(clustered_reads_sum[i])] = 1
    for i in indexes_not_covered_by_reads:
        if np.sum(overall_reads_sum[i][:]) != 0:
            if len(np.where(overall_reads_sum[i][:] == np.max(overall_reads_sum[i][:]))[0]) == 1:
                # if not set by reads -> choose max from all reads
                consensus_sequence[i, np.argmax(overall_reads_sum[i])] = 1
            else:
                # if multiple are the same -> choose one of them at random
                max_positions = np.where(overall_reads_sum[i][:] == max(overall_reads_sum[i][:]))[0]
                consensus_sequence[i, choice(max_positions)] = 1
    return consensus_sequence


def align_reads_per_cluster(reads, predicted_clusters, n_clusters, last=False):
    majority_consensus_sequences = []
    predicted_clusters_array = np.array(predicted_clusters)
    reads_sum = reads.sum(axis=0)
    for i in range(n_clusters):
        reads_of_cluster_n = reads[np.where(predicted_clusters_array == i)[0]]
        if last:
            majority_consensus_sequences.append(last_majority_vote_alignment(reads_of_cluster_n, reads_sum))
        else:
            majority_consensus_sequences.append(majority_vote_alignment(reads_of_cluster_n, reads_sum))

    # else:
    #     reads_sum = reads.sum(axis=0)
    #     for i in range(n_clusters):
    #         reads_of_cluster_n = reads[np.where(predicted_clusters_array == i)[0]]
    #         majority_consensus_sequences.append(majority_vote_alignment(reads_of_cluster_n, reads_sum))
    return majority_consensus_sequences


def assign_reads_to_best_fitting_consensus_sequence(n_clusters, consensus_sequences, one_hot_encoded_reads):
    predicted_clusters = np.zeros((one_hot_encoded_reads.shape[0], n_clusters), dtype=int)
    for read_index in range(one_hot_encoded_reads.shape[0]):
        hd = []
        for consensus_sequence in consensus_sequences:
            hd.append(performance.hamming_distance(one_hot_encoded_reads[read_index], consensus_sequence))

        predicted_clusters[read_index][np.argmin(hd)] = 1
    return predicted_clusters
