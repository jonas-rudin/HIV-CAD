import numpy as np

import performance


# old shape
# def majority_vote_alignment(clustered_reads):
#     clustered_reads_sum = clustered_reads.sum(axis=0)
#     consensus_sequence = np.zeros((clustered_reads.shape[1], 4, 1), dtype=np.int)
#     for i in range(clustered_reads_sum.shape[0]):
#         if clustered_reads_sum[i].sum(axis=0) != 0:
#             consensus_sequence[i, np.argmax(clustered_reads_sum[i])] = 1
#         # TODO check what if multiple the same count?
#     return consensus_sequence

# Backup....
# def majority_vote_alignment(clustered_reads):
#     for i in range(len(clustered_reads)):
#         clustered_reads[i][0][0] = 0
#         clustered_reads[i][1][0] = 0
#         clustered_reads[i][2][0] = 0
#         clustered_reads[i][3][0] = 0
#     clustered_reads_sum = clustered_reads.sum(axis=0)
#
#     consensus_sequence = np.zeros((4, clustered_reads.shape[2], 1), dtype=np.int)
#
#     indexes_not_covered_by_reads = np.where(np.sum(clustered_reads_sum, axis=0) == 0)[0]
#
#     print('indexes_not_covered_by_reads', indexes_not_covered_by_reads)
#     max_clustered_read_indexes = np.argmax(clustered_reads_sum, axis=0).sum(axis=1)
#     # print(max_clustered_read_indexes)
#     for cluster_index in range(clustered_reads_sum.shape[0]):
#         indexes = list(np.where(max_clustered_read_indexes == cluster_index))
#         print('indexes', indexes)
#
#         if cluster_index == 0:
#             indexes = np.array([i for i in indexes if i not in indexes_not_covered_by_reads])
#         print('1', indexes)
#         print('2', cluster_index)
#         print(indexes)
#         if len(indexes) != 0:
#             consensus_sequence[cluster_index, indexes] = 1
#     # print(consensus_sequence)
#     # exit(-1)
#     print('consensus_sequence', consensus_sequence)
#     exit(-1)
#     # TODO check what if multiple the same count?
#     return consensus_sequence

def majority_vote_alignment(clustered_reads):
    clustered_reads_sum = clustered_reads.sum(axis=0)

    consensus_sequence = np.zeros((4, clustered_reads.shape[2], 1), dtype=np.int)

    indexes_not_covered_by_reads = np.where(np.sum(clustered_reads_sum, axis=0) == 0)[0]
    max_clustered_read_indexes = np.argmax(clustered_reads_sum, axis=0).sum(axis=1)
    # print(max_clustered_read_indexes)
    for cluster_index in range(clustered_reads_sum.shape[0]):
        indexes = list(np.where(max_clustered_read_indexes == cluster_index))
        if cluster_index == 0:
            indexes = np.array([i for i in indexes[0] if i not in indexes_not_covered_by_reads])
        if len(indexes) != 0:
            consensus_sequence[cluster_index, indexes] = 1

        # for j in range(len(indexes_not_covered_by_reads)):
        #     if len(np.where(
        #             clustered_reads_sum[:, indexes_not_covered_by_reads[j]] == max(
        #                 clustered_reads_sum[:, indexes_not_covered_by_reads[j]]))[
        #                0]) != 1:  # if not covered, select the most dominant one based on 'ACGTcount'
        #         tem = np.where(clustered_reads_sum[:, indexes_not_covered_by_reads[j]] == max(
        #             clustered_reads_sum[:, indexes_not_covered_by_reads[j]]))[0]
        #         consensus_sequence[cluster_index, indexes_not_covered_by_reads[j]] = tem[int(np.floor(
        #             random.random() * len(tem)))] + 1
        #     else:
        #         consensus_sequence[cluster_index, indexes_not_covered_by_reads[j]] = np.argmax(
        #             clustered_reads_sum[:, indexes_not_covered_by_reads[j]]) + 1

    # print(consensus_sequence)
    # exit(-1)
    return consensus_sequence


def align_reads_per_cluster(reads, predicted_clusters, n_clusters):
    majority_consensus_sequences = []
    predicted_clusters_array = np.array(predicted_clusters)
    for i in range(n_clusters):
        reads_of_cluster_n = reads[np.where(predicted_clusters_array == i)[0]]
        majority_consensus_sequences.append(majority_vote_alignment(reads_of_cluster_n))
    return majority_consensus_sequences


def assign_reads_to_best_fitting_consensus_sequence(n_clusters, consensus_sequences, one_hot_encoded_reads):
    # counter = 0
    # for n in consensus_sequences:
    #     print('consensus', counter)
    #     counter += 1
    #
    #     for i in n:
    #         line = []
    #         for j in i:
    #             line.append(j[0])
    #         print(line)
    # for i in one_hot_encoded_reads:
    #     print('new read')
    #     for e in i:
    #         line = []
    #         for j in e:
    #             line.append(j[0])
    #         print(line)
    predicted_clusters = np.zeros((one_hot_encoded_reads.shape[0], n_clusters), dtype=int)
    for read_index in range(one_hot_encoded_reads.shape[0]):
        hd = []
        for consensus_sequence in consensus_sequences:
            hd.append(performance.hamming_distance(one_hot_encoded_reads[read_index], consensus_sequence))

        predicted_clusters[read_index][np.argmin(hd)] = 1
    # print(predicted_clusters)
    # exit(-1)
    return predicted_clusters
