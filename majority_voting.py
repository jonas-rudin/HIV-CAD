import numpy as np


def majority_vote_alignment(clustered_reads):
    clustered_reads_sum = clustered_reads.sum(axis=0)
    consensus_sequence = np.zeros((clustered_reads.shape[1], 4, 1), dtype=np.int)
    for i in range(clustered_reads_sum.shape[0]):
        consensus_sequence[i, np.argmax(clustered_reads_sum[i])] = 1
    return consensus_sequence


def align_reads_per_cluster(reads, predicted_clusters, n_clusters):
    majority_consensus_sequences = []
    predicted_clusters_array = np.array(predicted_clusters)
    for i in range(n_clusters):
        reads_of_cluster_n = reads[np.where(predicted_clusters_array == i)[0]]
        majority_consensus_sequences.append(majority_vote_alignment(reads_of_cluster_n))
    return majority_consensus_sequences
