import os

import numpy as np
import pysam
import yaml

with open('./config.yml', 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)
    data = config['data']


def prepare_ref_for_bwa():
    # TODO think if this is needed?

    #     if config['real_data']:
    #         reference_sequence_path = config['haplotype_file_path']
    #     else:
    #         reference_sequence_path = config['haplotype_file_path_created']
    #     reference_file = open(reference_sequence_path, 'r')
    #     reference_sequence = None
    #     while reference_sequence is None:
    #         if 'HXB2' in reference_file.readline():
    #             reference_sequence = reference_file.readline()
    #     return reference_sequence

    index_cmd = 'bwa index -a bwtsw ' + config[config['data']]['hxb2_path']
    os.system(index_cmd)
    return


def clean_and_reformat_sam(cluster_index):
    # TODO split into file with \n and remove where mapping score is too low
    #  not sure if pysam is needed
    pysam.sort("-o", config[config['data']]['aligned_path'] + str(cluster_index) + '.bam',
               config[config['data']]['aligned_path'] + str(cluster_index) + '.sam')
    samfile = pysam.AlignmentFile(config[config['data']]['aligned_path'] + str(cluster_index) + '.bam', "rb")
    for read in samfile.fetch('HXB2'):
        print(read)

    samfile.close()
    print(samfile)


def align(cluster_index):
    read_file = config[config['data']]['cluster_path'] + str(cluster_index) + '.fastq'
    aligned_file = config[config['data']]['aligned_path'] + str(cluster_index) + '.sam'
    reference_file = config[config['data']]['hxb2_path']
    cmd = 'bwa mem -t 4 ' + reference_file + ' ' + read_file + ' > ' + aligned_file
    os.system(cmd)
    # clean_and_reformat_sam(cluster_index)
    return


def align_reads(clustered_reads):
    for reads in clustered_reads:
        reads_string = ''
        for read in reads:
            print(read)
            reads_string += read + '\n'

    return None


def haplotype_alignment(reads, predicted_clusters, n_clusters):
    prepare_ref_for_bwa()
    print(np.count_nonzero(predicted_clusters == 0))
    # print(np.count_nonzero(predicted_clusters == 1))
    # print(np.count_nonzero(predicted_clusters == 2))
    clustered_reads = [[] for _ in range(n_clusters)]
    for i in range(len(reads)):
        clustered_reads[predicted_clusters[i]].append(reads[i])

    for cluster_index in range(len(clustered_reads)):
        with open(config[config['data']]['cluster_path'] + str(cluster_index) + '.fastq', "w") as cluster_fastq_file:
            cluster_fastq_file.write('\n'.join(clustered_reads[cluster_index]))
        align(cluster_index)
    print("done")
    # align_reads(clustered_reads[0])
    # print(clustered_reads)
