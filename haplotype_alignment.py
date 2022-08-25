import os

import numpy as np
import pysam

from helpers.IO import load_fastq_file_as_list
from helpers.config import get_config

config = get_config()
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

    bwa_index_cmd = 'bwa index -a bwtsw ' + config[config['data']]['hxb2_path']
    os.system(bwa_index_cmd)
    return


def get_consensus_sequence(cluster_index):
    samtools_consensus_cmd = 'samtools consensus -f fastq ' + config[config['data']]['aligned_path'] + str(
        cluster_index) + '.bam -o cons.fq'
    os.system(samtools_consensus_cmd)


#     samtools mpileup -uf REFERENCE.fasta SAMPLE.bam | bcftools call -c | vcfutils.pl vcf2fq > SAMPLE_cns.fastq
#
#   # vcfutils.pl is part of bcftools
#
# # Convert .fastq to .fasta and set bases of quality lower than 20 to N
# seqtk seq -aQ64 -q20 -n N SAMPLE_cns.fastq > SAMPLE_cns.fasta

def clean_sam_file(cluster_index):
    # TODO  remove where mapping score is too low
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
    # align to reference with bwa mem algorithm
    bwa_mem_cmd = 'bwa mem -t 4 ' + reference_file + ' ' + read_file + ' > ' + aligned_file
    os.system(bwa_mem_cmd)
    # transform sam to bam
    file_name = config[config['data']]['aligned_path'] + str(cluster_index)
    # samtools_view_cmd = 'samtools view -S -b ' + file_name + '.sam > ' + file_name + '.bam'
    # os.system(samtools_view_cmd)
    # get_consensus_sequence(cluster_index)
    # sort bam file
    samtools_sort_cmd = 'samtools sort ' + file_name + '.sam -o ' + file_name + '.bam'
    os.system(samtools_sort_cmd)

    # find consensus sequence
    samtools_consensus_cmd = 'samtools consensus -f fastq ' + file_name + '.bam -o ' + file_name + '.fq'
    os.system(samtools_consensus_cmd)
    # clean_sam_file(cluster_index)

    return


def haplotype_alignment(predicted_clusters, n_clusters):
    prepare_ref_for_bwa()
    print(np.count_nonzero(predicted_clusters == 0))
    print(np.count_nonzero(predicted_clusters == 1))
    print(np.count_nonzero(predicted_clusters == 2))
    print(np.count_nonzero(predicted_clusters == 3))
    print(np.count_nonzero(predicted_clusters == 4))

    clustered_reads = [[] for _ in range(n_clusters)]
    # reads = load_fastq_file_as_list(config[config['data']]['cluster_path'])
    reads = load_fastq_file_as_list(config[config['data']]['reads_path'])
    # TODO load read file
    for i in range(len(reads)):
        clustered_reads[predicted_clusters[i]].append(reads[i])

    for cluster_index in range(len(clustered_reads)):
        with open(config[config['data']]['cluster_path'] + str(cluster_index) + '.fastq', "w") as cluster_fastq_file:
            cluster_fastq_file.write('\n'.join(clustered_reads[cluster_index]))
        align(cluster_index)
        # sam_to_bam(cluster_index)
    print("done")

    # print(clustered_reads)
