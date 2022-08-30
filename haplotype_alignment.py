import os

import numpy as np
import pysam

from helpers.IO import load_fastq_file_as_list
from helpers.config import get_config

config = get_config()
data = config['data']
silence = ''


# if config['verbose'] == 0:
#     silence = ' >/dev/null 2>&1'


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

    bwa_index_cmd = 'bwa index -a bwtsw ' + config[config['data']]['hxb2_path'] + silence
    os.system(bwa_index_cmd)
    return


def get_consensus_sequence(cluster_index):
    samtools_consensus_cmd = 'samtools consensus -f fastq ' + config[config['data']]['aligned_path'] + str(
        cluster_index) + '.bam -o cons.fq' + silence
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


def ngs_alignment(path):
    clustered_read_file = config[config['data']]['cluster_path'] + path + '.fastq'
    aligned_file = config[config['data']]['aligned_path'] + config['alignment'] + '/' + path
    reference_file = config[config['data']]['hxb2_path']
    # align to reference with bwa mem algorithm
    bwa_mem_cmd = 'bwa mem -t 4 ' + reference_file + ' ' + clustered_read_file + ' > ' + aligned_file + '.sam' + silence
    os.system(bwa_mem_cmd)
    # transform sam to bam
    # samtools_view_cmd = 'samtools view -S -b ' + file_name + '.sam > ' + file_name + '.bam'
    # os.system(samtools_view_cmd)
    # get_consensus_sequence(cluster_index)
    # sort sam and transform to bam file
    samtools_sort_cmd = 'samtools sort ' + aligned_file + '.sam -o ' + aligned_file + '.bam' + silence
    os.system(samtools_sort_cmd)

    # find consensus sequence
    samtools_consensus_cmd = 'samtools consensus -f fastq ' + aligned_file + '.bam -o ' + aligned_file + '.fq' + silence
    os.system(samtools_consensus_cmd)
    # transform fastq to fasta and replace all base with a quality score below 20
    seqtk_cmd = 'seqtk seq -aQ64 -q20 -n N ' + aligned_file + '.fq > ' + aligned_file + '.fasta'
    os.system(seqtk_cmd)
    return


def de_novo_alignment(path):
    read_file = config[config['data']]['cluster_path'] + path + '.fastq'
    output_folder = config[config['data']]['aligned_path'] + config['alignment'] + '/'
    spades_cmd = 'python ../tools/SPAdes-3.12.0-Darwin/bin/spades.py -s ' + read_file + ' -o ' + output_folder \
                 + ' -m 1024' + silence  # -m flag memory necessary for M1
    os.system(spades_cmd)
    # TODO find where final data is stored and clean it
    return


def haplotype_alignment(kmeans_rep, predicted_clusters, n_clusters):
    prepare_ref_for_bwa()
    print('kmeans_rep', kmeans_rep)
    print('n_clusters', n_clusters)

    clustered_reads = [[] for _ in range(n_clusters)]
    # reads = load_fastq_file_as_list(config[config['data']]['cluster_path'])
    reads = load_fastq_file_as_list(config[config['data']]['reads_path'])
    # TODO load read file
    for i in range(len(reads)):
        clustered_reads[predicted_clusters[i]].append(reads[i])
    folder_path = str(n_clusters) + '_cluster/kmeans_rep_' + str(kmeans_rep) + '/'
    for cluster_index in range(len(clustered_reads)):
        print('cluster ' + str(cluster_index) + ': ' + str(
            np.count_nonzero(predicted_clusters == cluster_index)) + ' reads')

        with open(config[config['data']]['cluster_path'] + folder_path + str(cluster_index) + '.fastq',
                  "w") as cluster_fastq_file:
            cluster_fastq_file.write('\n'.join(clustered_reads[cluster_index]))
        if config['alignment'] == 'de_novo_alignment':
            de_novo_alignment(folder_path + str(cluster_index))
        else:
            ngs_alignment(folder_path + str(cluster_index))
        # sam_to_bam(cluster_index)
    print("done")

    # print(clustered_reads)
