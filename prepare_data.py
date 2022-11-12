import os
import time

import one_hot
from helpers.config import get_config

config = get_config()
data = config['data']

silent = ''


# if config['verbose'] == 0:
#     silent = ' >/dev/null 2>&1'


def prepare_data():
    if data == 'created':
        read_file = config['created']['reads_path'] + '_' + str(config[data]['n_clusters']) + '_' + str(
            config[data]['coverage']) + '_' + str(config[data]['read_length']) + '_' + str(
            config[data]['sequencing_error'])
        reference_file = config[data]['longest_path'] + '_' + str(config[data]['n_clusters']) + '.fasta'
        mapped_reads_file = config[data]['mapped_reads_path'] + '_' + str(config[data]['n_clusters']) + '_' + str(
            config[data]['coverage']) + '_' + str(config[data]['read_length']) + '_' + str(
            config[data]['sequencing_error'])

    else:
        read_file = config[data]['reads_path']
        reference_file = config[data]['hxb2_path']
        mapped_reads_file = config[data]['mapped_reads_path']

    # index reference file
    bwa_index_cmd = 'bwa index -a bwtsw ' + reference_file + silent
    os.system(bwa_index_cmd)
    print('indexed')
    bwa_mem_cmd = 'bwa mem -t 4 ' + reference_file + ' ' + read_file + '.fastq > ' + read_file + '.sam' + silent
    os.system(bwa_mem_cmd)
    print('aligned')

    if config[data]['source'] == 'llumina' or (data == 'created' and config[data]['read_length']):
        remove_low_quality_score_cmd = 'samtools view -Sq 59 -e \'length(seq)>150\' ' + read_file + '.sam > ' + mapped_reads_file + '.sam'
        os.system(remove_low_quality_score_cmd)
        print('only aligned with quality score over 60 and bp length greater than 150')
    else:
        # remove unmapped reads
        remove_unaligned_cmd = 'samtools view -Sq 59 -e \'length(seq)>300\' ' + read_file + '.sam > ' + mapped_reads_file + '.sam'
        os.system(remove_unaligned_cmd)
        print('only aligned with quality score over 60 and bp length greater than 250')

    time.sleep(1)
    one_hot.encode_fasta()
    print('Data Prepared')

    # one_hot.encode_fasta()


if __name__ == "__main__":
    prepare_data()
