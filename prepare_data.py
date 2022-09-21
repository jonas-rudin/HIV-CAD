import os

import one_hot
from helpers.config import get_config

config = get_config()
data = config['data']

silence = ''

# if config['verbose'] == 0:
#     silence = ' >/dev/null 2>&1'


if __name__ == '__main__':
    one_hot.encode_fasta()
    exit(-1)
    read_file = config[data]['reads_path']
    reference_file = config['hxb2_path']
    mapped_reads_file = config[data]['mapped_reads_path']
    # index reference file
    bwa_index_cmd = 'bwa index -a bwtsw ' + reference_file + silence
    os.system(bwa_index_cmd)
    print('indexed')

    if data == 'illumina':
        # align to reference with bwa mem algorithm (Illuimina)
        bwa_mem_cmd = 'bwa mem -t 4 ' + reference_file + ' ' + read_file + '.fastq > ' + read_file + '.sam'  # + silence
        os.system(bwa_mem_cmd)
        print('illumina aligned')
        # remove unmapped reads
        # remove_unaligned_cmd = 'samtools view -F 4 ' + illumina_read_files + '.sam > ' + illumina_mapped_reads_file
        # os.system(remove_unaligned_cmd)
        # print('only aligned')
        remove_low_quality_score_cmd = 'samtools view -Sq 59 -e \'length(seq)>150\' ' + read_file + '.sam > ' + mapped_reads_file
        os.system(remove_low_quality_score_cmd)
        print('only aligned with quality score over 60 and bp length greater than 150')
    else:
        # align to reference with bwa mem algorithm (454)
        bwa_mem_cmd = 'bwa mem -t 4 ' + reference_file + ' ' + read_file + '.fastq > ' + read_file + '.sam' + silence
        os.system(bwa_mem_cmd)
        print('454 aligned')

        # remove unmapped reads
        remove_unaligned_cmd = 'samtools view -Sq 59 -e \'length(seq)>250\' ' + read_file + '.sam > ' + mapped_reads_file
        os.system(remove_unaligned_cmd)
        print('cleaned')
    one_hot.encode_fasta()

    # one_hot.encode_fasta()
