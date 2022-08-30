import os

from helpers.config import get_config

config = get_config()
silence = ''

# if config['verbose'] == 0:
#     silence = ' >/dev/null 2>&1'


if __name__ == '__main__':
    read_file = config[config['data']]['reads_path']
    reference_file = config[config['data']]['hxb2_path']
    mapped_reads_file = config[config['data']]['mapped_reads_path']

    # index reference file
    bwa_index_cmd = 'bwa index -a bwtsw ' + reference_file + silence
    os.system(bwa_index_cmd)
    print('indexed')

    # align to reference with bwa mem algorithm
    bwa_mem_cmd = 'bwa mem -t 4 ' + reference_file + ' ' + read_file + '.fastq > ' + read_file + '.sam' + silence
    os.system(bwa_mem_cmd)
    print('aligned')

    # remove unmapped reads
    remove_unaligned_cmd = 'samtools view -F 4 ' + read_file + '.sam > ' + mapped_reads_file
    os.system(remove_unaligned_cmd)
    print('cleaned')

    # create_complementary_of
