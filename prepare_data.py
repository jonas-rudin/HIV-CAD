import os

import one_hot
from helpers.config import get_config

config = get_config()
data = config['data']

silence = ''

# if config['verbose'] == 0:
#     silence = ' >/dev/null 2>&1'


if __name__ == '__main__':

    # # with open(config[data]['unaligned_ref_path'] + '.fasta') as file:
    # with open('./data/reference/REF.fasta') as file:
    #     names = []
    #     sequences = []
    #     for line in file:
    #         print(line)
    #         if line.find('>') == -1:
    #             sequences.append(line)
    #         else:
    #             names.append(line)
    #     named_sequences = zip(names, sequences)
    #     print(named_sequences)
    #     longest_sequence = max(sequences, key=len)
    #     longest_index = sequences.index(longest_sequence)
    #     longest_name = names[longest_index]
    #     for name, sequence in named_sequences:
    #         print(name)
    #         print(len(sequence))
    #         if name == longest_name:
    #             continue
    #         print('out')
    #         alignment = pairwise2.align.globalxx(longest_sequence, sequence, one_alignment_only=True)
    #         print(alignment)
    # exit(-1)
    read_file = config[data]['reads_path']
    reference_file = config['hxb2_path']
    mapped_reads_file = config[data]['mapped_reads_path']

    if data == 'created':
        reference_file = config[data]['og_path']
    # index reference file
    print(reference_file)
    bwa_index_cmd = 'bwa index -a bwtsw ' + reference_file + silence
    os.system(bwa_index_cmd)
    print('indexed')
    bwa_mem_cmd = 'bwa mem -t 4 ' + reference_file + ' ' + read_file + '.fastq > ' + read_file + '.sam' + silence
    os.system(bwa_mem_cmd)
    print('aligned')

    if data == 'illumina' or data == 'created':
        remove_low_quality_score_cmd = 'samtools view -Sq 59 -e \'length(seq)>150\' ' + read_file + '.sam > ' + mapped_reads_file
        os.system(remove_low_quality_score_cmd)
        print('only aligned with quality score over 60 and bp length greater than 150')

    else:
        # remove unmapped reads
        remove_unaligned_cmd = 'samtools view -Sq 59 -e \'length(seq)>250\' ' + read_file + '.sam > ' + mapped_reads_file
        os.system(remove_unaligned_cmd)
        print('only aligned with quality score over 60 and bp length greater than 250')
    one_hot.encode_fasta()
    print('Done')

    # one_hot.encode_fasta()
