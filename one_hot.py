from os.path import exists

import tensorflow as tf

from helpers.IO import save_tensor_file, save_file
from helpers.config import get_config

config = get_config()


def switcher(base):
    return {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3
    }.get(base, -1)


def reverse_switcher(base):
    return {
        0: 'A',
        1: 'C',
        2: 'G',
        3: 'T'
    }.get(base, 'N')


# encrypt reads according to switcher
def encode_read(position, read, length):
    switched_read = []
    for base in read:
        switched_read.append(switcher(base))
    # position read in list of length of haplotype
    positioned_read = [-1] * position
    positioned_read.extend(switched_read)
    positioned_read.extend(-1 for _ in range(length - len(switched_read) - position))
    # switched_read_tensor = tf.cast(positioned_read, tf.int32)

    positioned_read_short = positioned_read[:length]

    switched_read_tensor = tf.cast(positioned_read_short, tf.int32)
    # one_hot encode 0 -> [1,0,0,0], 1 -> [0,1,0,0] ... -1 -> [0,0,0,0]
    one_hot_read = tf.one_hot(switched_read_tensor, depth=4, dtype=tf.int8)
    return one_hot_read


def decode_read(sequence):
    # TODO
    #  decode one hot
    base_read = []
    for base in sequence:
        base_read.append(switcher(base.indexOf(1)))  # TODO don't think that's how it works...
    decoded_sequence = base_read.join('')
    #  to stirng get back
    return decoded_sequence


def encode_sam():
    # check if file already exists
    # if config['load'] and exists(config[config['data']]['one_hot_path'] + '_0.npy'):
    if config['load'] and exists(config[config['data']]['one_hot_path'] + '.npy'):
        print('reads are already one hot encoded')
        return

    # read reads
    one_hot_encoded_reads = []
    # one_hot_encoded_reads_25k = []
    # file_number = 0
    # file_index = 0
    print('reading file: 454_mapped.sam')
    # read file line by line and one hot encode reads
    with open(config[config['data']]['mapped_reads_path']) as file:
        counter = 0
        for line in file:
            # print(line.split('\t')[3])
            # print(line.split('\t')[9])
            # TODO remove for final
            if int(line.split('\t')[3]) > config['haplotype_length']:
                continue
            counter += 1

            one_hot_encoded_read = encode_read(int(line.split('\t')[3]), line.split('\t')[9],
                                               config['haplotype_length'])
            one_hot_encoded_reads.append(one_hot_encoded_read)
            # one_hot_encoded_reads_25k.append(one_hot_encoded_read)
            if counter % 25000 == 0:
                print(counter)
            #     print('read 25\'000 lines, saved in file: ' + str(file_number) + '.npy')
            #     one_hot_encoded_reads_tensor_25k = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads_25k),
            #                                                       axis=3)
            #     save_tensor_file(config[config['data']]['one_hot_path'] + '_' + str(file_number),
            #                      one_hot_encoded_reads_tensor_25k)
            #     file_number += 1
            #     one_hot_encoded_reads_25k = []
        # print('read remaining lines, saved in file: ' + str(file_number) + '.npy')
        # one_hot_encoded_reads_tensor_50k = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads_25k),
        #                                                   axis=3)
        # save_tensor_file(config[config['data']]['one_hot_path'] + '_' + str(file_number),
        #                  one_hot_encoded_reads_tensor_50k)
        # file_number += 1
    # convert to tensor

    one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads), axis=3)

    # save
    save_tensor_file(config[config['data']]['one_hot_path'], one_hot_encoded_reads_tensor)
    print('Reads are one hot encoded and saved.')
    # return file_index


def encode_fasta():
    if config['load'] and exists(config[config['data']]['ref_path'] + '.npy'):
        print('sequences are already one hot encoded')
        return

    one_hot_encoded_sequences = []
    names_of_variants = []
    print('reading file: REF.fasta')
    # read file line by line and one hot encode reads
    with open(config[config['data']]['ref_path'] + '.fasta') as file:
        counter = 0
        for line in file:
            if line.find('>') == -1:
                # TODO get postition from alignment
                one_hot_encoded_sequence = encode_read(0, line,
                                                       config['haplotype_length'])
                one_hot_encoded_sequences.append(one_hot_encoded_sequence)
                continue
            counter += 1

            one_hot_encoded_sequence = encode_read(int(line.split('\t')[3]), line.split('\t')[9],
                                                   config['haplotype_length'])
            one_hot_encoded_sequences.append(one_hot_encoded_sequence)
            # one_hot_encoded_reads_25k.append(one_hot_encoded_read)
            if counter % 25000 == 0:
                print(counter)
            #     print('read 25\'000 lines, saved in file: ' + str(file_number) + '.npy')
            #     one_hot_encoded_reads_tensor_25k = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads_25k),
            #                                                       axis=3)
            #     save_tensor_file(config[config['data']]['one_hot_path'] + '_' + str(file_number),
            #                      one_hot_encoded_reads_tensor_25k)
            #     file_number += 1
            #     one_hot_encoded_reads_25k = []
        # print('read remaining lines, saved in file: ' + str(file_number) + '.npy')
        # one_hot_encoded_reads_tensor_50k = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads_25k),
        #                                                   axis=3)
        # save_tensor_file(config[config['data']]['one_hot_path'] + '_' + str(file_number),
        #                  one_hot_encoded_reads_tensor_50k)
        # file_number += 1
    # convert to tensor

    one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_sequences), axis=3)

    # save
    save_tensor_file(config[config['data']]['ref_path'], one_hot_encoded_reads_tensor)
    print('Reads are one hot encoded and saved.')


def decode(encoded_sequences):
    decoded_sequences = []
    for i in range(encoded_sequences):
        decoded_sequences.append('>' + str(i))
        decoded_sequences.append(decode_read(encoded_sequences[i]))
    decoded_sequences_string = '\n'.join(decoded_sequences)
    save_file(config['output_path'], decoded_sequences_string)
    return decoded_sequences

# TODO create reads of length max length and return reads as done in create data
