import json

import numpy as np
import yaml

with open('./config.yml', 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)
    data = config['data']


# TODO split into save/load general files and .npy
# saves reads and one hot encoded reads to files
# def save(path_to_result_file_one_hot, one_hot_encoded_reads, path_to_result_file_reads, reads):
#     print('saving files...')
#     np.save(path_to_result_file_one_hot, one_hot_encoded_reads)
#     with open(path_to_result_file_reads, "w") as fasta_file:
#         fasta_file.write(reads)
#     return

def save_text(path_to_file, text):
    print('saving files...')
    # np.save(path_to_result_file_one_hot, one_hot_encoded_reads)
    with open(path_to_file, "w") as fasta_file:
        fasta_file.write(text)
    return


def save_tensor_file(file_path, content):
    print('saving .npy file ')
    np.save(file_path, content)


# def save_tensor_file(file_path, index, content):
#     print('saving .npy file ' + str(index))
#     np.save(file_path + '_' + str(index) + '.npy', content)


def save_file(file_path, content):
    print('saving fastq file...')
    with open(file_path, "w") as file:
        file.write(content)


# loads reads and one hot encoded reads from files
def load(path_to_result_file_one_hot, path_to_result_file_reads):
    print('loading files...')
    one_hot_encoded_reads = np.load(path_to_result_file_one_hot + '.npy')
    with open(path_to_result_file_reads, "r") as json_file:
        reads = json.load(json_file)
    return one_hot_encoded_reads, reads


def load_tensor_file(file_path):
    print('loading ' + file_path + '.npy file')
    return np.load(file_path + '.npy')


# def load_tensor_file(file_path, index):
#     print('loading: ' + file_path + ' _ ' + str(index) + '.npy')
#     return np.load(file_path + '_' + str(index) + '.npy')


def load_fastq_file_as_list(file_path):
    print('loading fastq file')
    reads = []

    if config['data'] == 'experimental':
        if config['number_of_spots'] == 3000:
            with open(file_path) as file:
                for i in range(config['number_of_spots']):
                    reads.append(file.readline() + file.readline() + file.readline() + file.readline())
        else:
            with open(file_path) as file:
                while True:
                    header = file.readline()
                    if header:
                        reads.append(header + file.readline() + file.readline() + file.readline())
                    else:
                        break

    else:
        with open(file_path) as file:
            reads.append(file.readline() + file.readline() + file.readline() + file.readline())
    return reads

# def save_tfrecord(path, content):
#     print(content.shape)
#     serialized_content = tf.io.serialize_tensor(content)
#     print(tf.io.parse_tensor(serialized_content.numpy(), out_type=tf.float32).shape)
#     with tf.io.TFRecordWriter(path) as writer:
#         writer.write(serialized_content.numpy())
#
#
# def parse_tensor_int8(x):
#     return tf.io.parse_tensor(x, tf.float32)
#
#
# def load_tfrecord(path):
#     with open(path, 'rb') as reader:
#         serialized_content = reader.read()
#     return tf.io.parse_tensor(serialized_content)
