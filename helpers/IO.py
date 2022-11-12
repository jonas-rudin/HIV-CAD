import json

import numpy as np
import yaml

with open('./config.yml', 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)
    data = config['data']


def save_text(path_to_file, text):
    print('saving files...')
    with open(path_to_file, "w") as fasta_file:
        fasta_file.write(text)
    return


def save_tensor_file(file_path, content):
    print('saving .npy file ')
    np.save(file_path, content)


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
