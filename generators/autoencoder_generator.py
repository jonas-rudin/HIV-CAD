import numpy as np
import tensorflow as tf

from helpers.config import get_config

config = get_config()
batch_size = config[config['data']]['batch_size']
path = config[config['data']]['one_hot_path']


# TODO on epoch end

class AutoencoderGenerator(tf.keras.utils.Sequence):

    def __init__(self, file_names, number_of_batches):
        self.file_names = file_names
        self.number_of_batches = number_of_batches

    def __len__(self):
        return self.number_of_batches

    def __getitem__(self, idx):
        index = self.file_names[idx]
        x = np.load(path + str(index) + '.npy')
        y = x
        return np.array(x), np.array(y)
        # return np.array([
        #     resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
        #     for file_name in batch_x]) / 255.0, np.array(batch_y)
