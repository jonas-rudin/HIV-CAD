import numpy as np
import tensorflow as tf

from helpers.config import get_config

config = get_config()
batch_size = config['data']['batch_size']
path = config['data']['one_hot_path']


class CAEGenerator(tf.keras.utils.Sequence):

    def __init__(self, number_of_batches, file_names, predictions):
        self.file_names = file_names
        self.number_of_batches = number_of_batches
        self.predictions = predictions

    def __len__(self):
        return self.number_of_batches

    def __getitem__(self, idx):
        index = self.list_of_indexes[idx]
        np.load(path + str(index) + '.npy')
        x = np.load(path + str(index) + '.npy')
        y = x
        return x, y
        # return np.array([
        #     resize(imread('/content/all_images/' + str(file_name)), (80, 80, 3))
        #     for file_name in batch_x]) / 255.0, np.array(batch_y)
