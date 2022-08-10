import sys

import numpy as np
import tensorflow as tf
import yaml
from sklearn.cluster import KMeans
from tensorflow.python.keras import Model

import create_data
import one_hot
from helpers.colors_coding import ColorCoding
from models.autoencoder import Autoencoder
from models.layers.cluster import ClusteringLayer

with open('./config.yml', 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

if __name__ == '__main__':
    # TODO print config and info
    print('Python version: {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print('Tensorflow version: {}'.format(tf.__version__))
    if config['real_data']:
        print(f'Working with {ColorCoding.OKGREEN}454{ColorCoding.ENDC} reads')
        data = one_hot.encode()
        batch_size = int(np.ceil(data.shape[0] / 200))

    else:
        print(
            f'Working with {ColorCoding.OKBLUE}created{ColorCoding.ENDC} reads')
        data = create_data.create_reads()
        batch_size = int(np.ceil(data.shape[0] / 2))

    # get
    n_clusters = config['n_clusters']
    verbose = config['verbose']
    
    # print("1", data[0])
    # print("2", data[0][0])
    # print("3", data[0][0][0])
    # print("4", data[0][0][0][0])

    # create autoencoder
    autoencoder = Autoencoder(data.shape[1:])

    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    autoencoder.build(input_shape=data.shape)
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()
    autoencoder.summary()
    print(data.shape)

    # train autoencoder
    autoencoder.fit(x=data, y=data,
                    epochs=100,
                    batch_size=batch_size,
                    shuffle=False,
                    verbose=0)

    # create clustering model
    clustering_layer = ClusteringLayer(n_clusters=n_clusters, name='clustering')(autoencoder.encoder.output)
    cluster_model = Model(inputs=autoencoder.encoder.input, outputs=clustering_layer)

    # train k-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=30, verbose=verbose)
    y_pred_kmeans = kmeans.fit_predict(cluster_model.predict(data))

    # Todo MEC stuff and rebuild + evaluate
    output_cluster = cluster_model.predict(x=tf.expand_dims(data[0], axis=0))
    output_encoder = autoencoder.predict(x=tf.expand_dims(data[0], axis=0))
    print("output_cluster", output_cluster)
    print("evaluate")
    autoencoder.evaluate(x=data, y=data, verbose=verbose)

    if config['save']:
        if config['real_data']:
            autoencoder.save('./results/models/454_weights')
        else:
            autoencoder.save(
                './results/models/created_weights_' + str(config['number_of_strains']) + '_' + str(
                    config['read_length']) + '_' + str(config['min_number_of_reads_per_strain']))

    print(data[0].shape)
    print(tf.squeeze(output_encoder, [0]).shape)
