import sys

import numpy as np
import tensorflow as tf
import yaml
from sklearn.cluster import KMeans
from tensorflow.python.keras import Model

import create_data
import one_hot
from helpers.IO import load_tensor_file
from helpers.colors_coding import ColorCoding
from helpers.haplotype_alignment import haplotype_alignment
from models.autoencoder import Autoencoder
from models.layers.cluster import ClusteringLayer

with open('./config.yml', 'r') as ymlfile:
    config = yaml.safe_load(ymlfile)

if __name__ == '__main__':
    # TODO print config and info
    print('Python version: {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print('Tensorflow version: {}'.format(tf.__version__))
    if config['data'] == 'experimental':
        print(f'Working with {ColorCoding.OKGREEN}Illumina{ColorCoding.ENDC} reads')
        number_of_files = one_hot.encode()
        batch_size = int(np.ceil(100000 / 200))
    else:
        print(
            f'Working with {ColorCoding.OKBLUE}created{ColorCoding.ENDC} reads')
        one_hot_encoded_reads, reads = create_data.create_reads()
        number_of_files = 1
        batch_size = int(np.ceil(one_hot_encoded_reads.shape[0] / 2))
    # TODO change
    number_of_files = 1
    # get
    n_clusters = config['n_clusters']
    verbose = config['verbose']

    shape = [config['number_of_reads_per_file'], config['max_read_length'], 4, 1]
    print(shape)

    # create autoencoder
    autoencoder = Autoencoder(shape[1:])

    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    autoencoder.build(input_shape=shape)
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()
    autoencoder.summary()
    one_hot_encoded_reads = load_tensor_file(config[config['data']]['one_hot_path'], 0)
    print(one_hot_encoded_reads.shape)
    print(one_hot_encoded_reads[0].shape)

    # train autoencoder
    for i in range(number_of_files):
        one_hot_encoded_reads = load_tensor_file(config[config['data']]['one_hot_path'], i)
        autoencoder.fit(x=one_hot_encoded_reads, y=one_hot_encoded_reads,
                        epochs=5,
                        batch_size=batch_size,
                        shuffle=False,
                        verbose=config['verbose'])

    autoencoder.predict(x=tf.expand_dims(one_hot_encoded_reads[0], axis=0))
    print(one_hot_encoded_reads[0])
    # create clustering model
    clustering_layer = ClusteringLayer(n_clusters=n_clusters, name='clustering')(autoencoder.encoder.output)
    cluster_model = Model(inputs=autoencoder.encoder.input, outputs=clustering_layer)

    # train k-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=30, verbose=verbose)
    predicted_clusters = kmeans.fit_predict(cluster_model.predict(one_hot_encoded_reads))
    print(predicted_clusters)
    haplotype_alignment(reads=reads, predicted_clusters=predicted_clusters, n_clusters=n_clusters)
    # Todo MEC stuff and rebuild + evaluate
    output_cluster = cluster_model.predict(x=tf.expand_dims(one_hot_encoded_reads[0], axis=0))
    output_encoder = autoencoder.predict(x=tf.expand_dims(one_hot_encoded_reads[0], axis=0))
    print("output_cluster", output_cluster)
    print("evaluate")
    autoencoder.evaluate(x=one_hot_encoded_reads, y=one_hot_encoded_reads, verbose=verbose)

    if config['save']:
        if config['data'] == 'experimental':
            autoencoder.save('./results/models/454_weights')
        else:
            autoencoder.save(
                './results/models/created_weights_' + str(config['number_of_strains']) + '_' + str(
                    config['read_length']) + '_' + str(config['min_number_of_reads_per_strain']))

    print(one_hot_encoded_reads[0].shape)
    print(tf.squeeze(output_encoder, [0]).shape)
