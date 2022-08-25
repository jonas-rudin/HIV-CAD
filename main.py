import sys
from os.path import exists

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.python.keras import Model

import create_data
import one_hot_454
from haplotype_alignment import haplotype_alignment
from helpers.IO import load_tensor_file
from helpers.colors_coding import ColorCoding
from helpers.config import get_config
from models.autoencoder import Autoencoder
from models.layers.cluster import ClusteringLayer

config = get_config()

if __name__ == '__main__':
    # TODO print config and info
    print('Python version: {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print('Tensorflow version: {}'.format(tf.__version__))
    if config['data'] == 'experimental':
        print(f'Working with {ColorCoding.OKGREEN}454{ColorCoding.ENDC} reads')
        one_hot_454.encode()
    else:
        print(
            f'Working with {ColorCoding.OKBLUE}created{ColorCoding.ENDC} reads')
        one_hot_encoded_reads, reads = create_data.create_reads()
        # number_of_files = 1
    # TODO change

    # out = np.zeros((2, 4))
    # print(out)
    # out[0, 1] = 1
    # print(out)
    # out[0, 1] = 0
    # out[:, 0] = [1, 2]
    # print(out)
    #
    # exit(-1)

    # number_of_files = 1
    # get
    n_clusters = config['n_clusters']
    verbose = config['verbose']
    one_hot_encoded_reads = load_tensor_file(config[config['data']]['one_hot_path'])
    batch_size = int(np.ceil(one_hot_encoded_reads.shape[0] / 200))

    print(f'{ColorCoding.OKGREEN}Build Autoencoder Model{ColorCoding.ENDC}')

    # create autoencoder
    autoencoder = Autoencoder(one_hot_encoded_reads.shape[1:])
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    autoencoder.build(input_shape=one_hot_encoded_reads.shape)

    autoencoder.encoder.summary()
    autoencoder.decoder.summary()
    autoencoder.summary()
    print(one_hot_encoded_reads.shape[0])
    print(batch_size)

    # train autoencoder
    # for i in range(number_of_files):
    #     one_hot_encoded_reads = load_tensor_file(config[config['data']]['one_hot_path'], i)
    # checkpoint_path = "results/models/autoencoder"
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    if config['load'] and exists(config[config['data']]['weights_path'] + '.index'):

        print(f'{ColorCoding.OKGREEN}Loading weights{ColorCoding.ENDC}')
        autoencoder.load_weights(config[config['data']]['weights_path'])
    # autoencoder_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,
    #                                          verbose=config['verbose'], save_freq=10*batch_size)
    else:
        print(f'{ColorCoding.OKGREEN}Training Autoencoder{ColorCoding.ENDC}')
        autoencoder.fit(x=one_hot_encoded_reads, y=one_hot_encoded_reads,
                        epochs=1,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=config['verbose'])
        # callbacks=[autoencoder_checkpoint])

        print(f'{ColorCoding.OKGREEN}Saving Weights{ColorCoding.ENDC}')
        autoencoder.save_weights(config[config['data']]['weights_path'])

    # print(one_hot_encoded_reads[0])
    #
    # print(autoencoder.predict(x=tf.expand_dims(one_hot_encoded_reads[0], axis=0)))
    # print(one_hot_encoded_reads[0])
    # create clustering model
    print(f'{ColorCoding.OKGREEN}Build Clustering Model{ColorCoding.ENDC}')

    clustering_layer = ClusteringLayer(n_clusters=n_clusters, name='clustering')(autoencoder.encoder.output)
    cluster_model = Model(inputs=autoencoder.encoder.input, outputs=clustering_layer)
    cluster_model.summary()
    # train k-means

    kmeans_repetitions = 10
    print(
        f'{ColorCoding.OKGREEN}Initialize and Running KMeans{ColorCoding.ENDC} (' + str(kmeans_repetitions) + ' times)')
    # TODO according to: 2.3 clustering layer
    for i in range(kmeans_repetitions):
        kmeans = KMeans(n_clusters=n_clusters, n_init=30, verbose=0)  # TODO change to verbose
        predicted_clusters = kmeans.fit_predict(cluster_model.predict(one_hot_encoded_reads))
        # print(origin2haplotype(predicted_clusters, one_hot_encoded_reads, n_clusters))
        # exit(-1)
        haplotype_alignment(predicted_clusters=predicted_clusters, n_clusters=n_clusters)
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
