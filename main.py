import pickle
import sys
from os.path import exists

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.losses import MeanSquaredError, KLD

import create_data
import majority_voting
import one_hot
import performance
from helpers.IO import load_tensor_file
from helpers.colors_coding import ColorCoding
from helpers.config import get_config
from models.autoencoder import get_autoencoder_key_points
from models.layers.cluster import ClusteringLayer

config = get_config()

if __name__ == '__main__':
    # TODO print config and info
    print('Python version: {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print('Tensorflow version: {}'.format(tf.__version__))
    if config['data'] == 'experimental':
        print(f'Working with {ColorCoding.OKGREEN}454{ColorCoding.ENDC} reads')
        one_hot.encode_sam()
    else:
        print(
            f'Working with {ColorCoding.OKBLUE}created{ColorCoding.ENDC} reads')
        one_hot_encoded_reads, reads = create_data.create_reads()
        # number_of_files = 1

    n_clusters = config['n_clusters']
    verbose = config['verbose']
    one_hot_encoded_reads = load_tensor_file(config[config['data']]['one_hot_path'])
    number_of_batches = 200
    batch_size = int(np.ceil(one_hot_encoded_reads.shape[0] / number_of_batches))
    # predicted_clusters = [i % 2 for i in range(one_hot_encoded_reads.shape[0])]
    # consensus_sequence = majority_voting.align_reads_per_cluster(one_hot_encoded_reads, predicted_clusters, 2)
    # performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequence)
    # exit(-1)

    # create autoencoder
    model_input, encoder_output, decoder_output = get_autoencoder_key_points(one_hot_encoded_reads.shape[1:])
    autoencoder = Model(inputs=model_input, outputs=decoder_output, name='autoencoder')
    # autoencoder = Autoencoder(one_hot_encoded_reads.shape[1:])
    autoencoder.compile(optimizer='adam', loss=MeanSquaredError())
    autoencoder.build(input_shape=one_hot_encoded_reads.shape)

    autoencoder.summary()

    print('reads tensor shape:', one_hot_encoded_reads.shape)
    print('number of reads:', one_hot_encoded_reads.shape[0])
    print('batch_size:', batch_size)

    # train autoencoder
    if config['load'] and exists(config[config['data']]['weights_path'] + '.index'):
        print(f'{ColorCoding.OKGREEN}Loading weights{ColorCoding.ENDC}')
        autoencoder.load_weights(config[config['data']]['weights_path'])

    else:
        print(f'{ColorCoding.OKGREEN}Training Autoencoder{ColorCoding.ENDC}')
        autoencoder.fit(x=one_hot_encoded_reads, y=one_hot_encoded_reads,
                        epochs=10,
                        batch_size=batch_size,
                        shuffle=True,
                        verbose=config['verbose'])
        # callbacks=[autoencoder_checkpoint])

        print(f'{ColorCoding.OKGREEN}Saving Weights{ColorCoding.ENDC}')
        autoencoder.save_weights(config[config['data']]['weights_path'])

    print(f'{ColorCoding.OKGREEN}Build Clustering Model{ColorCoding.ENDC}')

    encoder = Model(inputs=model_input, outputs=encoder_output, name='encoder')
    encoder.compile(optimizer='adam', loss=MeanSquaredError())
    encoder.summary()
    # train k-means
    kmeans_rep = 1
    print(
        f'{ColorCoding.OKGREEN}Initialize and Running KMeans{ColorCoding.ENDC} (' + str(kmeans_rep) + ' times)')
    # TODO according to: 2.3 clustering layer
    old_mec_result = 0
    print('n_clusters:', n_clusters)
    # TODO check if load and save centroids

    # TODO uncomment -->
    # for i in range(kmeans_rep):
    #     print('kmeans_rep:', i)
    #     kmeans = KMeans(n_clusters=n_clusters, n_init=30, verbose=0)  # TODO change to verbose
    #     predicted_clusters = kmeans.fit_predict(encoder.predict(one_hot_encoded_reads))
    #     consensus_sequences = majority_voting.align_reads_per_cluster(one_hot_encoded_reads,
    #                                                                   predicted_clusters, n_clusters)
    #     new_mec_results = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)
    #     print("mec", new_mec_results)
    #     if i == 0 or old_mec_result > new_mec_results:
    #         print(old_mec_result)
    #         print(new_mec_results)
    #         centroids = kmeans.cluster_centers_
    #         print(centroids)
    #         old_mec_result = new_mec_results
    # TODO till here <--

    # take it all together and
    # print(autoencoder.encoder.input)
    # print(autoencoder.encoder.output)
    # print(autoencoder.decoder.input)
    # print(autoencoder.decoder.output)
    clustering_layer = ClusteringLayer(n_clusters=n_clusters, name='clustering')(encoder_output)
    # decoder = Model(inputs=autoencoder.decoder.input, outputs=autoencoder.decoder.output)
    # output_decoder = decoder(autoencoder.encoder.output)

    # Model autoencoder and clustering
    cae_model = Model(inputs=model_input,
                      outputs=[decoder_output, clustering_layer])
    print(cae_model.get_layer(name='clustering'))
    print(list(i.name for i in cae_model.layers))
    print(list(i.output for i in cae_model.layers))
    for i in cae_model.layers:
        if i.name == 'model':
            print('decoder')
            for j in i.layers:
                print(j.name, j.input, j.output)
        else:
            print(i.name, i.input, i.output)

    print(cae_model.output)
    lam = 0.1
    cae_model.compile(loss=[MeanSquaredError(), KLD],
                      loss_weights=[1 - lam, lam],
                      optimizer='adam')
    # print(centroids)
    # with open("centroids", "wb") as fp:
    #     pickle.dump(centroids, fp)
    with open("centroids", "rb") as fp:
        centroids = pickle.load(fp)
    print(centroids)
    cae_model.get_layer(name='clustering').set_weights([centroids])
    cae_model.build(input_shape=one_hot_encoded_reads.shape)
    cae_model.summary()

    mec_results = []

    old_mec_result = 0
    for epoch in range(20):  # TODO set to 2000
        print('epoch', 1)
        _, clustering_output = cae_model.predict(one_hot_encoded_reads, verbose=1)
        predicted_clusters = np.argmax(clustering_output, axis=1)
        print(clustering_output)
        print(predicted_clusters)

        print('number of clusters')
        print('0', np.count_nonzero(predicted_clusters == 0))
        print('1', np.count_nonzero(predicted_clusters == 1))
        print('2', np.count_nonzero(predicted_clusters == 2))
        print('3', np.count_nonzero(predicted_clusters == 3))
        print('4', np.count_nonzero(predicted_clusters == 4))

        print(predicted_clusters.shape)

        consensus_sequences = majority_voting.align_reads_per_cluster(one_hot_encoded_reads,
                                                                      clustering_output, n_clusters)
        new_mec_results = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)

        # build clustering output per read depending on hamming distance at non-zero places
        predicted_clusters_training = np.zeros((one_hot_encoded_reads.shape[0], n_clusters), dtype=int)
        print(predicted_clusters_training)
        for read_index in range(one_hot_encoded_reads.shape[0]):
            hd = []
            for consensus_sequence in consensus_sequences:
                hd.append(performance.hamming_distance(one_hot_encoded_reads[read_index], consensus_sequence))
            predicted_clusters_training[read_index][np.argmin(hd)] = 1
            predicted_clusters_training_tensor = tf.convert_to_tensor(predicted_clusters_training, dtype=tf.int8)
        print(predicted_clusters_training)
        print(predicted_clusters_training.shape)
        print('lets go')

        if epoch > 1 and old_mec_result == new_mec_results:
            break
        for i in range(number_of_batches):
            print('batch', i)
            first = i * batch_size
            last = (i + 1) * batch_size
            if last > one_hot_encoded_reads.shape[0]:
                last = one_hot_encoded_reads.shape[0] - 1
            cae_model.train_on_batch(x=one_hot_encoded_reads[first:last],
                                     y=[one_hot_encoded_reads[first:last],
                                        predicted_clusters_training_tensor[first:last]])
            # TODO how to run through batches

    # TODO train MODEL
    #  - have a look at train_on_batch
    #  - when update
    #  - maximise MEC
    # output_cluster = model.predict(x=tf.expand_dims(one_hot_encoded_reads[0], axis=0))
    # output_encoder = autoencoder.predict(x=tf.expand_dims(one_hot_encoded_reads[0], axis=0))
    # print("output_cluster", output_cluster)
    # print("evaluate")
    # autoencoder.evaluate(x=one_hot_encoded_reads, y=one_hot_encoded_reads, verbose=verbose)

    # if config['save']:
    #     if config['data'] == 'experimental':
    #         autoencoder.save('./results/models/454_weights')
    #     else:
    #         autoencoder.save(
    #             './results/models/created_weights_' + str(config['number_of_strains']) + '_' + str(
    #                 config['read_length']) + '_' + str(config['min_number_of_reads_per_strain']))

    print(one_hot_encoded_reads[0].shape)
    # print(tf.squeeze(output_encoder, [0]).shape)
    print(f'{ColorCoding.OKGREEN}FINITO{ColorCoding.ENDC}')
