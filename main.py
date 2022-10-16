import pickle
import sys
from os.path import exists

import numpy as np
# import psutil
import tensorflow as tf
from sklearn.cluster import KMeans
# from sklearn.cluster import KMeans
from tensorflow.python.keras import Model
from tensorflow.python.keras.losses import MeanSquaredError, KLD

import majority_voting
import one_hot
import performance
from helpers.IO import load_tensor_file
from helpers.colors_coding import ColorCoding
from helpers.config import get_config
from models.autoencoder import get_autoencoder_key_points_with_pooling, get_autoencoder_key_points
from models.layers.cluster import ClusteringLayer

# in bytes
config = get_config()
data = config['data']

# print(tf.keras.backend.floatx())
# keras_backend.set_floatx('float16')
# print("Num GPUs Available: ", tf.config.list_physical_devices('GPU'))
# physical_devices = tf.config.list_physical_devices('GPU')
#
# tf.config.set_visible_devices(physical_devices[1], 'GPU')
# visible_devices = tf.config.get_visible_devices()
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# for device in visible_devices:
#     assert device.device_type != 'GPU'
# distribution_strategy = tf.distribute.MirroredStrategy()

if __name__ == '__main__':
    print(tf.keras.backend.floatx())
    # TODO print config and info
    print('Python version: {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print('Tensorflow version: {}'.format(tf.__version__))
    print('Numpy version: {}'.format(np.version.version))
    print(f'Working with {ColorCoding.OKGREEN}{data}{ColorCoding.ENDC} reads')
    number_of_files = one_hot.encode_sam()

    n_clusters = config['n_clusters']
    verbose = config['verbose']

    if data == 'illumina':
        one_hot_encoded_reads = load_tensor_file(config[data]['one_hot_path'] + '_0')
        index = 1
        while exists(config[data]['one_hot_path'] + '_' + str(index) + '.npy'):
            next_one_hot_encoded_reads = load_tensor_file(config[data]['one_hot_path'] + '_' + str(index))
            index += 1
            print(exists(config[data]['one_hot_path'] + '_' + str(index) + '.npy'))
            one_hot_encoded_reads = tf.concat([one_hot_encoded_reads, next_one_hot_encoded_reads], axis=0)
    else:
        # print('not loading data')
        with tf.device('/cpu:0'):
            one_hot_encoded_reads = load_tensor_file(config[data]['one_hot_path'])
            shape = one_hot_encoded_reads.shape

    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices((one_hot_encoded_reads, one_hot_encoded_reads))
        dataset = dataset.shuffle(10).batch(config[data]['batch_size'])  # .prefetch(2)
        print(dataset)

    # create autoencoder
    print(f'{ColorCoding.OKGREEN}before{ColorCoding.ENDC}')

    # model_input, encoder_output, decoder_output = get_autoencoder_key_points(one_hot_encoded_reads.shape[1:])

    # model_input, encoder_output, decoder_output = get_autoencoder_key_points_with_pooling(
    #     (9850, 4, 1))
    # model_input, encoder_output, decoder_output = get_autoencoder_key_points((9850, 4, 1))
    pooling = ' pooling' if config[data]['pooling'] else 'out pooling';
    print(f'{ColorCoding.OKGREEN}Model build with{pooling}{ColorCoding.ENDC}')
    # with distribution_strategy.scope():
    if config[data]['pooling']:
        model_input, encoder_output, decoder_output = get_autoencoder_key_points_with_pooling(
            shape[1:])
    else:
        model_input, encoder_output, decoder_output = get_autoencoder_key_points(
            shape[1:])
    print(f'{ColorCoding.OKGREEN}building autoencoder{ColorCoding.ENDC}')

    autoencoder = Model(inputs=model_input, outputs=decoder_output, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss=MeanSquaredError())
    autoencoder.optimizer.lr.assign(0.001)
    # TODO delete following two lines
    autoencoder.build(input_shape=shape)
    autoencoder.summary()

    encoder = Model(inputs=model_input, outputs=encoder_output, name='encoder')
    encoder.compile(optimizer='adam', loss=MeanSquaredError())
    encoder.optimizer.lr.assign(0.001)
    encoder.summary()

    print(f'{ColorCoding.OKGREEN}building CAE{ColorCoding.ENDC}')

    clustering_layer = ClusteringLayer(n_clusters=n_clusters, name='clustering')(encoder_output)

    clustering_model = Model(inputs=model_input,
                             outputs=clustering_layer)
    encoder.compile(optimizer='adam', loss=KLD)
    encoder.optimizer.lr.assign(0.001)

    lam = 0.1
    cae_model = Model(inputs=model_input,
                      outputs=[decoder_output, clustering_layer])
    cae_model.compile(loss=[MeanSquaredError(), KLD],
                      loss_weights=[1 - lam, lam],
                      optimizer='adam')
    cae_model.optimizer.lr.assign(0.001)

    print('reads tensor shape:', shape)
    print('number of reads:', shape[0])
    print('batch_size:', config[data]['batch_size'])

    # train autoencoder
    # if False:
    if config['load'] and exists(config[data]['weights_path'] + '.index'):
        print(f'{ColorCoding.OKGREEN}Loading weights{ColorCoding.ENDC}')
        autoencoder.load_weights(config[data]['weights_path'])

    else:
        print(f'{ColorCoding.OKGREEN}Training Autoencoder{ColorCoding.ENDC}')
        autoencoder.fit(x=dataset,
                        epochs=100,
                        verbose=config['verbose'])

        # autoencoder.fit(x=one_hot_encoded_reads, y=one_hot_encoded_reads,
        #                 epochs=100,
        #                 batch_size=batch_size,
        #                 shuffle=True,
        #                 verbose=config['verbose'])

        print(f'{ColorCoding.OKGREEN}Saving Weights{ColorCoding.ENDC}')
        autoencoder.save_weights(config[data]['weights_path'])

    print(f'{ColorCoding.OKGREEN}Build Clustering Model{ColorCoding.ENDC}')

    # train k-means

    # NEW outside not inside loop
    # todo uncomment
    prediction_for_kmeans_training = encoder.predict(dataset, verbose=1)  # TODO change to verbose
    # max_clusters = config['max_clusters']
    # for n_clusters in range(1, max_clusters):
    n_clusters = config['n_clusters']
    # one_hot_encoded_reads = load_tensor_file(config[data]['one_hot_path'])

    kmeans_rep = 10  # TODO set to 10
    best_mec_result = 0
    print('n_clusters:', n_clusters)
    print(f'{ColorCoding.OKGREEN}Initialize KMeans{ColorCoding.ENDC} for ' + str(n_clusters) + ' clusters (' + str(
        kmeans_rep) + ' times)')
    # TODO uncomment -->
    for i in range(kmeans_rep):
        print('kmeans_rep:', i)
        kmeans = KMeans(n_clusters=n_clusters, n_init=30, verbose=0)  # TODO change to verbose
        predicted_clusters = kmeans.fit_predict(prediction_for_kmeans_training)
        consensus_sequences = majority_voting.align_reads_per_cluster(one_hot_encoded_reads,
                                                                      predicted_clusters, n_clusters)
        new_mec_result = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)
        print("mec", new_mec_result)
        if i == 0 or best_mec_result > new_mec_result:
            centroids = kmeans.cluster_centers_
            best_mec_result = new_mec_result
            best_consensus_sequences = consensus_sequences
    # TODO till here <--
    # print('Best MEC after cluster initialisation:', best_mec_result)
    # _, _, cpr = performance.correct_phasing_rate(best_consensus_sequences)
    # print('CRP after centroids:', cpr)

    # TODO delete incl. file
    with open(config[data]['centroids_name'], "wb") as fp:
        pickle.dump(centroids, fp)
    # with open(config[data]['centroids_name'], "rb") as fp:
    #     centroids = pickle.load(fp)

    cae_model.get_layer(name='clustering').set_weights([centroids])
    # cae_model.build(input_shape=one_hot_encoded_reads.shape)
    # cae_model.summary()

    mec_results = []

    old_mec_result = 0
    for epoch in range(1):  # TODO set to 2000
        print('epoch', epoch)
        # TODO try if possible
        clustering_output = clustering_model.predict(dataset, verbose=config['verbose'])
        # _, clustering_output = cae_model.predict(dataset, verbose=config['verbose'])
        # TODO delete incl. file
        with open("clustering_output", "wb") as fp:
            pickle.dump(clustering_output, fp)
        # with open("clustering_output", "rb") as fp:
        #     clustering_output = pickle.load(fp)
        predicted_clusters = np.argmax(clustering_output, axis=1)
        # print('number of clusters')
        # print('0', np.count_nonzero(predicted_clusters == 0))
        # print('1', np.count_nonzero(predicted_clusters == 1))
        # print('2', np.count_nonzero(predicted_clusters == 2))
        # print('3', np.count_nonzero(predicted_clusters == 3))
        # print('4', np.count_nonzero(predicted_clusters == 4))

        consensus_sequences = majority_voting.align_reads_per_cluster(one_hot_encoded_reads,
                                                                      predicted_clusters, n_clusters)
        decoded_sequences = one_hot.decode(consensus_sequences)

        new_mec_result = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)
        print('new mec result', new_mec_result)

        # build clustering output per read depending on hamming distance at non-zero places
        predicted_clusters = majority_voting.assign_reads_to_best_fitting_consensus_sequence(shape[0],
                                                                                             n_clusters,
                                                                                             consensus_sequences,
                                                                                             one_hot_encoded_reads)
        predicted_clusters_tensor = tf.convert_to_tensor(predicted_clusters, dtype=tf.int8)

        if epoch > 1 and old_mec_result == new_mec_result:
            break

        old_mec_result = new_mec_result

        dataset = tf.data.Dataset.from_tensor_slices(
            (one_hot_encoded_reads, (one_hot_encoded_reads, predicted_clusters_tensor)))

        dataset = dataset.shuffle(10).batch(config[data]['batch_size'])
        cae_model.fit(dataset, epochs=1, verbose=config['verbose'])

    # TODO what is happening here?
    print('Final MEC after training:', best_mec_result)
    _, _, cpr = performance.correct_phasing_rate(consensus_sequences)
    print('CRP after training:', cpr)

    # correction -> converge
    old_mec_result = 0
    new_mec_result = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)

    while old_mec_result != new_mec_result:
        old_mec_result = new_mec_result
        predicted_clusters = majority_voting.assign_reads_to_best_fitting_consensus_sequence(shape[0], n_clusters,
                                                                                             consensus_sequences,
                                                                                             one_hot_encoded_reads)

        consensus_sequences = majority_voting.align_reads_per_cluster(one_hot_encoded_reads,
                                                                      predicted_clusters, n_clusters)
        new_mec_result = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)

    min_sum, min_indexes, cpr = performance.correct_phasing_rate(consensus_sequences)
    print('final MEC:', new_mec_result)
    print('final CPR:', cpr)
    with open("CPR", "wb") as fp:
        pickle.dump([min_sum, min_indexes, cpr], fp)
    # print(tf.squeeze(output_encoder, [0]).shape)
    print(f'{ColorCoding.OKGREEN}FINITO{ColorCoding.ENDC}')
