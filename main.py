import pickle
import sys
from os.path import exists

import numpy as np
# import psutil
import tensorflow as tf
# from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
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

# tf.compat.v1.disable_eager_execution()
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

    # file = open('./data/per_gene/reference/snp_positions_p17.txt', 'r')
    # lines = file.readlines()
    # output = [str(int(line) - 1) for line in lines]
    # file2 = open('./data/per_gene/reference/snp_positions_p17.txt', 'w')
    # final = '\n'.join(output)
    # print(final)
    # file2.write(final)
    # exit(-1)
    print(tf.keras.backend.floatx())
    # TODO print config and info
    print('Python version: {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print('Tensorflow version: {}'.format(tf.__version__))
    print('Numpy version: {}'.format(np.version.version))
    print(f'Working with {ColorCoding.OKGREEN}{data}{ColorCoding.ENDC} reads')
    number_of_files = one_hot.encode_sam()

    n_clusters = config[data]['n_clusters']
    verbose = config['verbose']

    if data == 'created':
        with tf.device('/cpu:0'):
            one_hot_encoded_reads = load_tensor_file(
                config[data]['one_hot_path'] + '_' + str(config[data]['n_clusters']))
            shape = one_hot_encoded_reads.shape
    else:
        with tf.device('/cpu:0'):
            one_hot_encoded_reads = load_tensor_file(config[data]['one_hot_path'])
            shape = one_hot_encoded_reads.shape

    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_tensor_slices((one_hot_encoded_reads, one_hot_encoded_reads))
        dataset = dataset.shuffle(10).batch(config[data]['batch_size'])  # .prefetch(2)
        print(dataset)

    # create autoencoder
    print(f'{ColorCoding.OKGREEN}before{ColorCoding.ENDC}')

    pooling = ' pooling' if config[data]['pooling'] else 'out pooling';
    print(f'{ColorCoding.OKGREEN}Model build with{pooling}{ColorCoding.ENDC}')
    # with distribution_strategy.scope():
    print(shape[1:])
    if config[data]['pooling']:
        model_input, latent_space, decoder_output = get_autoencoder_key_points_with_pooling(shape[1:])
    else:
        model_input, latent_space, decoder_output = get_autoencoder_key_points(shape[1:])
    # model_input, latent_space, decoder_output = get_CAECseq(shape[1:])

    print(f'{ColorCoding.OKGREEN}building autoencoder{ColorCoding.ENDC}')

    autoencoder = Model(inputs=model_input, outputs=decoder_output, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss=MeanSquaredError())
    # autoencoder.optimizer.lr.assign(0.001)
    # TODO delete following two lines
    autoencoder.build(input_shape=shape)
    autoencoder.summary()

    encoder = Model(inputs=model_input, outputs=latent_space, name='encoder')
    encoder.compile(optimizer='adam', loss=MeanSquaredError())
    # encoder.optimizer.lr.assign(0.001)
    encoder.summary()

    print(f'{ColorCoding.OKGREEN}building CAE{ColorCoding.ENDC}')

    clustering_layer = ClusteringLayer(n_clusters=n_clusters, name='clustering')(latent_space)

    clustering_model = Model(inputs=model_input,
                             outputs=clustering_layer)
    encoder.compile(optimizer='adam', loss=KLD)
    # encoder.optimizer.lr.assign(0.001)

    lam = 0.1
    cae_model = Model(inputs=model_input,
                      outputs=[decoder_output, clustering_layer])
    cae_model.compile(loss=[MeanSquaredError(), KLD],
                      loss_weights=[1 - lam, lam],
                      optimizer='adam')
    # cae_model.optimizer.lr.assign(0.001)

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

        print(f'{ColorCoding.OKGREEN}Saving Weights{ColorCoding.ENDC}')
        autoencoder.save_weights(config[data]['weights_path'])

    print(f'{ColorCoding.OKGREEN}Build Clustering Model{ColorCoding.ENDC}')

    dataset = tf.data.Dataset.from_tensor_slices((one_hot_encoded_reads, one_hot_encoded_reads))
    dataset = dataset.batch(config[data]['batch_size'])
    prediction_for_kmeans_training = encoder.predict(dataset, verbose=verbose)

    kmeans_rep = 10
    best_mec_result = 0
    print('n_clusters:', n_clusters)
    print(f'{ColorCoding.OKGREEN}Initialize KMeans{ColorCoding.ENDC} for ' + str(n_clusters) + ' clusters (' + str(
        kmeans_rep) + ' times)')
    # TODO uncomment -->
    for i in range(kmeans_rep):
        print('kmeans_rep:', i)
        kmeans = KMeans(n_clusters=n_clusters, n_init=30, verbose=0)
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
    # _, _, cpr = performance.correct_phasing_rate(best_consensus_sequences, 'kmeans_init')
    # print('CRP after centroids:', cpr)

    # TODO delete incl. file
    with open(config[data]['centroids_name'], "wb") as fp:
        pickle.dump(centroids, fp)
    # with open(config[data]['centroids_name'], "rb") as fp:
    #     centroids = pickle.load(fp)

    cae_model.get_layer(name='clustering').set_weights([centroids])

    old_mec_result = 0
    for epoch in range(40):
        print('epoch', epoch)
        clustering_output = clustering_model.predict(x=dataset, verbose=config['verbose'])
        # clustering_output = clustering_model.predict(x=one_hot_encoded_reads, verbose=config['verbose'])
        # _, clustering_output = cae_model.predict(x=one_hot_encoded_reads, verbose=config['verbose'])
        # TODO delete incl. file
        with open("clustering_output", "wb") as fp:
            pickle.dump(clustering_output, fp)
        # with open("clustering_output", "rb") as fp:
        #     clustering_output = pickle.load(fp)
        predicted_clusters = np.argmax(clustering_output, axis=1)
        print('number of clusters')
        print('0', np.count_nonzero(predicted_clusters == 0))
        print('1', np.count_nonzero(predicted_clusters == 1))
        print('2', np.count_nonzero(predicted_clusters == 2))
        # print('3', np.count_nonzero(predicted_clusters == 3))
        # print('4', np.count_nonzero(predicted_clusters == 4))

        consensus_sequences = majority_voting.align_reads_per_cluster(one_hot_encoded_reads,
                                                                      predicted_clusters, n_clusters)
        # decoded_sequences = one_hot.decode(consensus_sequences)

        new_mec_result = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)
        print('new mec result', new_mec_result)

        # build clustering output per read depending on hamming distance at non-zero places
        predicted_clusters = majority_voting.assign_reads_to_best_fitting_consensus_sequence(n_clusters,
                                                                                             consensus_sequences,
                                                                                             one_hot_encoded_reads)
        predicted_clusters_tensor = tf.convert_to_tensor(predicted_clusters, dtype=tf.int8)

        if epoch > 1 and old_mec_result == new_mec_result:
            break

        old_mec_result = new_mec_result

        dataset = tf.data.Dataset.from_tensor_slices(
            (one_hot_encoded_reads, (one_hot_encoded_reads, predicted_clusters_tensor)))

        dataset = dataset.batch(config[data]['batch_size'])
        cae_model.fit(dataset, epochs=1, verbose=config['verbose'])

    print('Final MEC after training:', new_mec_result)
    _, min_indexes, cpr = performance.correct_phasing_rate(consensus_sequences, 'after_training')
    print('CRP after training:', cpr)
    print('min_indexes after training:', min_indexes)
    _, min_indexes, cpr = performance.correct_phasing_rate(consensus_sequences, 'reverse_after_training', reverse=True)
    print('CRP after training (reversed):', cpr)
    print('min_indexes after training (reversed):', min_indexes)

    # add deletions
    # consensus_sequences = consensus_sequences = majority_voting.align_reads_per_cluster(one_hot_encoded_reads,
    #                                                                                     predicted_clusters,
    #                                                                                     n_clusters, last=True)
    #
    # _, min_indexes, cpr = performance.correct_phasing_rate(consensus_sequences, 'deletion_added')
    # print('CRP after last:', cpr)
    # print('min_indexes after last:', min_indexes)

    # _, min_indexes, cpr = performance.correct_phasing_rate(consensus_sequences, 'reverse2_after_training', reverse=True)
    # print('CRP after training (reversed 2):', cpr)
    # print('min_indexes after training (reversed 2):', min_indexes)
    # # correction -> converge
    # old_mec_result = 0
    # new_mec_result = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)
    # while new_mec_result != old_mec_result:
    #     index = []
    #     for i in range(one_hot_encoded_reads.shape[0]):
    #         dis = np.zeros((consensus_sequences.shape[0]))
    #         for j in range(consensus_sequences.shape[0]):
    #             dis[j] = performance.hamming_distance(one_hot_encoded_reads[i, :], consensus_sequences[j, :])
    #         index.append(np.argmin(dis))
    #
    #     new_consensus_sequences = np.zeros(consensus_sequences.shape)
    #     for i in range(consensus_sequences.shape[0]):
    #         consensus_sequences[i, :] = np.argmax(ACGT_count(SNVmatrix[np.array(index) == i, :]), axis=1) + 1
    #     consensus_sequences = new_consensus_sequences.copy()
    #     old_mec_result = new_mec_result
    #     new_mec_result = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)
    # old_mec_result = 0
    # while old_mec_result != new_mec_result:
    #     old_mec_result = new_mec_result
    #     predicted_clusters = majority_voting.assign_reads_to_best_fitting_consensus_sequence(n_clusters,
    #                                                                                          consensus_sequences,
    #                                                                                          one_hot_encoded_reads)
    #

    old_mec_result = 0
    new_mec_result = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)

    count = 0
    while new_mec_result != old_mec_result:
        predicted_clusters = majority_voting.assign_reads_to_best_fitting_consensus_sequence(n_clusters,
                                                                                             consensus_sequences,
                                                                                             one_hot_encoded_reads)
        predicted_clusters_array = np.argmax(predicted_clusters, axis=1)
        new_consensus_sequences = []
        for i in range(n_clusters):
            reads_of_cluster_n = one_hot_encoded_reads[np.where(predicted_clusters_array == i, )[0]]
            if len(reads_of_cluster_n) != 0:
                clustered_reads_sum = reads_of_cluster_n.sum(axis=0)
                consensus_sequence = np.zeros(consensus_sequences[0].shape)
                uncovered_positions = np.where(np.sum(clustered_reads_sum, axis=1) == 0)[0]
                for j in range(clustered_reads_sum.shape[0]):
                    print(clustered_reads_sum[j].sum(axis=0))
                    threshold = n_clusters * reads_of_cluster_n.shape[0] / 10
                    print('reads_of_cluster_n.shape[0]', reads_of_cluster_n.shape[0])  # ~ 15
                    # if clustered_reads_sum[j].sum(axis=0) != 0:
                    if clustered_reads_sum[j].sum(axis=0) > reads_of_cluster_n.shape[1] / 2:
                        np.argmax(clustered_reads_sum[j])
                        consensus_sequence[j, np.argmax(clustered_reads_sum[j])] = 1
                new_consensus_sequences.append(consensus_sequence)

        consensus_sequences = new_consensus_sequences.copy()
        old_mec_result = new_mec_result
        new_mec_result = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)
        count += 1
        #
        #         if clustered_reads_sum[j].sum(axis=0) != 0:
        #             # if clustered_reads_sum[i].sum(axis=0) > clustered_reads.shape[1] / 16:
        #             if len(np.where(clustered_reads_sum[j][:] == np.max(clustered_reads_sum[j][:]))[0]) == 1:
        # for i in range(consensus_sequences.shape[0]):
        #     reads_single = one_hot_encoded_reads[np.array(index) == i, :]
        #     single_sta = np.zeros((n_SNV, 4))
        #     if len(reads_single) != 0:
        #         single_sta = ACGT_count(reads_single)
        #     new_haplo[i, :] = np.argmax(single_sta, axis=1) + 1
        #     uncov_pos = np.where(np.sum(single_sta, axis=1) == 0)[0]
        #     if len(uncov_pos) != 0:
        #         new_haplo[i, uncov_pos] = 0
    print(new_mec_result)

    # consensus_sequences = []
    # # TODO FIX MEEEEEEEEE
    # for i in range(n_clusters):
    #     reads_of_cluster_n = one_hot_encoded_reads[np.where(predicted_clusters == i)[0]]
    #     new_consensus_sequence = np.zeros((reads_of_cluster_n.shape[1], 4, 1), dtype=int)
    #
    #     clustered_reads_sum = reads_of_cluster_n.sum(axis=0)
    #
    #     for j in range(clustered_reads_sum.shape[0]):
    #         if clustered_reads_sum[j].sum(axis=0) != 0:
    #             #  print(clustered_reads_sum[j])
    #             #  print(np.argmax(clustered_reads_sum[j]))
    #             # if clustered_reads_sum[i].sum(axis=0) > clustered_reads.shape[1] / 16:
    #             new_consensus_sequence[j, np.argmax(clustered_reads_sum[j])] = 1
    #     # consensus_sequences = majority_voting.align_reads_per_cluster(one_hot_encoded_reads,
    #     #                                                               predicted_clusters, n_clusters)
    #
    #     consensus_sequences.append(new_consensus_sequence)
    # print(one_hot.decode(consensus_sequences, ''))
    # new_mec_result = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)
    # print(new_mec_result)
    min_sum, min_indexes, cpr = performance.correct_phasing_rate(consensus_sequences, 'correction')

    print('final MEC:', new_mec_result)
    print('final CPR:', cpr)
    print('final min_indexes:', min_indexes)

    min_sum, min_indexes, cpr = performance.correct_phasing_rate(consensus_sequences, 'correction_reversed',
                                                                 reverse=True)

    print('final MEC (reversed):', new_mec_result)
    print('final CPR (reversed):', cpr)
    print('final min_indexes (reversed):', min_indexes)
    with open("CPR", "wb") as fp:
        pickle.dump([min_sum, min_indexes, cpr], fp)

    print(f'{ColorCoding.OKGREEN}FINITO{ColorCoding.ENDC}')
