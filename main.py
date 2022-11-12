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
from create_data import create_data
from helpers.IO import load_tensor_file
from helpers.colors_coding import ColorCoding
from helpers.config import get_config
from models.autoencoder import get_autoencoder_key_points_with_pooling, get_autoencoder_key_points
from models.layers.cluster import ClusteringLayer
from prepare_data import prepare_data

config = get_config()
data = config['data']
clusters = [3, 4, 5, 6, 7, 8]

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if __name__ == '__main__':

    print('Python version: {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print('Tensorflow version: {}'.format(tf.__version__))
    print('Numpy version: {}'.format(np.version.version))
    print(f'Working with {ColorCoding.OKGREEN}{data}{ColorCoding.ENDC} reads')
    if data == 'experimental':
        name = config[data]['name']
        print(f'Looking at gene {ColorCoding.OKGREEN}{name}{ColorCoding.ENDC}')
    repetitions = 1
    if data == 'created':
        NOR = []
        repetitions = 10
        MEC = []
        CPR = []
        CPR_reversed = []
        zero_counts = []
        similarity_scores = []
    for repetition in range(repetitions):

        if data == 'created':
            create_data()
            prepare_data()

        one_hot.encode_sam()

        n_clusters = config[data]['n_clusters']
        verbose = config['verbose']
        path = ''
        if data == 'created':
            with tf.device('/cpu:0'):
                path += '_' + str(config[data]['n_clusters']) + '_' + str(
                    config[data]['coverage']) + '_' + str(config[data]['read_length']) + '_' + str(
                    config[data]['sequencing_error'])
                one_hot_encoded_reads = load_tensor_file(
                    config[data]['one_hot_path'] + path)
                shape = one_hot_encoded_reads.shape
        else:
            with tf.device('/cpu:0'):
                one_hot_encoded_reads = load_tensor_file(config[data]['one_hot_path'])
                shape = one_hot_encoded_reads.shape

        with tf.device('/cpu:0'):
            dataset = tf.data.Dataset.from_tensor_slices((one_hot_encoded_reads, one_hot_encoded_reads))
            dataset = dataset.shuffle(10).batch(config[data]['batch_size'])  # .prefetch(2)

        # create autoencoder

        pooling = ' pooling' if config['pooling'] else 'out pooling';
        print(f'{ColorCoding.OKGREEN}Model build with{pooling}{ColorCoding.ENDC}')
        # with distribution_strategy.scope():
        print(shape)
        if config['pooling']:
            model_input, latent_space, decoder_output = get_autoencoder_key_points_with_pooling(shape[1:])
        else:
            model_input, latent_space, decoder_output = get_autoencoder_key_points(shape[1:])
        # model_input, latent_space, decoder_output = get_CAECseq(shape[1:])

        print(f'{ColorCoding.OKGREEN}building autoencoder{ColorCoding.ENDC}')

        autoencoder = Model(inputs=model_input, outputs=decoder_output, name='autoencoder')
        autoencoder.compile(optimizer='adam', loss=MeanSquaredError())
        # autoencoder.optimizer.lr.assign(0.001)
        autoencoder.build(input_shape=shape)
        # autoencoder.summary()

        encoder = Model(inputs=model_input, outputs=latent_space, name='encoder')
        encoder.compile(optimizer='adam', loss=MeanSquaredError())
        # encoder.optimizer.lr.assign(0.001)
        # encoder.summary()

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

        # train autoencoder
        if config['load'] and not data == 'created' and exists(config[data]['weights_path'] + path + '.index'):
            print(f'{ColorCoding.OKGREEN}Loading weights{ColorCoding.ENDC}')
            autoencoder.load_weights(config[data]['weights_path'] + path)

        else:
            print(f'{ColorCoding.OKGREEN}Training Autoencoder{ColorCoding.ENDC}')
            autoencoder.fit(x=dataset,
                            epochs=100,
                            verbose=config['verbose'])

            print(f'{ColorCoding.OKGREEN}Saving Weights{ColorCoding.ENDC}')
            autoencoder.save_weights(config[data]['weights_path'] + path)

        print(f'{ColorCoding.OKGREEN}Build Clustering Model{ColorCoding.ENDC}')

        dataset = tf.data.Dataset.from_tensor_slices((one_hot_encoded_reads, one_hot_encoded_reads))
        dataset = dataset.batch(config[data]['batch_size'])
        prediction_for_kmeans_training = encoder.predict(dataset, verbose=verbose)
        repetitions = 1

        autoencoder.load_weights(config[data]['weights_path'] + path)
        kmeans_rep = 10
        best_mec_result = 0
        print('n_clusters:', n_clusters)
        print(f'{ColorCoding.OKGREEN}Initialize KMeans{ColorCoding.ENDC} for ' + str(n_clusters) + ' clusters (' + str(
            kmeans_rep) + ' times)')
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

        cae_model.get_layer(name='clustering').set_weights([centroids])

        old_mec_result = 0
        for epoch in range(40):
            print('epoch', epoch)
            clustering_output = clustering_model.predict(x=dataset, verbose=verbose)

            predicted_clusters = np.argmax(clustering_output, axis=1)

            consensus_sequences = majority_voting.align_reads_per_cluster(one_hot_encoded_reads,
                                                                          predicted_clusters, n_clusters)

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
        _, min_indexes, cpr_reversed = performance.correct_phasing_rate(consensus_sequences, 'reverse_after_training',
                                                                        reverse=True)
        print('CRP after training (reversed):', cpr_reversed)
        print('min_indexes after training (reversed):', min_indexes)

        if data == 'created':
            # check similarity:
            zero_counter = 0
            hd = 0
            first_sequence = consensus_sequences[0]
            for consensus_sequence in consensus_sequences:
                # print(consensus_sequence.shape[0])
                zero_counter += consensus_sequence.shape[0] - np.sum(np.sum(consensus_sequence, axis=1))
                hd += performance.hamming_distance(first_sequence, consensus_sequence)
                # print(hd)
            similarity_score = 100 * (consensus_sequences[0].shape[0] * len(consensus_sequences) - hd) / (
                    consensus_sequences[0].shape[0] * len(consensus_sequences))
            NOR.append(one_hot_encoded_reads.shape[0])
            if similarity_score > 95:
                print('similarity score to high -> results ignored')
                continue
            MEC.append(new_mec_result)
            CPR.append(cpr)
            CPR_reversed.append(cpr_reversed)
            zero_counts.append(100 * zero_counter / (consensus_sequences[0].shape[0] * len(consensus_sequences)))
            similarity_scores.append(100 * (consensus_sequences[0].shape[0] * len(consensus_sequences) - hd) / (
                    consensus_sequences[0].shape[0] * len(consensus_sequences)))
            continue

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
                        threshold = reads_of_cluster_n.shape[1] / 2
                        if data == 'created':
                            threshold = config[data]['threshold']
                        if clustered_reads_sum[j].sum(axis=0) > threshold:
                            np.argmax(clustered_reads_sum[j])
                            consensus_sequence[j, np.argmax(clustered_reads_sum[j])] = 1
                    new_consensus_sequences.append(consensus_sequence)

            consensus_sequences = new_consensus_sequences.copy()
            old_mec_result = new_mec_result
            new_mec_result = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)
            count += 1

        min_sum, min_indexes, cpr = performance.correct_phasing_rate(consensus_sequences, 'correction')

        print('final MEC:', new_mec_result)
        print('final CPR:', np.round(100 * cpr, 1))
        print('final min_indexes:', min_indexes)

        min_sum, min_indexes, cpr = performance.correct_phasing_rate(consensus_sequences, 'correction_reversed',
                                                                     reverse=True)

        print('final MEC (reversed):', new_mec_result)
        print('final CPR (reversed):', np.round(100 * cpr, 1))
        print('final min_indexes (reversed):', min_indexes)

        if data == 'experimental':
            name_of_reference_strains = ['HXB2', '89.6', 'JRCSF', 'NL43', 'YU2']
            print('Number of reads per cluster')
            for i in range(5):
                number_of_reads_per_cluster = np.count_nonzero(predicted_clusters_array == min_indexes[i])
                print(name_of_reference_strains[i], number_of_reads_per_cluster,
                      str(np.round((100 * number_of_reads_per_cluster / one_hot_encoded_reads.shape[0]), 1)) + '%')
                print('---------------------------------------------------------------------------------------')
    if data == 'created':
        print('Number Of Reads:', np.round(sum(NOR) / len(NOR), 1))

        print('MEC:', MEC)
        print('CPR:', CPR)
        print('CPR_r:', CPR_reversed)
        print('zero:', zero_counts)
        print('sim:', similarity_scores)
        if len(MEC) == 0:
            print('similarirty score to high for all')
        else:
            best_mec_index = np.argmin(np.array(MEC))
            print('Results with average MEC score (average/best)')
            print('Number Of Reads:', np.round(sum(NOR) / len(NOR), 1))
            print('MEC:', str(sum(MEC) / len(MEC)) + '/' + str(MEC[best_mec_index]))
            print('CPR:', np.round(100 * sum(CPR) / len(CPR), 1))
            print('CPR_r:', str(np.round(100 * sum(CPR_reversed) / len(CPR_reversed), 1)) + '/' + str(
                np.round(100 * CPR_reversed[best_mec_index], 1)))
            print('average_sim:', np.round(sum(similarity_scores) / len(similarity_scores), 1))
            print('sim:', similarity_scores)
        print('-----------------------------------------')
    print(f'{ColorCoding.OKGREEN}FINITO{ColorCoding.ENDC}')
