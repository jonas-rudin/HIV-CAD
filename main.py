import pickle
import sys
from os.path import exists

import numpy as np
import tensorflow as tf
# from sklearn.cluster import KMeans
from tensorflow.python.keras import Model
from tensorflow.python.keras.losses import MeanSquaredError, KLD

import majority_voting
import one_hot
import performance
from helpers.IO import load_tensor_file
from helpers.colors_coding import ColorCoding
from helpers.config import get_config
from models.autoencoder import get_autoencoder_key_points
from models.layers.cluster import ClusteringLayer

config = get_config()
data = config['data']
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if __name__ == '__main__':
    # TODO print config and info
    print('Python version: {}.{}.{}'.format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print('Tensorflow version: {}'.format(tf.__version__))
    print(f'Working with {ColorCoding.OKGREEN}{data}{ColorCoding.ENDC} reads')
    number_of_files = one_hot.encode_sam()
    # print(
    #     f'Working with {ColorCoding.OKBLUE}created{ColorCoding.ENDC} reads')
    # one_hot_encoded_reads, reads = create_data.create_reads()
    # number_of_files = 1

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
            print(one_hot_encoded_reads.shape)
    else:
        one_hot_encoded_reads = load_tensor_file(config[data]['one_hot_path'])
    # TODO uncomment
    # number_of_batches = 200
    number_of_batches = 200  # created
    batch_size = int(np.ceil(one_hot_encoded_reads.shape[0] / number_of_batches))

    # create autoencoder
    model_input, encoder_output, decoder_output = get_autoencoder_key_points(one_hot_encoded_reads.shape[1:])
    autoencoder = Model(inputs=model_input, outputs=decoder_output, name='autoencoder')
    # autoencoder = Autoencoder(one_hot_encoded_reads.shape[1:])
    autoencoder.compile(optimizer='adam', loss=MeanSquaredError())
    autoencoder.build(input_shape=one_hot_encoded_reads.shape)

    autoencoder.summary()
    batch_size = batch_size * 4

    print('reads tensor shape:', one_hot_encoded_reads.shape)
    print('number of reads:', one_hot_encoded_reads.shape[0])
    print('batch_size:', batch_size)

    # train autoencoder
    # if False:
    #     exit(-1)
    if config['load'] and exists(config[data]['weights_path'] + '.index'):
        print(f'{ColorCoding.OKGREEN}Loading weights{ColorCoding.ENDC}')
        autoencoder.load_weights(config[data]['weights_path'])

    else:
        print(f'{ColorCoding.OKGREEN}Training Autoencoder{ColorCoding.ENDC}')
        autoencoder.fit(x=one_hot_encoded_reads, y=one_hot_encoded_reads,
                        epochs=100,
                        batch_size=batch_size * 4,
                        shuffle=True,
                        verbose=config['verbose'])

        print(f'{ColorCoding.OKGREEN}Saving Weights{ColorCoding.ENDC}')
        autoencoder.save_weights(config[data]['weights_path'])

    print(f'{ColorCoding.OKGREEN}Build Clustering Model{ColorCoding.ENDC}')

    encoder = Model(inputs=model_input, outputs=encoder_output, name='encoder')
    encoder.compile(optimizer='adam', loss=MeanSquaredError())
    encoder.summary()
    # train k-means

    # NEW outside not inside loop
    # todo uncomment
    # prediction_for_kmeans_training = encoder.predict(one_hot_encoded_reads, verbose=1)  # TODO change to verbose
    # max_clusters = config['max_clusters']
    # for n_clusters in range(1, max_clusters):
    for n_clusters in [config['n_clusters']]:
        kmeans_rep = 10  # TODO set to 10
        best_mec_result = 0
        print('n_clusters:', n_clusters)
        print(f'{ColorCoding.OKGREEN}Initialize KMeans{ColorCoding.ENDC} for ' + str(n_clusters) + ' clusters (' + str(
            kmeans_rep) + ' times)')
        # TODO uncomment -->
        # for i in range(kmeans_rep):
        #     print('kmeans_rep:', i)
        #     kmeans = KMeans(n_clusters=n_clusters, n_init=30, verbose=0)  # TODO change to verbose
        #     predicted_clusters = kmeans.fit_predict(prediction_for_kmeans_training)
        #     consensus_sequences = majority_voting.align_reads_per_cluster(one_hot_encoded_reads,
        #                                                                   predicted_clusters, n_clusters)
        #     new_mec_results = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)
        #     print("mec", new_mec_results)
        #     if i == 0 or best_mec_result > new_mec_results:
        #         centroids = kmeans.cluster_centers_
        #         best_mec_result = new_mec_results
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
        # print model maybe in different
        # print(cae_model.get_layer(name='clustering'))
        # print(list(i.name for i in cae_model.layers))
        # for i in cae_model.layers:
        #     if i.name == 'model':
        #         print('decoder')
        #         for j in i.layers:
        #             print(j.name, j.input, j.output)
        #     else:
        #         print(i.name, i.input, i.output)
        #
        # print(cae_model.output)
        lam = 0.1
        cae_model.compile(loss=[MeanSquaredError(), KLD],
                          loss_weights=[1 - lam, lam],
                          optimizer='adam')

        # TODO delete incl. file
        # with open("centroids", "wb") as fp:
        #     pickle.dump(centroids, fp)
        with open("centroids", "rb") as fp:
            centroids = pickle.load(fp)

        cae_model.get_layer(name='clustering').set_weights([centroids])
        cae_model.build(input_shape=one_hot_encoded_reads.shape)
        cae_model.summary()

        mec_results = []

        old_mec_result = 0
        for epoch in range(20):  # TODO set to 2000
            print('epoch', epoch)
            _, clustering_output = cae_model.predict(one_hot_encoded_reads, verbose=1)
            # # TODO delete incl. file
            # with open("clustering_output", "wb") as fp:
            #     pickle.dump(clustering_output, fp)
            # with open("clustering_output", "rb") as fp:
            #     clustering_output = pickle.load(fp)
            predicted_clusters = np.argmax(clustering_output, axis=1)

            print('number of clusters')
            print('0', np.count_nonzero(predicted_clusters == 0))
            print('1', np.count_nonzero(predicted_clusters == 1))
            print('2', np.count_nonzero(predicted_clusters == 2))
            print('3', np.count_nonzero(predicted_clusters == 3))
            print('4', np.count_nonzero(predicted_clusters == 4))

            consensus_sequences = majority_voting.align_reads_per_cluster(one_hot_encoded_reads,
                                                                          predicted_clusters, n_clusters)
            decoded_sequences = one_hot.decode(consensus_sequences)
            for decoded_sequence in decoded_sequences:
                print(decoded_sequence)
            new_mec_results = performance.minimum_error_correction(one_hot_encoded_reads, consensus_sequences)
            print(new_mec_results)
            # build clustering output per read depending on hamming distance at non-zero places
            predicted_clusters_training = np.zeros((one_hot_encoded_reads.shape[0], n_clusters), dtype=int)
            for read_index in range(one_hot_encoded_reads.shape[0]):
                hd = []
                for consensus_sequence in consensus_sequences:
                    hd.append(performance.hamming_distance(one_hot_encoded_reads[read_index], consensus_sequence))
                predicted_clusters_training[read_index][np.argmin(hd)] = 1
                predicted_clusters_training_tensor = tf.convert_to_tensor(predicted_clusters_training, dtype=tf.int8)

            if epoch > 1 and old_mec_result == new_mec_results:
                break
            old_mec_result = new_mec_results
            # for i in range(number_of_batches):
            # # print('batch', i)
            # first = i * batch_size
            # last = (i + 1) * batch_size
            # if last > one_hot_encoded_reads.shape[0]:
            #     last = one_hot_encoded_reads.shape[0] - 1
            # cae_model.train_on_batch(x=one_hot_encoded_reads[first:last],
            #                          y=[one_hot_encoded_reads[first:last],
            #                             predicted_clusters_training_tensor[first:last]])
            # or (what is the difference?)
            cae_model.fit(x=one_hot_encoded_reads,
                          y=[one_hot_encoded_reads,
                             predicted_clusters_training_tensor],
                          epochs=1,
                          batch_size=batch_size,
                          verbose=1)

        # TODO what is happening here?
        # correction
        # pre_mec = 0
        # mec = MEC(SNVmatrix, haplotypes)
        # count = 0
        # while mec != pre_mec:
        #     index = []
        #     for i in range(SNVmatrix.shape[0]):
        #         dis = np.zeros((haplotypes.shape[0]))
        #         for j in range(haplotypes.shape[0]):
        #             dis[j] = hamming_distance(SNVmatrix[i, :], haplotypes[j, :])
        #         index.append(np.argmin(dis))
        #
        #     new_haplo = np.zeros((haplotypes.shape))
        #     for i in range(haplotypes.shape[0]):
        #         new_haplo[i, :] = np.argmax(ACGT_count(SNVmatrix[np.array(index) == i, :]), axis=1) + 1
        #     haplotypes = new_haplo.copy()
        #     pre_mec = mec
        #     mec = MEC(SNVmatrix, haplotypes)
        #     count += 1

        # TODO
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
    performance.correct_phasing_rate(consensus_sequences)
    # print(tf.squeeze(output_encoder, [0]).shape)
    print(f'{ColorCoding.OKGREEN}FINITO{ColorCoding.ENDC}')
