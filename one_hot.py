from os.path import exists

import numpy as np
import tensorflow as tf

from helpers.IO import save_tensor_file, save_file
from helpers.config import get_config

config = get_config()
data = config['data']

# snps_index = [806, 815, 817, 821, 827, 831, 832, 833, 837, 839, 860, 870, 877, 879, 917, 926, 927, 929, 933, 938,
#                   944, 945, 953, 977, 984, 986, 993, 1010, 1015, 1032, 1037, 1038, 1039, 1052, 1060, 1067, 1068, 1088,
#                   1094, 1114, 1123, 1130, 1158, 1160, 1162, 1165, 1166, 1200, 1202, 1205, 1229, 1263, 1265, 1277, 1286,
#                   1292, 1343, 1370, 1373, 1379, 1389, 1397, 1431, 1445, 1455, 1502, 1542, 1571, 1595, 1613, 1622, 1626,
#                   1627, 1635, 1645, 1658, 1673, 1685, 1697, 1716, 1721, 1766, 1781, 1787, 1807, 1808, 1811, 1847, 1850,
#                   1859, 1865, 1874, 1883, 1905, 1922, 1927, 1934, 1940, 1952, 1954, 1964, 1990, 1996, 2009, 2063, 2096,
#                   2109, 2117, 2126, 2132, 2163, 2167, 2175, 2182, 2186, 2188, 2189, 2196, 2205, 2235, 2243, 2247, 2272,
#                   2293, 2294, 2306, 2356, 2360, 2361, 2362, 2371, 2431, 2438, 2439, 2446, 2449, 2452, 2465, 2466, 2533,
#                   2557, 2566, 2602, 2611, 2668, 2683, 2686, 2707, 2710, 2793, 2800, 2806, 2827, 2839, 2852, 2884, 2909,
#                   2912, 2926, 2929, 2973, 2983, 3013, 3031, 3032, 3045, 3080, 3083, 3094, 3106, 3112, 3146, 3160, 3180,
#                   3187, 3188, 3229, 3284, 3295, 3298, 3316, 3325, 3334, 3355, 3361, 3362, 3378, 3394, 3400, 3415, 3425,
#                   3437, 3448, 3469, 3472, 3487, 3512, 3523, 3528, 3532, 3533, 3550, 3573, 3577, 3585, 3607, 3618, 3621,
#                   3643, 3664, 3674, 3678, 3679, 3705, 3719, 3721, 3727, 3733, 3746, 3760, 3799, 3811, 3820, 3829, 3838,
#                   3851, 3865, 3880, 3886, 3889, 3895, 3898, 3903, 3922, 3926, 3927, 3928, 3930, 3931, 3943, 3950, 3956,
#                   3958, 3976, 3983, 3995, 3997, 4015, 4036, 4057, 4060, 4075, 4082, 4083, 4084, 4104, 4117, 4121, 4138,
#                   4144, 4189, 4201, 4213, 4225, 4258, 4261, 4267, 4278, 4312, 4319, 4349, 4369, 4384, 4399, 4414, 4423,
#                   4442, 4448, 4492, 4504, 4513, 4525, 4528, 4531, 4532, 4546, 4560, 4565, 4598, 4600, 4603, 4609, 4627,
#                   4630, 4648, 4679, 4829, 4844, 4865, 4870, 4917, 4922, 4928, 4951, 4984, 5022, 5026, 5032, 5075, 5095,
#                   5096, 5097, 5105, 5108, 5130, 5136, 5146, 5149, 5150, 5154, 5155, 5161, 5177, 5178, 5181, 5188, 5190,
#                   5207, 5227, 5231, 5235, 5258, 5300, 5311, 5314, 5341, 5342, 5353, 5367, 5369, 5372, 5378, 5390, 5402,
#                   5405, 5406, 5409, 5418, 5419, 5420, 5422, 5429, 5435, 5458, 5477, 5490, 5500, 5503, 5509, 5511, 5512,
#                   5515, 5525, 5539, 5540, 5546, 5555, 5599, 5600, 5608, 5632, 5640, 5641, 5650, 5659, 5666, 5667, 5668,
#                   5674, 5678, 5679, 5680, 5686, 5690, 5707, 5710, 5720, 5735, 5746, 5755, 5767, 5770, 5788, 5805, 5809,
#                   5812, 5823, 5842, 5849, 6061, 6062, 6065, 6066, 6067, 6070, 6074, 6076, 6079, 6106, 6107, 6121, 6130,
#                   6132, 6139, 6171, 6180, 6183, 6190, 6194, 6196, 6200, 6204, 6209, 6231, 6234, 6241, 6258, 6270, 6272,
#                   6279, 6281, 6294, 6295, 6298, 6314, 6315, 6320, 6322, 6323, 6324, 6335, 6360, 6361, 6365, 6403, 6476,
#                   6477, 6483, 6490, 6497, 6518, 6529, 6532, 6538, 6545, 6552, 6563, 6569, 6592, 6606, 6613, 6618, 6619,
#                   6620, 6623, 6625, 6627, 6632, 6633, 6641, 6645, 6652, 6655, 6656, 6657, 6659, 6660, 6663, 6667, 6675,
#                   6682, 6691, 6701, 6708, 6711, 6722, 6723, 6730, 6731, 6745, 6746, 6748, 6752, 6756, 6757, 6760, 6764,
#                   6767, 6782, 6786, 6790, 6792, 6793, 6798, 6802, 6804, 6832, 6854, 6856, 6879, 6889, 6898, 6904, 6910,
#                   6911, 6916, 6918, 6919, 6929, 6930, 6936, 6940, 6942, 6976, 6985, 6988, 7004, 7028, 7030, 7031, 7033,
#                   7047, 7048, 7054, 7057, 7058, 7072, 7087, 7090, 7091, 7092, 7100, 7101, 7117, 7122, 7137, 7141, 7150,
#                   7151, 7153, 7155, 7156, 7174, 7175, 7176, 7178, 7182, 7184, 7188, 7196, 7201, 7220, 7227, 7229, 7232,
#                   7238, 7240, 7241, 7242, 7250, 7260, 7262, 7263, 7264, 7272, 7277, 7285, 7301, 7302, 7308, 7309, 7311,
#                   7312, 7327, 7341, 7342, 7382, 7385, 7409, 7415, 7425, 7431, 7433, 7434, 7436, 7440, 7442, 7443, 7455,
#                   7461, 7462, 7467, 7468, 7473, 7477, 7490, 7498, 7508, 7513, 7531, 7540, 7542, 7543, 7573, 7583, 7597,
#                   7603, 7604, 7605, 7606, 7607, 7608, 7609, 7611, 7612, 7614, 7616, 7617, 7618, 7642, 7645, 7692, 7703,
#                   7722, 7736, 7737, 7738, 7744, 7766, 7775, 7826, 7828, 7831, 7849, 7851, 7863, 7873, 7879, 7886, 7915,
#                   7916, 7945, 7961, 7970, 7986, 8000, 8002, 8005, 8042, 8057, 8058, 8078, 8081, 8083, 8084, 8085, 8086,
#                   8093, 8096, 8100, 8104, 8111, 8119, 8121, 8129, 8131, 8141, 8142, 8144, 8145, 8146, 8150, 8152, 8153,
#                   8154, 8156, 8159, 8165, 8173, 8175, 8179, 8195, 8208, 8212, 8243, 8254, 8271, 8273, 8300, 8321, 8322,
#                   8338, 8359, 8365, 8382, 8390, 8391, 8392, 8393, 8394, 8416, 8421, 8422, 8459, 8460, 8461, 8463, 8471,
#                   8473, 8476, 8478, 8482, 8486, 8489, 8496, 8500, 8501, 8503, 8504, 8508, 8514, 8518, 8522, 8523, 8532,
#                   8540, 8553, 8595, 8597, 8598, 8602, 8608, 8623, 8626, 8634, 8635, 8661, 8668, 8671, 8680, 8698, 8708,
#                   8714, 8717, 8720, 8726, 8727, 8730, 8733, 8737, 8741, 8745, 8746, 8747, 8756, 8757, 8764, 8782, 8783,
#                   8784, 8801, 8817, 8818, 8822, 8823, 8824, 8825, 8826, 8827, 8828, 8835, 8836, 8837, 8838, 8841, 8864,
#                   8867, 8874, 8878, 8879, 8880, 8893, 8906, 8908, 8909, 8911, 8921, 8947, 8948, 8956, 8958, 8959, 8969,
#                   8976, 8978, 8984, 8987, 8990, 9007, 9036, 9048, 9050, 9096, 9099, 9104, 9107, 9108, 9109, 9135, 9154,
#                   9161, 9166, 9180, 9185, 9191, 9192, 9193, 9194, 9199, 9200, 9223, 9242, 9248, 9252, 9267, 9268, 9280,
#                   9282, 9297, 9304, 9311, 9317, 9323, 9328, 9329, 9334, 9337, 9340, 9346, 9357, 9358, 9359, 9368, 9388,
#                   9390, 9403, 9408, 9409, 9413]
snps_index = []
if data == 'created':
    with open(config[data]['snp_positions'] + '_' + str(config[data]['n_clusters']) + '.txt', 'r') as fp:
        lines = fp.readlines()
else:
    with open(config[data]['snp_positions'], 'r') as fp:
        lines = fp.readlines()
for line in lines:
    snps_index.append(int(line[:-1]))


def switcher(base):
    return {
        'A': 0,
        'C': 1,
        'G': 2,
        'T': 3
    }.get(base, -1)


def reverse_switcher(base):
    return {
        0: 'A',
        1: 'C',
        2: 'G',
        3: 'T'
    }.get(base, '-')


# encrypt reads according to switcher
def encode_read(position, read, length):
    switched_read = []
    for base in read:
        switched_read.append(switcher(base))
    # position read in list of length of haplotype
    positioned_read = [-1] * position
    positioned_read.extend(switched_read)
    positioned_read.extend(-1 for _ in range(length - len(switched_read) - position))
    # switched_read_tensor = tf.cast(positioned_read, tf.int32)

    read_snps = [positioned_read[i] for i in snps_index]

    switched_read_tensor = tf.cast(read_snps, tf.int32)
    # one_hot encode 0 -> [1,0,0,0], 1 -> [0,1,0,0] ... -1 -> [0,0,0,0]
    one_hot_snps = tf.one_hot(switched_read_tensor, depth=4, dtype=tf.int8)
    return one_hot_snps


# def encode_read(position, read, length):
#     switched_read = []
#     for base in read:
#         switched_read.append(switcher(base))
#     # position read in list of length of haplotype
#     positioned_read = [-1] * position
#     positioned_read.extend(switched_read)
#     positioned_read.extend(-1 for _ in range(length - len(switched_read) - position))
#     # switched_read_tensor = tf.cast(positioned_read, tf.int32)
#
#     positioned_read_short = positioned_read[:length]
#
#     switched_read_tensor = tf.cast(positioned_read_short, tf.int32)
#     # one_hot encode 0 -> [1,0,0,0], 1 -> [0,1,0,0] ... -1 -> [0,0,0,0]
#     one_hot_read = tf.one_hot(switched_read_tensor, depth=4, dtype=tf.int8)
#     return one_hot_read


def encode_read_between(position, read, start, end):
    switched_read = []
    for base in read:
        switched_read.append(switcher(base))
    # position read in list of length of haplotype
    positioned_read = []
    if position > start:
        positioned_read = [-1] * (position - start)

    positioned_read.extend(switched_read)
    if position + len(read) < end:
        positioned_read.extend(-1 for _ in range(end - len(positioned_read)))
    # switched_read_tensor = tf.cast(positioned_read, tf.int32)

    positioned_read_short = positioned_read[:end - start]

    switched_read_tensor = tf.cast(positioned_read_short, tf.int32)
    # one_hot encode 0 -> [1,0,0,0], 1 -> [0,1,0,0] ... -1 -> [0,0,0,0]
    one_hot_read = tf.one_hot(switched_read_tensor, depth=4, dtype=tf.int8)
    return one_hot_read


def decode_read(sequence):
    base_read = []
    for base in sequence:
        i, _ = np.where(base == 1)
        if len(i) == 0:
            base_read.append('-')
        else:
            base_read.append(reverse_switcher(i[0]))
    decoded_sequence = ''.join(base_read)
    return decoded_sequence


def encode_sam():
    # check if file already exists
    # if data == 'illumina':
    #     if config['load'] and exists(config[data]['one_hot_path'] + '_0.npy'):
    #         print('reads are already one hot encoded')
    #         return
    # # elif data == 454:
    # #     if config['load'] and exists(config[data]['one_hot_path'] + '0.npy'):
    # #         print('reads are already one hot encoded')
    # #         return
    # else:
    path = ''
    if data == 'created':
        path = '_' + str(config[data]['n_clusters'])
    if config['load'] and exists(config[data]['one_hot_path'] + path + '.npy'):
        print('reads are already one hot encoded')
        return
    # read reads
    one_hot_encoded_reads = []
    print('reading file: encoding', config[data]['mapped_reads_path'])
    # read file line by line and one hot encode reads
    # TODO do better more abstract

    if data == 454 or data == 'illumina':
        with open(config[data]['mapped_reads_path'] + '.sam') as file:
            counter = 0
            for line in file:
                # if int(line.split('\t')[3]) > config[data]['haplotype_length']:
                #     continue
                counter += 1
                pos = int(line.split('\t')[3]) - 1
                one_hot_encoded_read = encode_read(pos, line.split('\t')[9],
                                                   config[data]['haplotype_length'])
                one_hot_encoded_reads.append(one_hot_encoded_read)
                if counter % 25000 == 0:
                    print(counter)
        one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads), axis=3)
        # save
        save_tensor_file(config[data]['one_hot_path'], one_hot_encoded_reads_tensor)
        print('Reads are one hot encoded and saved.')

    elif data == 'per_gene_454':
        start = config[data]['haplotype_start']
        end = config[data]['haplotype_end']
        # length must be dividable by 4 to work with pooling and upsampling
        end += (4 - (end - start) % 4)
        with open(config[data]['mapped_reads_path'] + '.sam') as file:
            counter = 0
            for line in file:
                pos = int(line.split('\t')[3])
                length = len(line.split('\t')[9])
                if pos > end or (pos + length) < start:
                    continue
                counter += 1
                one_hot_encoded_read = encode_read_between(pos, line.split('\t')[9], start, end)
                one_hot_encoded_reads.append(one_hot_encoded_read)
                counter += 1
                if counter % 25000 == 0:
                    print(counter)
        print(one_hot_encoded_reads)

        one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads), axis=3)
        # save
        save_tensor_file(config[data]['one_hot_path'], one_hot_encoded_reads_tensor)
        print('Reads are one hot encoded and saved.')


    elif data == 'created':
        with open(config[data]['mapped_reads_path'] + '_' + str(config[data]['n_clusters']) + '.sam') as file:
            counter = 0
            for line in file:
                pos = int(line.split('\t')[3]) - 1

                one_hot_encoded_read = encode_read(pos, line.split('\t')[9],
                                                   config[data]['haplotype_length'])
                one_hot_encoded_reads.append(one_hot_encoded_read)
                counter += 1
                if counter % 25000 == 0:
                    print(counter)
        if len(one_hot_encoded_reads) % 4 != 0:
            empty_read = encode_read(0, '', config[data]['haplotype_length'])
            for i in range(len(one_hot_encoded_reads) % 4):
                one_hot_encoded_reads.append(empty_read)
        one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads), axis=3)

        # save
        save_tensor_file(config[data]['one_hot_path'] + '_' + str(config[data]['n_clusters']),
                         one_hot_encoded_reads_tensor)
        print('Reads are one hot encoded and saved.')

    # elif data == 454:
    #     with open(config[data]['mapped_reads_path']) as file:
    #         counter = 0
    #         file_index = 0
    #         for line in file:
    #             one_hot_encoded_read = encode_read(int(line.split('\t')[3]), line.split('\t')[9],
    #                                                config[data]['haplotype_length'])
    #             one_hot_encoded_reads.append(one_hot_encoded_read)
    #             counter += 1
    #             if counter % 25000 == 0:
    #                 print(counter)
    #             if counter % config[data]['batch_size'] == 0:
    #                 one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads),
    #                                                               axis=3)
    #                 # save
    #                 print('Reads are one hot encoded and saved. file index:', file_index)
    #
    #                 save_tensor_file(config[data]['one_hot_path'] + str(file_index),
    #                                  one_hot_encoded_reads_tensor)
    #                 file_index += 1
    #                 one_hot_encoded_reads = []
    #     one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads), axis=3)
    #
    #     # save
    #
    #     save_tensor_file(config[data]['one_hot_path'] + str(file_index), one_hot_encoded_reads_tensor)
    #     print('Reads are one hot encoded and saved.')
    else:
        file_index = 0
        with open(config[data]['mapped_reads_path'] + '.sam') as file:
            counter = 0
            for line in file:
                one_hot_encoded_read = encode_read(int(line.split('\t')[3]), line.split('\t')[9],
                                                   config[data]['haplotype_length'])
                one_hot_encoded_reads.append(one_hot_encoded_read)
                counter += 1
                if counter % 25000 == 0:
                    print(counter)
                    if counter % 200000 == 0:
                        one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads),
                                                                      axis=3)
                        # save
                        print('Reads are one hot encoded and saved. file index:', file_index)

                        save_tensor_file(config[data]['one_hot_path'] + '_' + str(file_index),
                                         one_hot_encoded_reads_tensor)
                        file_index += 1
                        one_hot_encoded_reads = []
        one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_reads),
                                                      axis=3)
        # save
        save_tensor_file(config[data]['one_hot_path'] + '_' + str(file_index),
                         one_hot_encoded_reads_tensor)
        print('Reads are one hot encoded and saved. file index:', file_index)
        one_hot_encoded_reads = []
    # return file_index


def encode_fasta():
    one_hot_encoded_sequences = []
    names_of_variants = []
    print('reading file: REF.fasta')
    # read file line by line and one hot encode reads
    with open(config[data]['aligned_ref_path'] + '_' + str(config[data]['n_clusters']) + '.fasta') as file:
        for line in file:
            if line[0] != '>':
                one_hot_encoded_sequence = encode_read(0, line,
                                                       config[data]['haplotype_length'])
                one_hot_encoded_sequences.append(one_hot_encoded_sequence)

    one_hot_encoded_reads_tensor = tf.expand_dims(tf.convert_to_tensor(one_hot_encoded_sequences), axis=3)

    # save
    save_tensor_file(config[data]['aligned_ref_path'] + '_' + str(config[data]['n_clusters']),
                     one_hot_encoded_reads_tensor)
    print('Reads are one hot encoded and saved.')


def decode(encoded_sequences):
    decoded_sequences = []
    for i in range(len(encoded_sequences)):
        decoded_sequences.append('>' + str(i))
        decoded_sequences.append(decode_read(encoded_sequences[i]))
    decoded_sequences_string = '\n'.join(decoded_sequences)
    save_file(config[data]['output_path'], decoded_sequences_string)
    return decoded_sequences

# TODO create reads of length max length and return reads as done in create data
