from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import InputLayer, Conv2D, PReLU, Dropout, Flatten, Dense, Reshape, Conv2DTranspose, \
    AvgPool2D, UpSampling2D
from tensorflow.python.keras.models import Model, Input

from helpers import config

config = config.get_config()
data = config['data']


def get_autoencoder_key_points(input_shape):
    # input layer
    base_filter = 16
    model_input = Input(
        shape=(input_shape[0], input_shape[1], input_shape[2]), name='encoder_Input_0')
    # shape=(input_shape[0], input_shape[1], input_shape[2]), batch_size=batch_size, name='encoder_Input_0')

    # 1. layer
    encoder_conv2d_1 = Conv2D(
        filters=base_filter,
        kernel_size=[5, 4],
        strides=(1, 4),
        padding='same',
        name='encoder_Conv2D_1')(model_input)
    encoder_prelu_1 = PReLU(name='encoder_PReLU_1')(encoder_conv2d_1)
    encoder_dropout_1 = Dropout(0, name='encoder_Dropout_1')(encoder_prelu_1)

    # 2. layer
    encoder_conv2d_2 = Conv2D(filters=base_filter * 2,
                              kernel_size=[5, 1],
                              strides=(1, 1),
                              padding='same',
                              name='encoder_Conv2D_2')(encoder_dropout_1)
    encoder_prelu_2 = PReLU(name='encoder_PReLU_2')(encoder_conv2d_2)
    encoder_dropout_2 = Dropout(0, name='encoder_Dropout_2')(encoder_prelu_2)

    # 3. layer
    encoder_conv2d_3 = Conv2D(filters=base_filter * 4,
                              kernel_size=[3, 1],
                              strides=(1, 1),
                              padding='same', name='encoder_Conv2D_3')(encoder_dropout_2)
    encoder_prelu_3 = PReLU(name='encoder_PReLU_3')(encoder_conv2d_3)
    encoder_dropout_3 = Dropout(0, name='encoder_Dropout_3')(encoder_prelu_3)

    # 4. layer
    encoder_flatten_4 = Flatten(name='encoder_Flatten_4')(encoder_dropout_3)
    # 5. layer
    # TODO replace all others
    output_encoder = Dense(units=config[data]['haplotype_length'] / 4, name='encoder_Dense_5')(
        encoder_flatten_4)

    decoder_dense_1 = Dense(units=base_filter * 4 * input_shape[0], name='decoder_Dense_1')(output_encoder)
    decoder_prelu_2 = PReLU(name='decoder_PReLU_2')(decoder_dense_1)
    decoder_reshape_2 = Reshape((input_shape[0], 1, base_filter * 4), name='decoder_Reshape_2')(decoder_prelu_2)
    decoder_conv2dtranspose_3 = Conv2DTranspose(filters=base_filter * 2,
                                                kernel_size=[3, 1],
                                                strides=(1, 1),
                                                padding='same',
                                                name='decoder_Conv2DTranspose_3')(decoder_reshape_2)
    decoder_prelu_3 = PReLU(name='decoder_PReLU_3')(decoder_conv2dtranspose_3)
    decoder_dropout_3 = Dropout(0, name='decoder_Dropout_3')(decoder_prelu_3)
    decoder_conv2dtranspose_4 = Conv2DTranspose(filters=base_filter,
                                                kernel_size=[5, 1],
                                                strides=(1, 1),
                                                padding='same',
                                                name='decoder_Conv2DTranspose_4')(decoder_dropout_3)
    decoder_prelu_4 = PReLU(name='decoder_PReLU_4')(decoder_conv2dtranspose_4)
    decoder_dropout_4 = Dropout(0, name='decoder_Dropout_4')(decoder_prelu_4)
    decoder_conv2dtranspose_5 = Conv2DTranspose(filters=1,
                                                kernel_size=[5, 4],
                                                strides=(1, 4),
                                                padding='same',
                                                name='decoder_Conv2DTranspose_5')(decoder_dropout_4)
    decoder_prelu_5 = PReLU(name='decoder_PReLU_5')(decoder_conv2dtranspose_5)
    decoder_output = Dropout(0, name='decoder_Dropout_5')(decoder_prelu_5)

    return model_input, output_encoder, decoder_output


def get_autoencoder_key_points_with_pooling(input_shape):
    # input layer
    base_filter = 16
    model_input = Input(
        shape=(input_shape[0], input_shape[1], input_shape[2]), name='encoder_Input_0')
    # shape=(input_shape[0], input_shape[1], input_shape[2]), batch_size=batch_size, name='encoder_Input_0')

    # 1. layer
    encoder_conv2d_1 = Conv2D(
        filters=base_filter,
        kernel_size=[5, 4],
        strides=(1, 4),
        padding='same',
        name='encoder_Conv2D_1')(model_input)
    encoder_prelu_1 = PReLU(name='encoder_PReLU_1')(encoder_conv2d_1)
    encoder_dropout_1 = Dropout(0, name='encoder_Dropout_1')(encoder_prelu_1)

    # 2. layer
    encoder_conv2d_2 = Conv2D(filters=base_filter * 2,
                              kernel_size=[5, 1],
                              strides=(1, 1),
                              padding='same',
                              name='encoder_Conv2D_2')(encoder_dropout_1)
    encoder_prelu_2 = PReLU(name='encoder_PReLU_2')(encoder_conv2d_2)
    encoder_pooling_2 = AvgPool2D((1, 1), strides=(2, 2))(encoder_prelu_2)
    encoder_dropout_2 = Dropout(0, name='encoder_Dropout_2')(encoder_pooling_2)

    # 3. layer
    encoder_conv2d_3 = Conv2D(filters=base_filter * 4,
                              kernel_size=[3, 1],
                              strides=(1, 1),
                              padding='same', name='encoder_Conv2D_3')(encoder_dropout_2)
    encoder_prelu_3 = PReLU(name='encoder_PReLU_3')(encoder_conv2d_3)
    encoder_pooling_3 = AvgPool2D((1, 1), strides=(2, 2))(encoder_prelu_3)
    encoder_dropout_3 = Dropout(0, name='encoder_Dropout_3')(encoder_pooling_3)

    # 4. layer
    encoder_flatten_4 = Flatten(name='encoder_Flatten_4')(encoder_dropout_3)
    # 5. layer
    # TODO replace all others
    output_encoder = Dense(units=config[data]['haplotype_length'] / 4, name='encoder_Dense_5')(
        encoder_flatten_4)
    # output_encoder = Dense(units=9852 / 4, name='encoder_Dense_5')(
    #     encoder_flatten_4)

    decoder_dense_1 = Dense(units=encoder_flatten_4.shape[1], name='decoder_Dense_1')(output_encoder)
    decoder_prelu_2 = PReLU(name='decoder_PReLU_2')(decoder_dense_1)
    decoder_reshape_2 = Reshape((encoder_dropout_3.shape[1], encoder_dropout_3.shape[2], encoder_dropout_3.shape[3]),
                                name='decoder_Reshape_2')(decoder_prelu_2)
    decoder_conv2dtranspose_3 = Conv2DTranspose(filters=base_filter * 2,
                                                kernel_size=[3, 1],
                                                strides=(1, 1),
                                                padding='same',
                                                name='decoder_Conv2DTranspose_3')(decoder_reshape_2)
    decoder_prelu_3 = PReLU(name='decoder_PReLU_3')(decoder_conv2dtranspose_3)
    decoder_up_samling_3 = UpSampling2D((2, 1))(decoder_prelu_3)
    decoder_dropout_3 = Dropout(0, name='decoder_Dropout_3')(decoder_up_samling_3)
    decoder_conv2dtranspose_4 = Conv2DTranspose(filters=base_filter,
                                                kernel_size=[5, 1],
                                                strides=(1, 1),
                                                padding='same',
                                                name='decoder_Conv2DTranspose_4')(decoder_dropout_3)
    decoder_prelu_4 = PReLU(name='decoder_PReLU_4')(decoder_conv2dtranspose_4)
    decoder_up_samling_4 = UpSampling2D((2, 1))(decoder_prelu_4)
    decoder_dropout_4 = Dropout(0, name='decoder_Dropout_4')(decoder_up_samling_4)
    decoder_conv2dtranspose_5 = Conv2DTranspose(filters=1,
                                                kernel_size=[5, 4],
                                                strides=(1, 4),
                                                padding='same',
                                                name='decoder_Conv2DTranspose_5')(decoder_dropout_4)
    decoder_prelu_5 = PReLU(name='decoder_PReLU_5')(decoder_conv2dtranspose_5)
    decoder_output = Dropout(0, name='decoder_Dropout_5')(decoder_prelu_5)

    return model_input, output_encoder, decoder_output


def get_autoencoder_key_points_with_pooling_with_relu(input_shape):
    # input layer
    base_filter = 16
    model_input = Input(
        shape=(input_shape[0], input_shape[1], input_shape[2]), name='encoder_Input_0')
    # shape=(input_shape[0], input_shape[1], input_shape[2]), batch_size=batch_size, name='encoder_Input_0')

    # 1. layer
    encoder_conv2d_1 = Conv2D(
        filters=base_filter,
        kernel_size=[5, 4],
        strides=(1, 4),
        padding='same',
        activation='relu',
        name='encoder_Conv2D_1')(model_input)
    encoder_prelu_1 = PReLU(name='encoder_PReLU_1')(encoder_conv2d_1)
    AvgPool2D(2, strides=2)(encoder_prelu_1)
    encoder_dropout_1 = Dropout(0, name='encoder_Dropout_1')(encoder_prelu_1)

    # 2. layer
    encoder_conv2d_2 = Conv2D(filters=base_filter * 2,
                              kernel_size=[5, 1],
                              strides=(1, 1),
                              padding='same',
                              activation='relu',
                              name='encoder_Conv2D_2')(encoder_dropout_1)
    encoder_prelu_2 = PReLU(name='encoder_PReLU_2')(encoder_conv2d_2)
    encoder_dropout_2 = Dropout(0, name='encoder_Dropout_2')(encoder_prelu_2)

    # 3. layer
    encoder_conv2d_3 = Conv2D(filters=base_filter * 4,
                              kernel_size=[3, 1],
                              strides=(1, 1),
                              padding='same',
                              activation='relu',
                              name='encoder_Conv2D_3')(encoder_dropout_2)
    encoder_prelu_3 = PReLU(name='encoder_PReLU_3')(encoder_conv2d_3)
    encoder_dropout_3 = Dropout(0, name='encoder_Dropout_3')(encoder_prelu_3)

    # 4. layer
    encoder_flatten_4 = Flatten(name='encoder_Flatten_4')(encoder_dropout_3)
    # 5. layer
    # TODO replace all others
    output_encoder = Dense(units=config[data]['haplotype_length'] / 4, name='encoder_Dense_5')(
        encoder_flatten_4)

    decoder_dense_1 = Dense(units=base_filter * 4 * input_shape[0], name='decoder_Dense_1')(output_encoder)
    decoder_prelu_2 = PReLU(name='decoder_PReLU_2')(decoder_dense_1)
    decoder_reshape_2 = Reshape((input_shape[0], 1, base_filter * 4), name='decoder_Reshape_2')(decoder_prelu_2)
    decoder_conv2dtranspose_3 = Conv2DTranspose(filters=base_filter * 2,
                                                kernel_size=[3, 1],
                                                strides=(1, 1),
                                                padding='same',
                                                activation='relu',
                                                name='decoder_Conv2DTranspose_3')(decoder_reshape_2)
    decoder_prelu_3 = PReLU(name='decoder_PReLU_3')(decoder_conv2dtranspose_3)
    decoder_dropout_3 = Dropout(0, name='decoder_Dropout_3')(decoder_prelu_3)
    decoder_conv2dtranspose_4 = Conv2DTranspose(filters=base_filter,
                                                kernel_size=[5, 1],
                                                strides=(1, 1),
                                                padding='same',
                                                name='decoder_Conv2DTranspose_4')(decoder_dropout_3)
    decoder_prelu_4 = PReLU(name='decoder_PReLU_4')(decoder_conv2dtranspose_4)
    decoder_dropout_4 = Dropout(0, name='decoder_Dropout_4')(decoder_prelu_4)
    decoder_conv2dtranspose_5 = Conv2DTranspose(filters=1,
                                                kernel_size=[5, 4],
                                                strides=(1, 4),
                                                padding='same',
                                                name='decoder_Conv2DTranspose_5')(decoder_dropout_4)
    decoder_prelu_5 = PReLU(name='decoder_PReLU_5')(decoder_conv2dtranspose_5)
    decoder_output = Dropout(0, name='decoder_Dropout_5')(decoder_prelu_5)

    return model_input, output_encoder, decoder_output


# class Dense(Layer):


class Autoencoder(Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential(name="encoder", layers=[
            InputLayer(
                input_shape=(input_shape[0], input_shape[1], input_shape[2]), name='encoder_Input_0'),
            # batch_size=input_dimension[0]/4)
            # 1. Layer
            Conv2D(
                filters=32,
                kernel_size=[5, 4],
                strides=(1, 4),
                padding='same',
                name='encoder_Conv2D_1'),
            PReLU(name='encoder_PReLU_1'),
            Dropout(0, name='encoder_Dropout_1'),

            # 2. Layer
            Conv2D(filters=64,
                   kernel_size=[5, 1],
                   strides=(1, 1),
                   padding='same',
                   name='encoder_Conv2D_2'),
            PReLU(name='encoder_PReLU_2'),
            Dropout(0, name='encoder_Dropout_2'),

            # 3. Layer
            Conv2D(filters=128,
                   kernel_size=[3, 1],
                   strides=(1, 1),
                   padding='same', name='encoder_Conv2D_3'),
            PReLU(name='encoder_PReLU_3'),
            Dropout(0, name='encoder_Dropout_3'),
            Flatten(name='encoder_Flatten_4'),
            Dense(units=config[data]['haplotype_length'] / 4, name='encoder_Dense_5')
        ])

        # self.clustering_1 = ClusteringLayer(n_clusters=1, name='clustering_1')
        # self.clustering_2 = ClusteringLayer(n_clusters=2, name='clustering_2')
        # self.clustering_3 = ClusteringLayer(n_clusters=3, name='clustering_3')
        # self.clustering_4 = ClusteringLayer(n_clusters=4, name='clustering_4')
        # self.clustering_5 = ClusteringLayer(n_clusters=5, name='clustering_5')
        # self.clustering_6 = ClusteringLayer(n_clusters=6, name='clustering_6')
        # self.clustering_7 = ClusteringLayer(n_clusters=7, name='clustering_7')
        # self.clustering_8 = ClusteringLayer(n_clusters=8, name='clustering_8')
        # self.clustering_9 = ClusteringLayer(n_clusters=9, name='clustering_9')
        # self.clustering_10 = ClusteringLayer(n_clusters=10, name='clustering_10')

        self.decoder = Sequential(name="decoder", layers=[
            Dense(units=128 * input_shape[0], name='decoder_Dense_1'),
            PReLU(name='decoder_PReLU_2'),
            Reshape((input_shape[0], 1, 128), name='decoder_Reshape_2'),
            Conv2DTranspose(filters=64,
                            kernel_size=[3, 1],
                            strides=(1, 1),
                            padding='same',
                            name='decoder_Conv2DTranspose_3'),
            PReLU(name='decoder_PReLU_3'),
            Dropout(0, name='decoder_Dropout_3'),
            Conv2DTranspose(filters=32,
                            kernel_size=[5, 1],
                            strides=(1, 1),
                            padding='same',
                            name='decoder_Conv2DTranspose_4'),
            PReLU(name='decoder_PReLU_4'),
            Dropout(0, name='decoder_Dropout_4'),
            Conv2DTranspose(filters=1,
                            kernel_size=[5, 4],
                            strides=(1, 4),
                            padding='same',
                            name='decoder_Conv2DTranspose_5'),
            PReLU(name='decoder_PReLU_5'),
            Dropout(0, name='decoder_Dropout_5'),
        ]
                                  )

    def call(self, original):
        encoded = self.encoder(original)
        decoded = self.decoder(encoded)
        # clustered = [self.clustering_1(encoded),
        #              self.clustering_2(encoded),
        #              self.clustering_3(encoded),
        #              self.clustering_4(encoded),
        #              self.clustering_5(encoded),
        #              self.clustering_6(encoded),
        #              self.clustering_7(encoded),
        #              self.clustering_8(encoded),
        #              self.clustering_9(encoded),
        #              self.clustering_10(encoded)]

        return decoded  # , clustered
