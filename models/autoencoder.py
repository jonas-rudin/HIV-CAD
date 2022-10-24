import numpy as np
from tensorflow.python.keras.layers import Conv2D, PReLU, Dropout, Flatten, Dense, Reshape, Conv2DTranspose, \
    AvgPool2D, UpSampling2D
from tensorflow.python.keras.models import Input

from helpers import config

config = config.get_config()
data = config['data']


def get_autoencoder_key_points(input_shape):
    # input layer
    base_filter = 32
    print(input_shape)
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
    latent_space = Dense(units=int(np.ceil(input_shape[0] / 2)), name='encoder_Dense_5')(
        encoder_flatten_4)

    decoder_dense_1 = Dense(units=base_filter * 4 * input_shape[0], name='decoder_Dense_1')(latent_space)
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

    return model_input, latent_space, decoder_output


def get_autoencoder_key_points_with_pooling(input_shape):
    # input layer
    base_filter = 32
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

    encoder_pooling_2 = AvgPool2D((1, 1), strides=(2, 2))(encoder_dropout_2)
    # 3. layer
    encoder_conv2d_3 = Conv2D(filters=base_filter * 4,
                              kernel_size=[3, 1],
                              strides=(1, 1),
                              padding='same', name='encoder_Conv2D_3')(encoder_pooling_2)
    encoder_prelu_3 = PReLU(name='encoder_PReLU_3')(encoder_conv2d_3)
    encoder_dropout_3 = Dropout(0, name='encoder_Dropout_3')(encoder_prelu_3)
    encoder_pooling_3 = AvgPool2D((1, 1), strides=(2, 2))(encoder_dropout_3)

    # 4. layer
    encoder_flatten_4 = Flatten(name='encoder_Flatten_4')(encoder_pooling_3)
    # 5. layer
    # TODO replace all others
    latent_space = Dense(units=int(np.ceil(input_shape[0] / 4)), name='encoder_Dense_5')(
        encoder_flatten_4)
    # output_encoder = Dense(units=9852 / 4, name='encoder_Dense_5')(
    #     encoder_flatten_4)

    decoder_dense_1 = Dense(units=encoder_flatten_4.shape[1], name='decoder_Dense_1')(latent_space)
    decoder_prelu_2 = PReLU(name='decoder_PReLU_2')(decoder_dense_1)
    decoder_reshape_2 = Reshape((encoder_pooling_3.shape[1], encoder_pooling_3.shape[2], encoder_pooling_3.shape[3]),
                                name='decoder_Reshape_2')(decoder_prelu_2)
    decoder_up_samling_3 = UpSampling2D((2, 1))(decoder_reshape_2)
    decoder_conv2dtranspose_3 = Conv2DTranspose(filters=base_filter * 2,
                                                kernel_size=[3, 1],
                                                strides=(1, 1),
                                                padding='same',
                                                name='decoder_Conv2DTranspose_3')(decoder_up_samling_3)
    decoder_prelu_3 = PReLU(name='decoder_PReLU_3')(decoder_conv2dtranspose_3)
    decoder_dropout_3 = Dropout(0, name='decoder_Dropout_3')(decoder_prelu_3)
    decoder_up_samling_4 = UpSampling2D((2, 1))(decoder_dropout_3)
    decoder_conv2dtranspose_4 = Conv2DTranspose(filters=base_filter,
                                                kernel_size=[5, 1],
                                                strides=(1, 1),
                                                padding='same',
                                                name='decoder_Conv2DTranspose_4')(decoder_up_samling_4)
    decoder_prelu_4 = PReLU(name='decoder_PReLU_4')(decoder_conv2dtranspose_4)
    decoder_dropout_4 = Dropout(0, name='decoder_Dropout_4')(decoder_prelu_4)
    decoder_conv2dtranspose_5 = Conv2DTranspose(filters=1,
                                                kernel_size=[5, 4],
                                                strides=(1, 4),
                                                padding='same',
                                                name='decoder_Conv2DTranspose_5')(decoder_dropout_4)
    decoder_prelu_5 = PReLU(name='decoder_PReLU_5')(decoder_conv2dtranspose_5)
    decoder_output = Dropout(0, name='decoder_Dropout_5')(decoder_prelu_5)

    return model_input, latent_space, decoder_output


def get_CAECseq(shape):
    drop_prob = 0

    len_code = int(np.ceil(shape[1] / 4))
    kernel = [5, 5, 3]
    filters = [32, 64, 128]
    model_input = Input(
        shape=(shape[0], shape[1], shape[2]), name='model_input')
    # convolutional layer - 1
    conv_1 = Conv2D(filters=filters[0],
                    kernel_size=[4, kernel[0]],
                    strides=(4, 1),
                    padding='same',
                    name='conv_1')(model_input)
    PRelu_1 = PReLU(name='PRelu_1')(conv_1)
    drop_1 = Dropout(drop_prob, name='drop_1')(PRelu_1)

    conv_2 = Conv2D(filters=filters[1],
                    kernel_size=[1, kernel[1]],
                    strides=(1, 1),
                    padding='same',
                    name='conv_2')(drop_1)
    PRelu_2 = PReLU(name='PRelu_2')(conv_2)
    drop_2 = Dropout(drop_prob, name='drop_2')(PRelu_2)

    conv_3 = Conv2D(filters=filters[2],
                    kernel_size=[1, kernel[2]],
                    strides=(1, 1),
                    padding='same',
                    name='conv_3')(drop_2)
    PRelu_3 = PReLU(name='PRelu_3')(conv_3)
    drop_3 = Dropout(drop_prob, name='drop_3')(PRelu_3)
    # flatten
    flatten = Flatten(name='flatten')(drop_3)
    # code
    code = Dense(units=len_code,
                 name='code')(flatten)
    # dense
    dense_1 = Dense(units=flatten.shape[1],
                    name='dense_1')(code)
    PRelu_4 = PReLU(name='PRelu_4')(dense_1)
    # reshape
    reshape = Reshape((drop_3.shape[1], drop_3.shape[2], drop_3.shape[3]),
                      name='reshape')(PRelu_4)
    # transposed convolution layer
    convT_1 = Conv2DTranspose(filters=filters[1],
                              kernel_size=[1, kernel[2]],
                              strides=(1, 1),
                              padding='same',
                              name='convT_1')(reshape)
    PRelu_5 = PReLU(name='PRelu_5')(convT_1)
    drop_4 = Dropout(drop_prob, name='drop_4')(PRelu_5)

    convT_2 = Conv2DTranspose(filters=filters[0],
                              kernel_size=[1, kernel[1]],
                              strides=(1, 1),
                              padding='same',
                              name='convT_2')(drop_4)
    PRelu_6 = PReLU(name='PRelu_6')(convT_2)
    drop_5 = Dropout(drop_prob, name='drop_5')(PRelu_6)

    convT_3 = Conv2DTranspose(filters=1,
                              kernel_size=[4, kernel[0]],
                              strides=(4, 1),
                              padding='same',
                              name='convT_3')(drop_5)
    drop_6 = Dropout(drop_prob, name='drop_6')(convT_3)
    return model_input, code, drop_6
