from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, PReLU, Dropout, Conv2DTranspose, InputLayer
from tensorflow.python.keras.models import Model


class Autoencoder(Model):
    def __init__(self, input_dimension):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential(
            [
                InputLayer(
                    input_shape=(input_dimension[0], input_dimension[1], input_dimension[2])),
                # batch_size=input_dimension[0]/4)
                Conv2D(
                    filters=32,
                    kernel_size=[4, 5],
                    strides=(4, 1),
                    padding='same'),
                PReLU(),
                Dropout(0)
                # 2 more conv2D and then flatten
            ]
        )

        self.decoder = Sequential(
            [
                Dropout(0),
                Conv2DTranspose(filters=1,
                                kernel_size=[4, 5],
                                strides=(4, 1),
                                padding='same'),
            ]
        )

    def call(self, original):
        encoded = self.encoder(original)
        decoded = self.decoder(encoded)
        return decoded
