from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import InputLayer, Conv2D, PReLU, Dropout, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.python.keras.models import Model


class Autoencoder(Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential(name="encoder", layers=[
            InputLayer(
                input_shape=(input_shape[0], input_shape[1], input_shape[2])),
            # batch_size=input_dimension[0]/4)
            # 1. Layer
            Conv2D(
                filters=32,
                kernel_size=[5, 4],
                strides=(1, 4),
                padding='same'),
            PReLU(),
            Dropout(0),

            # 2. Layer
            Conv2D(filters=64,
                   kernel_size=[5, 1],
                   strides=(1, 1),
                   padding='same'),
            PReLU(),
            Dropout(0),

            # 3. Layer
            Conv2D(filters=128,
                   kernel_size=[3, 1],
                   strides=(1, 1),
                   padding='same'),
            PReLU(),
            Dropout(0),
            Flatten(),
            # Dense(units=input_shape[1] / 4)
            Dense(units=2500)

        ])

        self.decoder = Sequential(name="encoder", layers=[
            Dense(units=128 * input_shape[0]),
            PReLU(),
            Reshape((input_shape[0], 1, 128)),
            Conv2DTranspose(filters=64,
                            kernel_size=[3, 1],
                            strides=(1, 1),
                            padding='same'),
            PReLU(),
            Dropout(0),
            Conv2DTranspose(filters=32,
                            kernel_size=[5, 1],
                            strides=(1, 1),
                            padding='same'),
            PReLU(),
            Dropout(0),
            Conv2DTranspose(filters=1,
                            kernel_size=[5, 4],
                            strides=(1, 4),
                            padding='same'),
            Dropout(0),
        ]
                                  )

    def call(self, original):
        encoded = self.encoder(original)
        decoded = self.decoder(encoded)
        # clustered = self.encoder(original)
        return decoded
