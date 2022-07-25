import sys

import numpy as np
import tensorflow as tf
import yaml

import create_data
import one_hot
from helpers.colors_coding import ColorCoding
from models.autoencoder import Autoencoder

with open("./config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

if __name__ == '__main__':
    # TODO print config and info
    print("Python version: {}.{}.{}".format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
    print("Tensorflow version: {}".format(tf.__version__))
    if config["real_data"]:
        print(f"Working with {ColorCoding.OKCYAN}454{ColorCoding.ENDC} reads")
        data = one_hot.encode()
    else:
        print(
            f"Working with {ColorCoding.OKCYAN}created{ColorCoding.ENDC} reads")
        data = create_data.create_reads()
    batch_size = int(np.ceil(data.shape[0] / 20))
    autoencoder = Autoencoder(data.shape[1:])

    autoencoder.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())
    autoencoder.build(input_shape=(data.shape))
    autoencoder.fit(x=data, y=data,
                    epochs=2,
                    batch_size=batch_size,
                    shuffle=True,
                    verbose=1)
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()
    autoencoder.summary()
    output = autoencoder.predict(x=tf.expand_dims(data[0], axis=0))
    print(data[0][0][3][0])
    print(output[0][3][0])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
