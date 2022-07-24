import tensorflow as tf

import create_data

if __name__ == '__main__':
    print("cool")
    # data = one_hot.encode()
    data = create_data.create_reads()

    Input = tf.keras.Input(shape=(data.shape[0], 4),
                           name='Input')
    print(tf.__version__)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
