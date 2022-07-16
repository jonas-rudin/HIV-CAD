# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf

import create_data

# Press the green button in the gutter to run the script.
# import one_hot

if __name__ == '__main__':
    # one_hot_encoded_reads = one_hot.encode()
    print(create_data.create_reads())
    print(tf.__version__)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
