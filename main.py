# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf

import hot_one


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hot_one_encoded_reads = hot_one.encode_hot_one()
    print_hi('PyCharm')
    print(tf.__version__)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
