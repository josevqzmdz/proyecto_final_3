# Deep Learning with Python by François Chollet

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from example1 import *
from NaiveDense import *
from NaiveSequential import *

from keras import layers

if __name__ == '__main__':
    # 28 francois chollet
    #e = example1()
    #print(e.example2())
    # 64
    """
    Using this NaiveDense class and this NaiveSequential class, we can create a mock
    Keras model:
    """
    model = NaiveSequential([
        NaiveDense(input_size = 28 * 28, output_size = 512, activation = tf.nn.relu),
        NaiveDense(input_size = 512, output_size = 10, activation= tf.nn.softmax)
    ])
    assert len(model.weights) == 4