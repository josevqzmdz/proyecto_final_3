# Deep Learning with Python by François Chollet

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from example1 import *

from keras import layers

if __name__ == '__main__':
    # 28 francois chollet
    e = example1()
    print(e.example2())
