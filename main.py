# Deep Learning with Python by François Chollet

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#97
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
num_words=10000)

# how to import classes inside a folder
# https://favtutor.com/blogs/import-class-from-another-file-python
from chapter_4_IMDB.IMDB_examples import *

from Training_baby_steps import *
from example1 import *
from NaiveDense import *
from NaiveSequential import *
from SimpleDense import *

from keras import layers

from BatchGenerator import *

if __name__ == '__main__':
    print()