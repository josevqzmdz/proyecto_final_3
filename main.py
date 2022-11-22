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

from BatchGenerator import *

if __name__ == '__main__':
    # 28 francois chollet
    #e = example1()
    #print(e.example2())
    # 64
    """
    Using this NaiveDense class and this NaiveSequential class, we can create a mock
    Keras model:
    """
    """
    model = NaiveSequential([
        NaiveDense(input_size = 28 * 28, output_size = 512, activation = tf.nn.relu),
        NaiveDense(input_size = 512, output_size = 10, activation= tf.nn.softmax)
    ])
    assert len(model.weights) == 4

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255
    #batch = BatchGenerator()
    model.fit(model, train_images, train_labels, epochs=10, batch_size=128)
    """

    # We can evaluate the model by taking the argmax of its predictions over the test images,
    # and comparing it to the expected labels:
    # 66
    predictions = model(test_images)
    predictions = predictions.numpy()
    # Calling .numpy() on a
    # TensorFlow tensor converts
    # it to a NumPy tensor
    predicted_labels = np.argmax(predictions, axis=1)
    matches = predicted_labels == test_labels
    print(f"accuracy: {matches.mean(): .2f}")