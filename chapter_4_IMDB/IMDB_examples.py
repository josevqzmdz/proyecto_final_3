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


class IMDB_examples:

    def __init__(self):
        # Plotting the training and validation loss

        # 99
        """
        The first argument being passed to each Dense layer is the number of
        units in the
        layer: the dimensionality of representation space of the layer.
        You remember from
        chapters 2 and 3 that each such Dense layer with a relu activation
         implements the fol-
        lowing chain of tensor operations:
        output = relu(dot(input, W) + b)
        """
        train_data[0]

        train_labels[0]

        word_index = imdb.get_word_index()
        reverse_word_index = dict(
            [(value, key) for (key, value) in word_index.items()])
        decoded_review = " ".join(
            [reverse_word_index.get(i - 3, "?") for i in train_data[0]])

        model = keras.Sequential([
            layers.Dense(16, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        x_train = self.vectorize_sequences(train_data)
        x_test = self.vectorize_sequences(test_data)
        x_train[0]
        # You should also vectorize your labels, which is straightforward
        # Now the data is ready to be fed into a neural network.
        y_train = np.asarray(train_labels).astype("float32")
        y_test = np.asarray(test_labels).astype("float32")

        x_val = x_train[:10000]
        partial_x_train = x_train[:10000]
        y_val = y_train[:10000]
        partial_y_train = y_train[:10000]

        model.compile(optimizer="rmsprop",
                      loss="binary_crossentropy",
                      metrics=["accuracy"])

        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=20,
                            batch_size=512,
                            validation_data=(x_val, y_val))
        history_dict = history.history
        # The dictionary contains four entries: one per metric that was being monitored during
        # training and during validation
        history_dict.keys()
        #104
        loss_values = history_dict["loss"]
        val_loss_values = history_dict["val_loss"]
        epochs = range(1, len(loss_values) + 1)
        # bo = blue dot
        plt.plot(epochs, loss_values, "bo", label="Training loss")
        # b -> blue line
        plt.plot(epochs, val_loss_values, "b", label="Validation loss")
        plt.title("training and validation loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

        # Plotting the training and validation accuracy
        plt.clf()
        acc = history_dict["accuracy"]
        val_acc = history_dict["val_accuracy"]
        plt.plot(epochs, acc, "bo", label="Training acc")
        plt.plot(epochs, val_acc, "b", label="validation acc")
        plt.title("training and validation accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.show()

    # 98
    def vectorize_sequences(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            for j in sequence:
                results[i, j] = 1
        return results



    def training_model(self):
        # Plotting the training and validation loss
        print()


