from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

import numpy as np

from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt

class Data_Noise:

    def __init__(self):
        # Adding white noise channels or all-zeros channels to MNIST
        # 125

        (train_images, train_labels), _ = mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28))
        train_images = train_images.astype("float32") / 255

        train_images_with_noise_channels = np.concatenate(
            [train_images,
             np.random.random((len(train_images), 28, 28))],
            axis=1)

        train_images_with_zeros_channels = np.concatenate(
            [train_images, np.zeros((len(train_images), 28, 28))],
            axis=1)

        # Now, let’s train the model from chapter 2 on both of these training sets
        # 126
        """
        - one epoch = one forward pass and one backward pass of all the training
         examples
        - batch size = the number of training examples in one forward/backward 
        pass. The higher the batch size, the more memory space you'll need.
        - number of iterations = number of passes, each pass using [batch size]
         number of examples. To be clear, one pass = one forward pass + 
         one backward pass (we do not count the forward pass and 
         backward pass as two different passes).
        """
        model = self.get_model()
        history_noise = model.fit(
            train_images_with_noise_channels,
            train_labels,
            epochs=10,
            batch_size=64,
            validation_split=0.2)

        model = self.get_model()
        history_zeros = model.fit(
            train_images_with_zeros_channels,
            train_labels,
            epochs=10,
            batch_size=64,
            validation_split=0.2)

        # Plotting a validation accuracy comparison
        val_acc_noise = history_noise.history["val_accuracy"]
        val_acc_zeros = history_zeros.history["val_accuracy"]
        epochs = range(1, 11)
        plt.plot(epochs, val_acc_noise, "b-",
                 label="Validation accuracy with noise channels")
        plt.plot(epochs, val_acc_zeros, "b--",
                 label="Validation accuracy with zeros channels")
        plt.title("effect of noise channels on validation accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()

    #126
    def get_model(self):
        model = keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])

        model.compile(optimizer="rmsprop",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model