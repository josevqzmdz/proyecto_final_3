from keras.datasets import mnist
import numpy as np

from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt

class Data_Noise:

    def __init__(self):
        # Adding white noise channels or all-zeros channels to MNIST
        # 125
        (self.train_images, self.train_labels), _ = mnist.load_data()
        train_images = self.train_images.reshape((60000, 28 * 28))
        train_images = self.train_images.astype("float32") / 255
        
        train_images_with_noise_channels = np.concatenate(
            [self.train_images, np.random.random((len(train_images), 784))], axis=1
        )

        train_images_with_zeros_channels = np.concatenate(
            [self.train_images, np.zeros((len(train_images), 784))], axis=1
        )

        # Now, let’s train the model from chapter 2 on both of these training sets
        # 126
        model = self.get_model()
        history_noise = model.fit(
            train_images_with_noise_channels, self.train_labels,
            epochs=10,
            batch_size=128,
            validation_split=0.2)
        model = self.get_model()
        history_zeros = model.fit(
            train_images_with_zeros_channels, self.train_labels,
            epochs=10,
            batch_size=128,
            validation_split=0.2)

        # Plotting a validation accuracy comparison
        val_acc_noise = self.history_noise.history["val_accuracy"]
        val_acc_zeros = self.history_zeros.history["val_accuracy"]
        epochs = range(1, 11)
        plt.plot(epochs, val_acc_noise, "b-",
                 label="Validation accuracy with noise channels"
                 )
        plt.plot(epochs, val_acc_zeros, "b--",
                 label="Validation accuracy with zeros channels"
                 )
        plt.title("effect of noise channels on validation accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()

    #126
    def get_model(self):
        model = keras.Sequential([
            layers.Dense(512, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])
        model.compile(optimizer="rmsprop",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model