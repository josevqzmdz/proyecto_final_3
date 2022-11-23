from keras.datasets import mnist
import numpy as np

from tensorflow import keras
from keras import layers

class Data_Noise:
    def __init__(self):
        # Adding white noise channels or all-zeros channels to MNIST
        # 125
        (self.train_images, train_labels), _ = mnist.load_data()
        train_images = self.train_images.reshape((60000, 28 * 28))
        train_images = self.train_images.astype("float32") / 255
        train_images_with_noise_channels = np.concatenate(
            [train_images, np.zeros((len(train_images), 784))], axis=1
        )
        train_images_with_zeros_channels = np.concatenate(
            [train_images, np.zeros((len(train_images), 784))], axis=1
        )
        # Now, let’s train the model from chapter 2 on both of these training sets
        # 126
        model = self.get_model()
        history_noise = model.fit(
            train_images_with_noise_channels, train_labels,
            epochs=10,
            batch_size=128,
            validation_split=0.2)
        model = self.get_model()
        history_zeros = model.fit(
            train_images_with_zeros_channels, train_labels,
            epochs=10,
            batch_size=128,
            validation_split=0.2)

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