#111
import matplotlib.pyplot as plt
from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
num_words=10000)

import numpy as np
from keras import layers
import keras

class Predictions:
    def __init__(self):

        x_train = self.vectorize_sequences(train_data)
        x_test = self.vectorize_sequences(test_data)

        y_train = self.to_one_hot(train_labels)
        y_test = self.to_one_hot(test_labels)

        x_val = x_train[:1000]
        partial_x_train = x_train[1000:]
        y_val = y_train[:1000]
        partial_y_train = y_train[1000:]

        """
        model = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(46, activation="softmax")
        ])
        
        model.compile(optimizer="rmsprop",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(x_train,
                  y_train,
                  epochs=9,
                  batch_size=512)
        """
        model = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(4, activation="relu"),
            layers.Dense(46, activation="sigmoid")
        ])
        model.compile(optimizer="rmsprop",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(partial_x_train,
                  partial_y_train,
                  epochs=20,
                  batch_size=128,
                  validation_data=(x_val, y_val))

        results = model.evaluate(x_test, y_test)

        predictions = model.predict(x_test)
        # Each entry in “predictions” is a vector of length 46:
        print(predictions[0].shape)
        # The coefficients in this vector sum to 1, as they form a probability distribution:
        print(np.sum(predictions[0]))
        # The largest entry is the predicted class—the class with the highest probability:
        print(np.argmax(predictions[0]))
        #112
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        model.compile(optimizer="rmsprop",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        """
        We mentioned earlier that because the final outputs are 46-dimensional,
         you should
        avoid intermediate layers with many fewer than 46 units. Now let’s
         see what happens
        when we introduce an information bottleneck by having intermediate
         layers that are
        significantly less than 46-dimensional: for example, 4-dimensional.
        """
        """"
        63/63 [==============================] - 0s 7ms/step - loss: 0.6609 - accuracy: 0.7975 - val_loss: 1.7030 - val_accuracy: 0.6790
        71/71 [==============================] - 0s 2ms/step - loss: 1.7333 - accuracy: 0.6692
        71/71 [==============================] - 0s 2ms/step
        (46,)
        2.4782863
        3
        """



    def vectorize_sequences(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            for j in sequence:
                results[i, j] = 1
        return results

    def to_one_hot(self, labels, dimension=46):
        results = np.zeros((len(labels), dimension))
        for i, label in enumerate(labels):
            results[i, label] = 1
        return results