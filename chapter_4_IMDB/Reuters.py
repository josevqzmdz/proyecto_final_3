
# Like IMDB and MNIST, the Reuters dataset comes packaged as part of Keras. Let’s
# take a look.
import matplotlib.pyplot as plt
from keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
num_words=10000)

import numpy as np
from keras import layers
import keras
from chapter_4_IMDB.IMDB_examples import IMDB_examples as IMDB

class Reuters:
    
    def __init__(self):

        x_train = self.vectorize_sequences(train_data)
        x_test = self.vectorize_sequences(test_data)

        y_train = self.to_one_hot(train_labels)
        y_test = self.to_one_hot(test_labels)

        #108
        """
        This topic-classification problem looks similar to the previous 
        movie-review classifica-
        tion problem: in both cases, we’re trying to classify short snippets
         of text. But there is
        a new constraint here: the number of output classes has gone from 2
         to 46. The
        dimensionality of the output space is much larger.
        In a stack of Dense layers like those we’ve been using, each layer
         can only access
        information present in the output of the previous layer. If one 
        layer drops some
        information relevant to the classification problem, this information
         can never be
        recovered by later layers: each layer can potentially become an 
        information bottle-
        neck. In the previous example, we used 16-dimensional intermediate 
        layers, but a
        16-dimensional space may be too limited to learn to separate 46 
        different classes:
        such small layers may act as information bottlenecks, permanently
         dropping rele-
        vant information.
        For this reason we’ll use larger layers. Let’s go with 64 units.
        """

        model = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(46, activation="softmax")
        ])
        # 109
        model.compile(optimizer="rmsprop",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

        x_val = x_train[:1000]
        partial_x_train = x_train[1000:]
        y_val = y_train[:1000]
        partial_y_train = y_train[1000:]

        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=20,
                            batch_size=512,
                            validation_data=(x_val, y_val))

        # Plotting the training and validation loss
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        # Plotting the training and validation accuracy
        plt.clf()
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        plt.plot(epochs, acc, "bo", label="Training accuracy")
        plt.plot(epochs, val_acc, "b", label="Validation accuracy")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

    """
    To vectorize the labels, there are two possibilities: you can cast the 
    label list as an inte-
    ger tensor, or you can use one-hot encoding. One-hot encoding is a 
    widely used format
    for categorical data, also called categorical encoding. In this case, 
    one-hot encoding of
    the labels consists of embedding each label as an all-zero vector 
    with a 1 in the place of
    the label index. The following listing shows an example.
    """
    def to_one_hot(self, labels, dimension=46):
        results = np.zeros((len(labels), dimension))
        for i, label in enumerate(labels):
            results[i, label] = 1
        return results

    def vectorize_sequences(self, sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            for j in sequence:
                results[i, j] = 1
        return results