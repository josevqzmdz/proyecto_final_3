import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
from keras import layers
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 29
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

class example1:

    def ejemplo1(self):
        train_image = train_images.reshape((60000, 28 * 28))
        train_image = train_images.astype("float32") / 255
        test_image = test_images.reshape((10000, 28 * 28))
        test_image = test_images.astype("float32") / 255

        print("ho")
        train_image.shape
        len(train_labels)
        print(train_labels)
        model.fit(train_image, train_labels, epochs=5, batch_size=128)

        # 30
        test_digits = test_image[0:10]
        predictions = model.predict(test_digits)
        print(predictions[0].argmax())

        test_loss, test_acc = model.evaluate(test_image, test_labels)
        print(f"test_acc: {test_acc}")

        """
        The test-set accuracy turns out to be 97.8%—that’s quite a bit lower 
        than the training-
        set accuracy (98.9%). This gap between training accuracy and test 
        accuracy is an
        example of overfitting: the fact that machine learning models 
        tend to perform worse
        on new data than on their training data. 
        """

        # 34
        my_slice = train_image[10:100]
        print(my_slice.shape)


    def example2(self):
        # 40
        X = np.random.random((32, 10))
        Y = np.random.random((10,))
        # First, we add an empty first axis to y, whose shape becomes (1, 10):
        yy = np.expand_dims(Y, axis=0)
        # Then, we repeat y 32 times alongside this new axis, so that we end up with a tensor Y
        # with shape (32, 10), where Y[i, :] == y for i in range(0, 32):
        Y = np.concatenate([yy] * 32, axis=0)
        # At this point, we can proceed to add X and Y, because they have the same shape.
        # In terms of implementation, no new rank-2 tensor is created, because that would
        # be terribly inefficient. The repetition operation is entirely virtual: it happens at the
        # algorithmic level rather than at the memory level. But thinking of the vector being
        # repeated 10 times alongside a new axis is a helpful mental model. Here’s what a naive
        # implementation would look like

        assert len(X.shape) == 2
        assert len(Y.shape) == 1
        assert X.shape[1] == Y.shape[0]
        X = X.copy()
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                X[i, j] += Y[j]
        return X

    def ejemplo3(self, x, y):
        # 42
        # You can also take the dot product between a matrix x and a vector y, which returns
        # a vector where the coefficients are the dot products between y and the rows of x. You
        # implement it as follows:
        assert len(x.shape) == 2
        assert len(y.shape) == 1
        assert x.shape[1] == y.shape[0]
        z = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i] += x[i, j] * y[j]
        return z

    def naive_vector_dot(x, y):
        assert len(x.shape) == 1

        assert len(y.shape) == 1
        assert x.shape[0] == y.shape[0]
        z = 0.
        for i in range(x.shape[0]):
            z += x[i] * y[i]
        return z


    # Note that as soon as one of the two tensors has an ndim greater than 1, dot is no lon-
    # ger symmetric, which is to say that dot(x, y) isn’t the same as dot(y, x).
    # Of course, a dot product generalizes to tensors with an arbitrary number of axes.
    # The most common applications may be the dot product between two matrices. You can
    # take the dot product of two matrices x and y (dot(x, y)) if and only if x.shape[1] ==
    # y.shape[0]. The result is a matrix with shape (x.shape[0], y.shape[1]), where the
    # coefficients are the vector products between the rows of x and the columns of y.
    # Here’s the naive implementation:
    
    def naive_matrix_dot(self, x, y):
        assert len(x.shape) == 2

        assert len(y.shape) == 2
        assert x.shape[1] == y.shape[0]
        z = np.zeros((x.shape[0], y.shape[1]))
        for i in range(x.shape[0]):
            for j in range(y.shape[1]):
                row_x = x[i, :]
                column_y = y[:, j]
                z[i, j] = self.naive_vector_dot(row_x, column_y)
        return z