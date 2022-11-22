import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from Training_baby_steps import *
from example1 import *
from NaiveDense import *
from NaiveSequential import *
from SimpleDense import *
import numpy as np
from keras import layers

from BatchGenerator import *

"""
Now you understand that this model consists of a chain of two Dense layers,
 that each
layer applies a few simple tensor operations to the input data, 
and that these opera-
tions involve weight tensors. Weight tensors, which are attributes of the layers, are
where the knowledge of the model persists.
"""

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 29
"""
Now you understand that sparse_categorical_crossentropy is the loss function
that’s used as a feedback signal for learning the weight tensors, and which 
the train-
ing phase will attempt to minimize. You also know that this reduction 
of the loss
happens via mini-batch stochastic gradient descent. The exact rules governing
 a spe-
cific use of gradient descent are defined by the rmsprop optimizer passed as 
the first
argument.
"""
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

class example1:

    def ejemplo1(self):

        """
        Now you understand that the input images are stored in NumPy tensors,
         which are
        here formatted as float32 tensors of shape (60000, 784)
        (training data) and (10000,
        784) (test data) respectively.
        :return:
        """
        train_image = train_images.reshape((60000, 28 * 28))
        train_image = train_images.astype("float32") / 255
        test_image = test_images.reshape((10000, 28 * 28))
        test_image = test_images.astype("float32") / 255

        print("ho")
        train_image.shape
        len(train_labels)
        print(train_labels)
        """
        Now you understand what happens when you call fit: the model will 
        start to iterate
        on the training data in mini-batches of 128 samples, 5 times over
         (each iteration over
        all the training data is called an epoch). For each batch, 
        the model will compute the
        gradient of the loss with regard to the weights (using the 
        Backpropagation algorithm,
        which derives from the chain rule in calculus) and move the
         weights in the direction
        that will reduce the value of the loss for this batch
        
        After these 5 epochs, the model will have performed 2,345 gradient 
        updates (469
        per epoch), and the loss of the model will be sufficiently low
        that the model will be
        capable of classifying handwritten digits with high accuracy.
        """
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

    def gradientTape1(self):
        x = tf.Variable(0)
        with tf.GradientTape() as tape:
            y = 2 * x + 3
        grad_of_y_wrt_x = tape.gradient(y, x)

    def codigoBasura(self):
        # 28 francois chollet
        # e = example1()
        # print(e.example2())
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
        print(f"accuracy: {matches.mean: .2f}")
        """
        # 76
        """
        x = tf.ones(shape=(2, 1))
        print(x)
        y = tf.Tensor(
            [[0.]
             [0]], shape=(2, 1), dtype="float32"
        )
        print(y)
        """
        # 77
        """
        x = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
        print(x)
        tf.Tensor(
            [[-0.14208166]
             [-0.95319825]
             [1.1096532]], shape=(3, 1), dtype="float32")
        x = tf.random.uniform(shape=(3, 1), minval=0., maxval=1.)
        print(x)
        """
        # v = tf.Variable(initial_value=tf.random.normal(shape=(3,1)))
        # print(v)

        # 78
        """
        input_var = tf.Variable(initial_value=3)
        with tf.GradientTape() as tape:
            result = tf.square(input_var)
        gradient = tape.gradient(result, input_var)
        """

        # 79
        """
        It’s actually possible for these inputs to be any arbitrary tensor.
        However, only trainable variables are tracked by default. With a constant tensor, you’d
        have to manually mark it as being tracked by calling tape.watch() on it.
        """
        # input_const = tf.constant(3)
        # with tf.GradientTape() as tape:
        #    tape.watch(input_const)
        #    result = tf.square(input_const)
        # gradient = tape.gradient(result, input_const)
        """
        Why is this necessary? Because it would be too expensive to preemptively 
        store the
        information required to compute the gradient of anything with respect 
        to anything.
        To avoid wasting resources, the tape needs to know what to watch. 
        Trainable variables
        are watched by default because computing the gradient of a loss
         with regard to a list of
        trainable variables is the most common use of the gradient tape.
        The gradient tape is a powerful utility, even capable of computing 
        second-order gra-
        dients, that is to say, the gradient of a gradient. For instance, 
        the gradient of the posi-
        tion of an object with regard to time is the speed of that object,
        and the second-order
        gradient is its acceleration.
        """
        # time = tf.Variable(0)
        # with tf.GradientTape() as outer_tape:
        #    with tf.GradientTape() as inner_tape:
        #        position = 4.9 * time ** 2
        #    speed = inner_tape.gradient(position, time)
        # acceleration = outer_tape.gradient(speed, time)
        """
        We use the outer tape to
        compute the gradient of
        the gradient from the inner
        tape. Naturally, the answer
        is 4.9 * 2 = 9.8.
        """
        """
        After 40 steps, the training loss seems to have stabilized around 
        0.025. Let’s plot how
        our linear model classifies the training data points.
         Because our targets are zeros and
        ones, a given input point will be classified as “0” 
        if its prediction value is below 0.5, and
        as “1” if it is above 0.5 (see figure 3.7):
        """

        # 80
        """
        First, let’s come up with some nicely linearly separable synthetic 
        data to work with:
        two classes of points in a 2D plane. We’ll generate each class 
        of points by drawing their
        coordinates from a random distribution with a specific covariance
         matrix and a spe-
        cific mean. Intuitively, the covariance matrix describes the 
        shape of the point cloud,
        and the mean describes its position in the plane (see figure 3.6).
         We’ll reuse the same
        covariance matrix for both point clouds, but we’ll use two 
        different mean values—the
        point clouds will have the same shape, but different positions.
        """
        """
        num_samples_per_class = 1000
        negative_samples = np.random.multivariate_normal(
            mean=[0, 3],
            cov=[[1, 0.5], [0.5, 1]],
            size=num_samples_per_class
        )
        positive_samples = np.random.multivariate_normal(
            mean=[3, 0],
            cov=[[1, 0.5], [0.5, 1]],
            size=num_samples_per_class
        )
        """
        """
        In the preceding code, negative_samples and positive_samples are 
        both arrays
        with shape (1000, 2). Let’s stack them into a single array with shape
         (2000, 2).
        """
        # inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
        """
        Let’s generate the corresponding target labels, an array of zeros and 
        ones of shape
        (2000, 1), where targets[i, 0] is 0 if inputs[i] belongs to class 0 
        (and inversely).
        """
        # targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
        #                     np.ones((num_samples_per_class, 1), dtype="float32")))
        # predictions = model(inputs)
        # plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
        # plt.show()

        # x = np.linspace(-1, 4, 100)
        # y = - W[0] / W[1] * x + (0.5 - b) / W[1]
        # plt.plot(x, y, "-r")
        # plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
        # plt.show()

        # 86
        # Once instantiated, a layer like this can be used just like a function, taking as input
        # a TensorFlow tensor:
        """
        my_dense = SimpleDense(units=32, activation=tf.nn.relu)
        input_tensor = tf.ones(shape=(2, 784))
        output_tensor = my_dense(input_tensor)
        print(output_tensor.shape)

        """
        # 89
        model = keras.Sequential([keras.layers.Dense(1)])  # Define a linear classifier.
        model.compile(optimizer="rmsprop",  # Specify the optimizer by name: RMSprop (it’s case-insensitive)
                      loss="mean_squared_error",  # Specify the loss by name: mean squared error.
                      metrics=["accuracy"]  # specify accuracy
                      )
        """
        In the preceding call to compile(), we passed the optimizer, loss, 
        and metrics as
        strings (such as "rmsprop"). These strings are actually shortcuts 
        that get converted to
        Python objects. For instance, "rmsprop" becomes keras.optimizers.RMSprop().
        Importantly, it’s also possible to specify these arguments as 
        object instances, like this:
        """

        model.compile(optimizer=keras.optimizers.RMSprop(),
                      loss=keras.losses.MeanSquaredError(),
                      metrics=[keras.metrics.BinaryAccuracy()]
                      )
        """
        This is useful if you want to pass your own custom losses or metrics, 
        or if you want to
        further configure the objects you’re using—for instance, by passing a 
        learning_rate
        argument to the optimizer:
        """

        model.compile(
            optimizer=keras.optimizers.RMSprop(learning_rate=1e-4),
            loss=0.1,
            metrics=[0.1, 0.01]
        )
        # en 0.1 deberia venir la variable my_custom_loss y my_custom_metric
        # pero en el libro es pseudocode

        # 91
        """
        The goal of machine learning is not to obtain models that perform well 
        on the train-
        ing data, which is easy—all you have to do is follow the gradient. 
        The goal is to obtain
        models that perform well in general, and particularly on data points
         that the model
        has never encountered before. Just because a model performs well on
         its training data
        doesn’t mean it will perform well on data it has never seen! For 
        instance, it’s possible
        that your model could end up merely memorizing a mapping between
         your training
        samples and their targets, which would be useless for the task of
        predicting targets for
        data the model has never seen before. 

        To keep an eye on how the model does on new data, it’s standard 
        practice to
        reserve a subset of the training data as validation data: you won’t
         be training the model
        on this data, but you will use it to compute a loss value and metrics
         value. You do this
        by using the validation_data argument in fit(). Like the training data,
         the valida-
        tion data could be passed as NumPy arrays or as a TensorFlow Dataset object.
        """
        # 92
        model = keras.Sequential([keras.layers.Dense(1)])
        model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.1),
                      loss=keras.losses.MeanSquaredError(),
                      metrics=[keras.metrics.BinaryAccuracy()]
                      )
        """
        To avoid having samples
        from only one class in
        the validation data,
        shuffle the inputs and
        targets using a random
        indices permutation
        """
        indices_permutation = np.random.permutation(len(inputs))
        shuffled_inputs = inputs[indices_permutation]
        shuffled_targets = targets[indices_permutation]

        """
        Reserve 30% of the
        training inputs and
        targets for validation
        (we’ll exclude these
        samples from training
        and reserve them to
        compute the validation
        loss and metrics).
        """

        num_validation_samples = int(0.3 * len(inputs))
        val_inputs = shuffled_inputs[:num_validation_samples]
        val_targets = shuffled_targets[:num_validation_samples]
        training_inputs = shuffled_inputs[num_validation_samples:]
        training_targets = shuffled_targets[num_validation_samples:]

        """
        Validation data, used only
        to monitor the validation
        loss and metrics
        """

        model.fit(
            training_inputs,
            training_targets,
            epochs=5,
            batch_size=16,
            validation_data=(val_inputs, val_targets)
        )