# Deep Learning with Python by François Chollet

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
    #76
    """
    x = tf.ones(shape=(2, 1))
    print(x)
    y = tf.Tensor(
        [[0.]
         [0]], shape=(2, 1), dtype="float32"
    )
    print(y)
    """
    #77
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
    #v = tf.Variable(initial_value=tf.random.normal(shape=(3,1)))
    #print(v)

    #78
    """
    input_var = tf.Variable(initial_value=3)
    with tf.GradientTape() as tape:
        result = tf.square(input_var)
    gradient = tape.gradient(result, input_var)
    """

    #79
    """
    It’s actually possible for these inputs to be any arbitrary tensor.
    However, only trainable variables are tracked by default. With a constant tensor, you’d
    have to manually mark it as being tracked by calling tape.watch() on it.
    """
    #input_const = tf.constant(3)
    #with tf.GradientTape() as tape:
    #    tape.watch(input_const)
    #    result = tf.square(input_const)
    #gradient = tape.gradient(result, input_const)
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
    #time = tf.Variable(0)
    #with tf.GradientTape() as outer_tape:
    #    with tf.GradientTape() as inner_tape:
    #        position = 4.9 * time ** 2
    #    speed = inner_tape.gradient(position, time)
    #acceleration = outer_tape.gradient(speed, time)
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
    #inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
    """
    Let’s generate the corresponding target labels, an array of zeros and 
    ones of shape
    (2000, 1), where targets[i, 0] is 0 if inputs[i] belongs to class 0 
    (and inversely).
    """
    #targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
    #                     np.ones((num_samples_per_class, 1), dtype="float32")))
    #predictions = model(inputs)
    #plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
    #plt.show()

    #x = np.linspace(-1, 4, 100)
    #y = - W[0] / W[1] * x + (0.5 - b) / W[1]
    #plt.plot(x, y, "-r")
    #plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
    #plt.show()

    #86
    # Once instantiated, a layer like this can be used just like a function, taking as input
    # a TensorFlow tensor:
    """
    my_dense = SimpleDense(units=32, activation=tf.nn.relu)
    input_tensor = tf.ones(shape=(2, 784))
    output_tensor = my_dense(input_tensor)
    print(output_tensor.shape)
    
    """
    #89
    model = keras.Sequential([keras.layers.Dense(1)]) #Define a linear classifier.
    model.compile(optimizer="rmsprop", #Specify the optimizer by name: RMSprop (it’s case-insensitive)
                  loss="mean_squared_error", # Specify the loss by name: mean squared error.
                  metrics=["accuracy"] # specify accuracy
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