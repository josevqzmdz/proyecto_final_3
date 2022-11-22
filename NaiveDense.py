#63
import tensorflow as tf

"""
Let’s implement a simple Python class, NaiveDense, that creates two TensorFlow
variables, W and b, and exposes a __call__() method that applies the preceding
transformation.
"""

class NaiveDense:
    # 63
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        w_shape = (input_size, output_size)

        # Create a matrix,
        # W, of shape
        # (input_size,
        # output_size),
        # initialized with
        # random values.

        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.w = tf.Variable(w_initial_value)

        b_shape = output_size
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)

    @property
    def weights(self):
        return [self.w, self.b]