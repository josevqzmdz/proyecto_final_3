from tensorflow import keras
import tensorflow as tf

class SimpleDense(keras.layers.Layer):
    #85
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units), initializer="random_normal")
        self.b = self.add_weight(shape=(self.units), initializer="zeros")

    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y

    """
    In SimpleDense, we no longer create weights in the constructor like 
    in the Naive-
    Dense example; instead, we create them in a dedicated state-creation
     method,
    build(), which receives as an argument the first input shape seen by
     the layer. The
    build() method is called automatically the first time the layer is
     called (via its
    __call__() method). In fact, that’s why we defined the computation 
    in a separate
    call() method rather than in the __call__() method directly. 
    The __call__() method
    of the base layer schematically looks like this:
    """
    # 86
    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)
            self.built = True
        return self.call(inputs)