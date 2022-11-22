import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from example1 import *
from NaiveDense import *
from NaiveSequential import *

from keras import layers

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
In the preceding code, negative_samples and positive_samples are 
both arrays
with shape (1000, 2). Let’s stack them into a single array with shape
 (2000, 2).
"""
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
"""
Let’s generate the corresponding target labels, an array of zeros and 
ones of shape
(2000, 1), where targets[i, 0] is 0 if inputs[i] belongs to class 0 
(and inversely).
"""
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))

# now lets plot
#plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
#plt.show()

# 81
"""
Now let’s create a linear classifier that can learn to separate these 
two blobs. A linear
classifier is an affine transformation (prediction = W • input + b) 
trained to minimize
the square of the difference between predictions and the targets.
As you’ll see, it’s actually a much simpler example than the end-to-end
 example of
a toy two-layer neural network you saw at the end of chapter 2. However, 
this time you
should be able to understand everything about the code, line by line.
Let’s create our variables, W and b, initialized with random values and 
with zeros,
respectively.
"""
input_dim = 2
output_dim = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim)))

class Training_baby_steps:

    # 82
    def square_loss(self, targets, predictions):
        per_sample_losses = tf.square(targets - predictions)
        # per_sample_losses will be a tensor with the same shape as
        # targets and predictions, containing per-sample loss scores.
        return tf.reduce_mean(per_sample_losses)

    # 83
    def training_step(self, inputs, targets):
        learning_rate = 0.1
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            loss = self.square_loss(predictions, targets)
        grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])
        W.assign_sub(grad_loss_wrt_W * learning_rate)
        b.assign_sub(grad_loss_wrt_b * learning_rate)
        #82
        for step in range(40):
            a = Training_baby_steps()
            loss = a.training_step(inputs, targets)
            print(f"loss at step: {step}: {loss:4f}")

        return loss
