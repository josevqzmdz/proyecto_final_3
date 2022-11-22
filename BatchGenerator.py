# 64
import math
import tensorflow as tf
from keras import optimizers
# Next, we need a way to iterate over the MNIST data in mini-batches. This is easy:

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels


    """
    The most difficult part of the process is the “training step”: updating the weights of
    the model after running it on one batch of data. We need to
    1) Compute the predictions of the model for the images in the batch.
    2) Compute the loss value for these predictions, given the actual labels
    3) Compute the gradient of the loss with regard to the model’s weights.
    4) Move the weights by a small amount in the direction opposite to the gradient
    """
    # 65
    def one_training_step(self, model, images_batch, labels_batch):
        with tf.GradientTape() as tape:
            predictions = model(images_batch)
            per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
                labels_batch, predictions
            )
            average_loss = tf.reduce_mean(per_sample_losses)
        """
        Compute the gradient of the loss with
        regard to the weights. The output gradients
        is a list where each entry corresponds to
        a weight from the model.weights list.
        """
        gradients = tape.gradient(average_loss, model.weights)
        self.update_weights(gradients, model.weights)
        return average_loss

    """
    As you already know, the purpose of the “weight update” step 
    (represented by the pre-
    ceding update_weights function) is to move the weights by “a bit” 
    in a direction that
    will reduce the loss on this batch. The magnitude of the move 
    is determined by the
    “learning rate,” typically a small quantity. The simplest way 
    to implement this
    update_weights function is to subtract gradient * learning_rate
     from each weight:
    """
    def update_weights(self, gradients, weights):
        learning_rate = 1e-3
        for g, w in zip(gradients, weights):
            # assign_sub is the
            # equivalent of -= for
            # TensorFlow variables.
            w.assign_sub(g * learning_rate)

    # In practice, you would almost never implement a weight update step like this by hand.
    # Instead, you would use an Optimizer instance from Keras, like this:

    def update_weights_keras(self, gradients, weights):
        optimizer = optimizers.SGD(learning_rate=1e-3)
        optimizer.apply_gradients(zip(gradients, weights))

    ## 66
    """
    An epoch of training simply consists of repeating the training step for 
    each batch in
    the training data, and the full training loop is simply the repetition of 
    one epoch:
    """
    def fit(self, model, images, labels, epochs, batch_size=128):
        for epoch_counter in range(epochs):
            print(f"epoch: {epoch_counter}")
        batch_generator = BatchGenerator(images, labels)
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()
            loss = self.one_training_step(model, images_batch, labels_batch)
            if batch_counter % 100 == 0:
                print(f"loss at batch: {batch_counter}: {loss:.2f}")