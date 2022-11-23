#114
from keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = (
boston_housing.load_data())

import numpy as np
from keras import layers
import keras

class BostonHouses:
    def __init__(self):
        #114
        mean = self.train_data.mean(axis=0)
        self.train_data -= mean
        std = self.train_data.std(axis=0)
        self.train_data /= std
        self.test_data -= mean
        self.test_data /= std

        """
        Because so few samples are available, we’ll use a very small model 
        with two intermedi-
        ate layers, each with 64 units. In general, the less training data 
        you have, the worse
        overfitting will be, and using a small model is one way to mitigate
         overfitting.
        """
        # 116
        # k fold validation

        k = 4
        num_val_samples = len(self.train_data) // k
        num_epochs = 100
        all_scores = []
        for i in range(k):
            print(f"processing fold #{i}")
            val_data = self.train_data[i * num_val_samples: (i + 1) * num_val_samples]
            val_targets = self.train_targets[i * num_val_samples: (i + 1) * num_val_samples]
            partial_train_data = np.concatenate(
                [self.train_data[:i * num_val_samples],
                 self.train_data[(i + 1) * num_val_samples:]],
                axis=0
            )
            partial_train_targets = np.concatenate(
                [train_targets[:i * num_val_samples],
                 train_targets[(i + 1) * num_val_samples:]],
                axis=0
            )
            model = self.build_model()
            model.fit(partial_train_data, partial_train_targets,
                      epochs=num_epochs, batch_size=16, verbose=0
                      )
            val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
            all_scores.append(val_mae)

    def build_model(self):
        """
        The model ends with a single unit and no activation (it will be a linear
         layer). This is a
        typical setup for scalar regression (a regression where you’re trying
        to predict a single
        continuous value). Applying an activation function would constrain
        the range the out-
        put can take; for instance, if you applied a sigmoid activation
        function to the last layer,
        the model could only learn to predict values between 0 and 1.
        Here, because the last
        layer is purely linear, the model is free to learn to predict
         values in any range.
        Note that we compile the model with the mse loss function—mean
         squared error, the
        square of the difference between the predictions and the targets.
         This is a widely used
        loss function for regression problems.
        We’re also monitoring a new metric during training: mean absolute
        error (MAE). It’s the
        absolute value of the difference between the predictions and the
         targets. For instance, an
        MAE of 0.5 on this problem would mean your predictions are off by
         $500 on average.
        :return:
        """
        model = keras.Sequential([
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1)
        ])
        model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
        return model