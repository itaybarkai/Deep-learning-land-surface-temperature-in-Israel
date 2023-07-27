import numpy as np
import logging
from consts import DATASET_CONSTS
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.metrics import RootMeanSquaredError
import mlflow

class DeepModel:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(1024, activation="relu", input_shape=(DATASET_CONSTS.FEATURES_COUNT, )))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(1))

        # Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9)
        self.model.compile(optimizer=RMSprop(learning_rate=lr_schedule),
                            loss='mean_squared_error')  #, metrics =[RootMeanSquaredError(name='rmse')]
        self.epochs = 5
        self.batch_size = 32
        
    def summary(self):
        self.model.summary()

    def fit(self, x_train, x_valid, y_train, y_valid):
        history = self.model.fit(x_train,
                                y_train,
                                epochs=self.epochs,
                                verbose=1,
                                batch_size=self.batch_size,
                                validation_data=(x_valid, y_valid))
        return history

    def plot_history(self, history, trivial_rmse=None):
        fig, ax = plt.subplots()
        ax.plot(np.sqrt(history.history["loss"]),'r-x', label="Train RMSE")
        ax.plot(np.sqrt(history.history["val_loss"]),'b-x', label="Validation RMSE")
        if trivial_rmse:
            ax.plot([trivial_rmse for i in range(self.epochs)], '--', label="Trivial RMSE")
        ax.legend()
        ax.set_title('root_mean_squared_error loss')
        ax.grid(True)
        plt.show()


    def predict(self, samples):
        return self.model.predict(samples)