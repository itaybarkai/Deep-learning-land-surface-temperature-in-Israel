import numpy as np
import logging
from consts import DATASET_CONSTS
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError

class DeepModel:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(400, activation="relu", input_shape=(DATASET_CONSTS.FEATURES_COUNT, )))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(200, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(100, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(50, activation="relu"))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(1))

        self.model.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                            loss='mean_squared_error')
        #, metrics =[RootMeanSquaredError(name='rmse')]
        
    def summary(self):
        self.model.summary()

    def fit(self, samples, targets, batch_size=32):
        history = self.model.fit(samples,
                                targets,
                                epochs=5,
                                verbose=1,
                                batch_size=batch_size,
                                validation_split = 0.4)
        return history

    def plot_history(self, history):
        fig, ax = plt.subplots()
        ax.plot(np.sqrt(history.history["loss"]),'r-x', label="Train RMSE")
        ax.plot(np.sqrt(history.history["val_loss"]),'b-x', label="Validation RMSE")
        ax.legend()
        ax.set_title('root_mean_squared_error loss')
        ax.grid(True)
        plt.show()


    def predict(self, samples):
        return self.model.predict(samples)