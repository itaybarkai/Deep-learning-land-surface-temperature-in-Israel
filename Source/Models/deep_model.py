import os
import numpy as np
import logging
from consts import DATASET_CONSTS, FILE_CONSTS
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
import mlflow
import mlflow.sklearn

class DeepModel:
    def __init__(self, print_summary=False):
        self.model = Sequential()
        self.model.add(Dense(1024, activation="relu", input_shape=(DATASET_CONSTS.FEATURES_COUNT, ) 
                             , kernel_initializer="he_normal"))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(512, activation="relu"
                             , kernel_initializer="he_normal"))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(256, activation="relu"
                             , kernel_initializer="he_normal"))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(128, activation="relu"
                             , kernel_initializer="he_normal"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        opt = Adam(learning_rate=0.001)
        
        self.model.compile(optimizer=opt, loss='mean_squared_error')
        
        self.epochs = 5
        self.batch_size = 128
        
        if print_summary:
            self.summary()

    def summary(self):
        self.model.summary()

    def fit(self, x_train, x_valid, y_train, y_valid):
        best_model_file = 'temp_best_model.x'
        callbacks = [ModelCheckpoint(filepath=best_model_file, save_best_only=True, monitor='val_loss', mode='min')]
        history = self.model.fit(x_train,
                                y_train,
                                epochs=self.epochs,
                                verbose=2,
                                shuffle=True,
                                batch_size=self.batch_size,
                                validation_data=(x_valid, y_valid),
                                use_multiprocessing = True,
                                callbacks=callbacks)
        self.model = load_model(best_model_file)
        # Log Config
        mlflow.log_artifact(best_model_file)
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
    
    def evaluate(self, samples, targets, verbose=2):
        return self.model.evaluate(samples, targets, batch_size=self.batch_size, verbose=verbose)
    
    def log_model(self, signature):
        mlflow.sklearn.log_model(self.model, "my model", signature=signature)