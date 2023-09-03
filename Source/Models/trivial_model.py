import numpy as np
import logging

class TrivialModel:
    def predict(samples):
        return np.zeros(samples.shape[0]).reshape(-1, 1)