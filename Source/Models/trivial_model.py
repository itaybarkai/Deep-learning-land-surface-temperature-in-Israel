import numpy as np
import logging

class TrivialModel:
    def predict(samples):
        logging.info(f"TrivialModel used to predict {samples.shape[0]} samples")
        return np.zeros(samples.shape[0]).reshape(-1, 1)