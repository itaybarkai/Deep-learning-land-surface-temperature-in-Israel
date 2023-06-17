import math
import logging
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_targets, y_predicted):
    MSE = mean_squared_error(y_targets, y_predicted)
    RMSE = math.sqrt(MSE)
    logging.info(f"calculated RMSE: {RMSE}")
    return RMSE