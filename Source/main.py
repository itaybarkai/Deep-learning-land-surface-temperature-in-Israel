import os
from preprocessing import get_day_dataset, Preproccesser
from netCDFHandler import NetCDFHandler
from consts import FILE_CONSTS, DATASET_CONSTS
from geoTiffHandler import get_raw_topography_data
from Models.trivial_model import TrivialModel
from Models.utils import calculate_rmse
from Models.deep_model import DeepModel
from config import Configuration
import mlflow
from sklearn.model_selection import train_test_split
import numpy as np; np.set_printoptions(suppress=True)

# Run from base folder
if os.path.basename(os.getcwd()) == "Source":
    os.chdir("..")

import logging
logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":

    samples, targets = get_day_dataset(year=2020, test_lower_days=1)
    print ("samples shape is: ", samples.shape)
    print ("targets shape is: ", targets.shape)
        
    x_train, x_valid, y_train, y_valid = train_test_split(samples, targets, test_size = 0.3, random_state=42)

    # Normalization
    preprocessor = Preproccesser("default.yaml")
    preprocessor.preprocess_dataset(x_train, train=True)
    preprocessor.preprocess_dataset(x_valid, train=False)

    import ipdb; ipdb.set_trace()
    # Trivial Model
    trivial_train_predictions = TrivialModel.predict(x_train)
    trivial_valid_predictions = TrivialModel.predict(x_valid)
    trivial_train_rmse = calculate_rmse(trivial_train_predictions, y_train)
    trivial_valid_rmse = calculate_rmse(trivial_valid_predictions, y_valid)
    print("Trivial Train MSE loss is: ", trivial_train_rmse ** 2)
    print("Trivial Valid MSE loss is: ", trivial_valid_rmse ** 2)

    # Deep Model
    with mlflow.start_run():
        deep_model = DeepModel()
        deep_model.summary()
        history = deep_model.fit(x_train, x_valid, y_train, y_valid)
        deep_model.plot_history(history, trivial_valid_rmse)


    







"""

def _test_validity_map():
    "only for debug - shows a plot of valid/invalid lst data points"
    import matplotlib.pyplot as plt

    longs, lats, days, raw_data = get_raw_LST_data(year=2020, test_lower_days=2)
    # raw_data.shape = (days, lats, longs)

    day = 0
    raw_data_0 = raw_data[day, :, :]

    fig, (ax0, ax1) = plt.subplots(1,2)
    ax0.imshow(raw_data_0.mask, cmap=plt.cm.binary, interpolation='nearest')
    ax0.set_title("day 0")
    ax0.set_xlabel("longs")
    ax0.set_ylabel("lats")


    day = 1
    raw_data_1 = raw_data[day, :, :]

    ax1.imshow(raw_data_1.mask, cmap=plt.cm.binary, interpolation='nearest')
    ax1.set_title("day 1")
    ax1.set_xlabel("longs")
    ax1.set_ylabel("lats")
    plt.show()

"""