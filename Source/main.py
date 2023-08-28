import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

from preprocessing import get_cached_day_dataset, Preproccesser
from netCDFHandler import NetCDFHandler
from consts import FILE_CONSTS, DATASET_CONSTS
from geoTiffHandler import get_raw_topography_data
from Models.trivial_model import TrivialModel
from Models.utils import calculate_rmse, train_test_split_by_days, train_test_split_by_space
from Models.deep_model import DeepModel
from config import Configuration
import mlflow
from mlflow.models import infer_signature
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np; np.set_printoptions(suppress=True)

# Run from base folder
if os.path.basename(os.getcwd()) == "Source":
    os.chdir("..")

def eval_trivial_model(x_train, x_valid, y_train, y_valid):
    # Trivial Model
    trivial_train_predictions = TrivialModel.predict(x_train)
    trivial_valid_predictions = TrivialModel.predict(x_valid)
    trivial_train_rmse = calculate_rmse(trivial_train_predictions, y_train)
    trivial_valid_rmse = calculate_rmse(trivial_valid_predictions, y_valid)
    print("Trivial Train MSE loss is: ", trivial_train_rmse ** 2)
    print("Trivial Valid MSE loss is: ", trivial_valid_rmse ** 2)
    mlflow.log_metric("Trivial Train RMSE", trivial_train_rmse)
    mlflow.log_metric("Trivial Valid RMSE", trivial_valid_rmse)
    mlflow.log_metric("Trivial Train MSE", trivial_train_rmse ** 2)
    mlflow.log_metric("Trivial Valid MSE", trivial_valid_rmse ** 2)

    return trivial_train_rmse, trivial_valid_rmse


if __name__ == "__main__":
    mlflow.tensorflow.autolog(silent=True)
    with mlflow.start_run() as run:
        print("\n")
        config_file = "default.yaml"
        config = Configuration("default.yaml")
        # Get Processed Data:
        year = 2020
        test_lower_days = 10
        print("Days: ", test_lower_days, "  Year: ", year)
        samples, targets = get_cached_day_dataset(year=year, test_lower_days=test_lower_days)

        print ("samples shape is: ", samples.shape)
        print ("targets shape is: ", targets.shape)
            
        if config.split_days and not config.split_space:
            x_train, x_valid, y_train, y_valid = train_test_split_by_days(samples, targets, test_size = 0.3,
                                                                        test_lower_days=test_lower_days,
                                                                        seed=42)
        elif config.split_space:
            x_train, x_valid, y_train, y_valid = train_test_split_by_space(samples, targets, test_size = 0.3,
                                                                        seed=42)
        else:
            x_train, x_valid, y_train, y_valid = train_test_split(samples, targets, test_size = 0.3,
                                                                random_state=42)

        # Free some ram, after the split creates new arrays
        del samples
        del targets

        print ("x_train shape is: ", x_train.shape)
        print ("y_train shape is: ", y_train.shape)
        print ("x_valid shape is: ", x_valid.shape)
        print ("y_valid shape is: ", y_valid.shape)

        # Feature Scaling
        preprocessor = Preproccesser(config)
        preprocessor.preprocess_dataset(x_train, train=True)
        preprocessor.preprocess_dataset(x_valid, train=False)

        print ("\n----  Starting Training ----\n")

        # Deep Model
        # Calculate baseline
        trivial_train_rmse, trivial_valid_rmse = eval_trivial_model(x_train, x_valid, y_train, y_valid)

        # Log Config
        mlflow.log_artifact(os.path.join(FILE_CONSTS.CONFIG_FOLDER(), config_file))

        # Train Model
        deep_model = DeepModel(print_summary=True)
        history = deep_model.fit(x_train, x_valid, y_train, y_valid)

        print("\n")

        # Train Evaluates
        train_mse = deep_model.evaluate(x_train, y_train, verbose=0)
        print("Final Train MSE: ", train_mse)
        mlflow.log_metric("Final Train MSE", train_mse)

        # Valid Evaluates
        valid_mse = deep_model.evaluate(x_valid, y_valid, verbose=0)
        print("Final Valid MSE: ", valid_mse)
        mlflow.log_metric("Final Valid MSE", valid_mse)
        
        # Improvements from baseline
        print("MSE Improvement : ", (valid_mse - trivial_valid_rmse**2))
        mlflow.log_metric("MSE Improvement", valid_mse - trivial_valid_rmse**2)
        print("MSE Improvement Percent : ", 100 *(valid_mse - trivial_valid_rmse**2) / trivial_valid_rmse**2, "%")
        mlflow.log_metric("MSE Improvement Percent", 100 *(valid_mse - trivial_valid_rmse**2) / trivial_valid_rmse**2)

        print(f"Done, Run ID: {run.info.run_id}\n\n")


    







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