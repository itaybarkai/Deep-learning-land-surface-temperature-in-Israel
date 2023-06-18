import os
import logging
from preprocessing import get_day_dataset, preprocess_dataset
from netCDFHandler import NetCDFHandler
from consts import FILE_CONSTS, DATASET_CONSTS
from geoTiffHandler import get_raw_topography_data
from Models.trivial_model import TrivialModel
from Models.utils import calculate_rmse
from Models.deep_model import DeepModel
from config import Configuration
import mlflow




if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # Run from Source folder or Git folder
    if os.path.basename(os.getcwd()) == "git":
        os.chdir("Source")

    samples, targets = get_day_dataset(year=2020, test_lower_days=1)
    print ("samples shape is: ", samples.shape)
    print ("targets shape is: ", targets.shape)

    config = Configuration("default.yaml")
    preprocess_dataset(samples, config)

    trivial_predictions = TrivialModel.predict(samples)
    trivial_rmse = calculate_rmse(trivial_predictions, targets)
    print("trivial loss is: ", trivial_rmse)

    with mlflow.start_run():
        deep_model = DeepModel()
        deep_model.summary()
        history = deep_model.fit(samples, targets)
        deep_model.plot_history(history, trivial_rmse)
