import os
import logging
from preprocessing import get_day_dataset, preprocess_dataset
from netCDFHandler import NetCDFHandler
from consts import FILE_CONSTS, DATASET_CONSTS
from geoTiffHandler import get_raw_topography_data
from Models.trivial_model import TrivialModel
from Models.utils import calculate_rmse
from Models.deep_model import DeepModel

class Configuration():
    def __init__(self, filename=None):
        self.scale_method_by_feature_index = {DATASET_CONSTS.LONG_INDEX:"normalize",
                                                DATASET_CONSTS.LAT_INDEX:"normalize",
                                                DATASET_CONSTS.LST_AVG_INDEX:"normalize",
                                                DATASET_CONSTS.HEIGHT_AVG_INDEX:"standardize",
                                                DATASET_CONSTS.FIRST_HEIGHT_DIFF:"standardize",
                                                }


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    if os.path.basename(os.getcwd()) == "git":
        os.chdir("Source")

    samples, targets = get_day_dataset(year=2020, test_lower_days=20)
    print ("samples shape is: ", samples.shape)
    print ("targets shape is: ", targets.shape)

    config = Configuration()
    preprocess_dataset(samples, config)

    trivial_predictions = TrivialModel.predict(samples)
    print("trivial loss is: ", calculate_rmse(trivial_predictions, targets))

    deep_model = DeepModel()
    deep_model.summary()
    history = deep_model.fit(samples, targets)
    deep_model.plot_history(history)
