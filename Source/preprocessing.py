import os
import logging
import math
import numpy as np
from consts import GENERAL_CONSTS, DATASET_CONSTS, FILE_CONSTS
from netCDFHandler import get_raw_LST_data
from geoTiffHandler import get_raw_topography_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import Configuration


def get_day_dataset(year, test_lower_days=None):
    """
    Returns tuple of Samples (shape = samples x features) and Targets (shape = samples x 1)
    year: only read this year's LST data
    test_lower_days: only read this year's first X days (used for faster training as debug)
    """
    print(f"get_day_dataset START: year={year}, lower_days={test_lower_days}")
    longs, lats, days, lst_raw_data = get_raw_LST_data(year, test_lower_days)
    height_data = get_raw_topography_data()

    grid_size = GENERAL_CONSTS.GRID_SIZE
    grid_count_long = longs.shape[0] - grid_size + 1
    grid_count_lat = lats.shape[0] - grid_size + 1
    count_days = days.shape[0]

    # init output in maximal size (to avoid appending which is many allocations)
    output_samples = np.empty((count_days * grid_count_long * grid_count_lat, DATASET_CONSTS.FEATURES_COUNT))
    output_targets = np.empty((count_days * grid_count_long * grid_count_lat, 1))
    cur_output_index = 0

    for day in range(count_days):
        for long in range(grid_count_long):
            for lat in range(grid_count_lat):
                lst_grid = lst_raw_data[day, lat:lat+grid_size, long:long+grid_size]
                if lst_grid.mask.any():
                    # lst_grid has invalid data, ignore.
                    continue

                lst_grid_avg =  np.mean(lst_grid)

                lst_grid_middle_long = long + (grid_size // 2)
                lst_grid_middle_lat = lat + (grid_size // 2)
                lst_grid_middle = lst_raw_data[day, lst_grid_middle_lat, lst_grid_middle_long]

                height_grid = height_data[lat:lat+grid_size, long:long+grid_size]
                height_grid_avg =  np.mean(height_grid)
                height_grid_diffs = height_grid - height_grid_avg

                day_cos = math.cos(2 * math.pi * days[day] / 365)
                day_sin = math.sin(2 * math.pi * days[day] / 365)

                # consts.DATASET_CONSTS.FEATURES has list of these expected features
                single_column_features = np.array([day_cos,
                                            day_sin,
                                            longs[lst_grid_middle_long],
                                            lats[lst_grid_middle_lat],
                                            lst_grid_avg,
                                            height_grid_avg])
                output_samples[cur_output_index] = np.concatenate((single_column_features, 
                                                                   height_grid_diffs.reshape(-1)))
                output_targets[cur_output_index] = np.array([lst_grid_middle])
                cur_output_index += 1
                
        print(f"finished processing day: {day}")
    output_samples = output_samples[:cur_output_index]
    output_targets = output_targets[:cur_output_index].reshape(-1, 1)

    output_targets_diffs = output_targets - output_samples[:, DATASET_CONSTS.LST_AVG_INDEX].reshape(-1,1)
    print(f"get_day_dataset DONE: year={year}, lower_days={test_lower_days}, with {cur_output_index} samples")
    return output_samples, output_targets_diffs


def get_cached_day_dataset(year, test_lower_days=None):
    if os.path.exists(FILE_CONSTS.PROCESSED_BY_YEAR_FILE(year, test_lower_days)):
        samples, targets = load_processed_data(year=year, days=test_lower_days)
        print("Loaded processed data")
    else:
        samples, targets = get_day_dataset(year=year, test_lower_days=test_lower_days)
        save_processed_data(samples, targets, year=year, days=test_lower_days)
        print("Created and Saved processed data")

    return samples, targets


def save_processed_data(samples, targets, year, days):
    filename = FILE_CONSTS.PROCESSED_BY_YEAR_FILE(year, days)
    np.savez(filename, samples=samples, targets=targets)

def load_processed_data(year, days):
    filename = FILE_CONSTS.PROCESSED_BY_YEAR_FILE(year, days)
    npz = np.load(filename)
    samples = npz["samples"]
    targets = npz["targets"]
    return samples, targets


class Preproccesser:
    def __init__(self, config):
        if isinstance(config, str):
            config = Configuration(config)
        self.config = config
        self.scalers_dict = {}

    def preprocess_dataset(self, samples, train=True):
        """
        Does feature scaling based on the methods in the given config. Works in place.
        returns dict of feature index to sklearn scaler used on feature
        """
        # daycos and daysin remain the same

        print(f"preprocess_dataset: starting train") if train else \
            print(f"preprocess_dataset: starting valid") 
        for feature_index in [DATASET_CONSTS.LONG_INDEX,
                            DATASET_CONSTS.LAT_INDEX,
                            DATASET_CONSTS.LST_AVG_INDEX,
                            DATASET_CONSTS.HEIGHT_AVG_INDEX]:
            self._preprocess_single_feature(samples, feature_index, train)
        self._preprocess_multiple_features(samples, DATASET_CONSTS.FIRST_HEIGHT_DIFF, DATASET_CONSTS.LAST_HEIGHT_DIFF, train)


    def _preprocess_single_feature(self, samples, feature_index, train=True):
        """
        Perform a method of scaling based on the index method in given config. In Place.
        The performed scaler is kept in the given scalers_dict
        """
        scaler = self._get_scaler(feature_index) if train else self.scalers_dict[feature_index]

        if scaler is not None:
            if train:
                scaler.fit(samples[:, feature_index].reshape(-1,1))
            scaler.transform(samples[:, feature_index].reshape(-1,1))

        self.scalers_dict[feature_index] = scaler


    def _preprocess_multiple_features(self, samples, first_feature_index, last_feature_index, train=True):
        """
        Perform a method of scaling based on the first index feature in the given config. In Place.
        Scaling is fitted and transformed over first to last features as they were a single feature.
        The performed scaler is kept in the given scalers_dict
        """
        scaler = self._get_scaler(first_feature_index) if train else self.scalers_dict[first_feature_index]
        if scaler is not None:
            if train:
                for feature_index in range(first_feature_index, last_feature_index + 1):
                    scaler.partial_fit(samples[:, feature_index].reshape(-1,1))

            for feature_index in range(first_feature_index, last_feature_index + 1):
                scaler.transform(samples[:, feature_index].reshape(-1,1))
        
        self.scalers_dict[first_feature_index] = scaler


    def _get_scaler(self, feature_index):
        """Returns the wanted scaling method object based on the config of the given feature index"""
        scaling_method = self.config.scale_method_by_feature_index[feature_index]
        if scaling_method == "normalize":
            return MinMaxScaler(copy=False)
        elif scaling_method == "standardize":
            return StandardScaler(copy=False)
        elif scaling_method == "raw":
            return None
        raise ValueError(f"scaling method for index {feature_index} is unknown")

