import logging
import math
import numpy as np
from consts import GENERAL_CONSTS, DATASET_CONSTS
from netCDFHandler import get_raw_LST_data
from geoTiffHandler import get_raw_topography_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_day_dataset(year, test_lower_days=None):
    """
    Returns tuple of Samples (shape = samples x features) and Targets (shape = samples x 1)
    year: only read this year's LST data
    test_lower_days: only read this year's first X days (used for faster training as debug)
    """
    logging.info(f"get_day_dataset START: year={year}, lower_days={test_lower_days}")
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

                if (days[day] >= 365):
                    logging(f"get_day_dataset: got day {days[day]}, cyclic on 365")
                day_cos = math.cos(2 * math.pi * days[day] / 365)
                day_sin = math.sin(2 * math.pi * days[day] / 365)

                # consts.DATASET_CONSTS.FEATURES has list of these expected features
                single_column_features =    np.array([day_cos,
                                            day_sin,
                                            longs[lst_grid_middle_long],
                                            lats[lst_grid_middle_lat],
                                            lst_grid_avg,
                                            height_grid_avg])
                output_samples[cur_output_index] = np.concatenate((single_column_features, 
                                                                   height_grid_diffs.reshape(-1)))
                output_targets[cur_output_index] = np.array([lst_grid_middle])
                cur_output_index += 1
                
        logging.info(f"finished processing day: {day}")
    output_samples = output_samples[:cur_output_index]
    output_targets = output_targets[:cur_output_index].reshape(-1, 1)

    output_targets_diffs = output_targets - output_samples[:, DATASET_CONSTS.LST_AVG_INDEX].reshape(-1,1)
    logging.info(f"get_day_dataset DONE: year={year}, lower_days={test_lower_days}, with {cur_output_index} samples")
    return output_samples, output_targets_diffs


def preprocess_dataset(samples, config):
    """
    Does feature scaling based on the methods in the given config. Works in place.
    returns dict of feature index to sklearn scaler used on feature
    """
    scalers_dict = {}
    # daycos and daysin remain the same
    for feature_index in [DATASET_CONSTS.LONG_INDEX,
                          DATASET_CONSTS.LAT_INDEX,
                          DATASET_CONSTS.LST_AVG_INDEX,
                          DATASET_CONSTS.HEIGHT_AVG_INDEX]:
        _preprocess_single_feature(samples, scalers_dict, config, feature_index)
    _preprocess_multiple_features(samples, scalers_dict, config, DATASET_CONSTS.FIRST_HEIGHT_DIFF, DATASET_CONSTS.LAST_HEIGHT_DIFF)
    return scalers_dict


def _preprocess_single_feature(samples, scalers_dict, config, feature_index):
    """
    Perform a method of scaling based on the index method in given config. In Place.
    The performed scaler is kept in the given scalers_dict
    """
    scaler = _get_scaler(config, feature_index)

    scaler.fit(samples[:, feature_index].reshape(-1,1))
    scaler.transform(samples[:, feature_index].reshape(-1,1))

    scalers_dict[feature_index] = scaler


def _preprocess_multiple_features(samples, scalers_dict, config, first_feature_index, last_feature_index):
    """
    Perform a method of scaling based on the first index feature in the given config. In Place.
    Scaling is fitted and transformed over first to last features as they were a single feature.
    The performed scaler is kept in the given scalers_dict
    """
    scaler = _get_scaler(config, first_feature_index)
    for feature_index in range(first_feature_index, last_feature_index + 1):
        scaler.partial_fit(samples[:, feature_index].reshape(-1,1))

    for feature_index in range(first_feature_index, last_feature_index + 1):
        scaler.transform(samples[:, feature_index].reshape(-1,1))
    
    scalers_dict[first_feature_index] = scaler


def _get_scaler(config, feature_index):
    """Returns the wanted scaling method object based on the config of the given feature index"""
    scaling_method = config.scale_method_by_feature_index[feature_index]
    if scaling_method == "normalize":
        return MinMaxScaler(copy=False)
    elif scaling_method == "standardize":
        return StandardScaler(copy=False)
    raise ValueError(f"scaling method for index {feature_index} is unknown")


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
