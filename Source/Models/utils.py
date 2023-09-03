import math
import random
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from consts import DATASET_CONSTS, GENERAL_CONSTS


def calculate_rmse(y_targets, y_predicted):
    MSE = mean_squared_error(y_targets, y_predicted)
    RMSE = math.sqrt(MSE)
    # logging.info(f"calculated RMSE: {RMSE}")
    return RMSE

    
def train_test_split_by_days(samples, targets, test_lower_days, test_size=0.3, seed=None):
    """
    Splits to train and test by splitting different days for each set,
    Rounding up the number of days into the test (eg, 0.3 of 5 days = 2 test, 3 train)
    """
    print(f"train_test_split_by_days: starting with test_size={test_size}")
    # get an array with len of number of samples, of the day of each sample
    samples_days = sample_dates_to_indices(samples)

    # create a permutation of available days
    num_days = test_lower_days
    booleans_days = np.zeros(shape=(num_days,), dtype=bool) # Array with "rows" False
    
    booleans_days[:math.ceil(test_size * num_days)] = True  # Set the first k% of the elements to True

    if seed:
        np.random.seed(seed)
    np.random.shuffle(booleans_days)

    # get a boolean array of train/test belonging for each row, by checking if the permutated day
    # is over the threshold determind by test_size
    samples_test_indices = [booleans_days[d] for d in samples_days]
    samples_train_indices = np.logical_not(samples_test_indices)

    return (samples[samples_train_indices], samples[samples_test_indices], 
            targets[samples_train_indices], targets[samples_test_indices])

def sample_dates_to_indices(samples):
    """
    Samples have a cos(day) and sin(day) data, this extracts the day from it.
    Returns a list with length==number of samples, each element (int) is the day_of_year of that row.
    """
    a_tan = np.arctan2(samples[:, DATASET_CONSTS.DAY_SIN_INDEX], samples[:,DATASET_CONSTS.DAY_COS_INDEX])
    a_tan_positives = (a_tan + 2 * np.pi) % (2 * np.pi)
    days = ((a_tan_positives * 365)/(2 * np.pi)).round().astype(int)
    return days
    
      
def train_test_split_by_space(samples, targets, test_size=0.3, seed=None):
    """
    Splits to train and test by splitting different location for each set,
    """
    print(f"train_test_split_by_space: starting with test_size={test_size}")
    num_places = GENERAL_CONSTS.LONG_UNITS * GENERAL_CONSTS.LAT_UNITS
    booleans_places = np.zeros(shape=(num_places,), dtype=bool) # Array with "rows" False
    booleans_places[:int(test_size * num_places)] = True  # Set the first k% of the elements to True
    if seed:
        np.random.seed(seed)
    np.random.shuffle(booleans_places)  # Shuffle the array
    booleans_places = booleans_places.reshape(GENERAL_CONSTS.LONG_UNITS, GENERAL_CONSTS.LAT_UNITS)

    samples_test_indices = [booleans_places[sample_long_lat_to_indices(long,lat)] for 
                            (long, lat) in samples[:, (DATASET_CONSTS.LONG_INDEX, DATASET_CONSTS.LAT_INDEX)]]
    samples_train_indices = np.logical_not(samples_test_indices)

    return (samples[samples_train_indices], samples[samples_test_indices], 
            targets[samples_train_indices], targets[samples_test_indices])

def sample_long_lat_to_indices(long, lat):
    """
    Samples have a "longitude" and "latitude data,
    this extracts the indices of the sample in the frame.
    """
    return (round((long - GENERAL_CONSTS.LONG_MINIMUM) / GENERAL_CONSTS.UNIT_LENGTH),
            round((lat - GENERAL_CONSTS.LAT_MINIMUM) / GENERAL_CONSTS.UNIT_LENGTH))
    