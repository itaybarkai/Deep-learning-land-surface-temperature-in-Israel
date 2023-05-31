import logging
import numpy as np
from consts import GENERAL_CONSTS, DATASET_CONSTS
from netCDFHandler import get_raw_data


def get_day_dataset(year, test_lower_days=None):
    logging.info(f"get_day_dataset START: year={year}, lower_days={test_lower_days}")
    longs, lats, days, raw_data = get_raw_data(year, test_lower_days)

    grid_size = GENERAL_CONSTS.GRID_SIZE
    grid_count_long = longs.shape[0] - grid_size + 1
    grid_count_lat = lats.shape[0] - grid_size + 1
    count_days = days.shape[0]

    output = np.empty((count_days * grid_count_long * grid_count_lat, DATASET_CONSTS.FEATURES_COUNT))
    cur_output_index = 0
    for day in range(count_days):
        for long in range(grid_count_long):
            for lat in range(grid_count_lat):
                grid = raw_data[day, lat:lat+grid_size, long:long+grid_size]
                if grid.mask.any():
                    # grid has invalid data, ignore.
                    continue

                grid_avg =  np.mean(grid)

                grid_middle_long = long + (grid_size // 2)
                grid_middle_lat = lat + (grid_size // 2)
                grid_middle = raw_data[day, grid_middle_lat, grid_middle_long]

                output[cur_output_index] = np.array([days[day],
                                                    longs[grid_middle_long],
                                                    lats[grid_middle_lat],
                                                    grid_avg,
                                                    grid_middle])
                cur_output_index += 1
                
        logging.info(f"finished processing day: {day}")
    # output was init in maximal size (to avoid )
    output = output[:cur_output_index]
    logging.info(f"get_day_dataset DONE: year={year}, lower_days={test_lower_days}")
    return output


def _test_validity_map():
    import matplotlib.pyplot as plt

    longs, lats, days, raw_data = get_raw_data(year=2020, test_lower_days=2)
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
