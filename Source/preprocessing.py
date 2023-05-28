import numpy as np

from consts import DATA_CONSTS, GENERAL_CONSTS
from netCDFHandler import NetCDFHandler

DATASET_FEATURES = 5

def get_day_dataset_per_year(year, test_lower_days=None):
    n = NetCDFHandler(year=year, only_day=True)
    longs = n.get_variable(DATA_CONSTS.VARIABLES.LON)
    lats = n.get_variable(DATA_CONSTS.VARIABLES.LAT)
    days = n.get_variable(DATA_CONSTS.VARIABLES.DAY)
    raw_data = n.get_variable(DATA_CONSTS.VARIABLES.DATA)
    # shape = (days, lats, longs)
    if isinstance(test_lower_days, int):
        days = days[:test_lower_days]
        raw_data = raw_data[:test_lower_days, :, :]

    grid_size = GENERAL_CONSTS.GRID_SIZE
    grid_count_long = longs.shape[0] - grid_size + 1
    grid_count_lat = lats.shape[0] - grid_size + 1
    count_days = days.shape[0]
    print(raw_data[0, :, 0:0+9, 0:0+9])
    print("long:", grid_count_long, " lat:", grid_count_lat, " days:", count_days)

    output = np.empty((count_days * grid_count_long * grid_count_lat, DATASET_FEATURES))
    for day in range(count_days):
        for long in range(grid_count_long):
            for lat in range(grid_count_lat):
                print(raw_data[day, lat:lat+grid_size, long:long+grid_size])
                grid_avg =  np.mean(raw_data[day, lat:lat+grid_size, long:long+grid_size])
                print(grid_avg)

                grid_middle_long = long + (grid_size // 2)
                grid_middle_lat = lat + (grid_size // 2)
                
                grid_middle = raw_data[day, grid_middle_lat, grid_middle_long]
                print(grid_avg)

                output_idx = (day * grid_count_long * grid_count_lat
                            + long * grid_count_lat + lat)
                
                output[output_idx] = [days[day],
                                      longs[grid_middle_long],
                                      lats[grid_middle_lat],
                                      grid_avg,
                                      grid_middle]
        print("finished processing day: ", day)
    return output
# untestes not finished func