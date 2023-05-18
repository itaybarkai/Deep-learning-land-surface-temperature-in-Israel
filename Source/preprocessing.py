import numpy as np

from consts import DATA_CONSTS, GENERAL_CONSTS
from netCDFHandler import NetCDFHandler

DATASET_FEATURES = 5

def get_day_dataset_per_year(year):
    n = NetCDFHandler(year=year, only_day=True)
    longs = n.get_variable(DATA_CONSTS.VARIABLES.LON)
    lats = n.get_variable(DATA_CONSTS.VARIABLES.LAT)
    days = n.get_variable(DATA_CONSTS.VARIABLES.DAY)
    raw_data = n.get_variable(DATA_CONSTS.VARIABLES.DATA)
    # shape = (days, longs, lats)
##########
    #day_mesh, long_mesh, lat_mesh = \
    #    np.meshgrid(days, longs, lats, indexing='ij')

    # stack the arrays together and reshape to a 2D array of tuples
    # (lat, long, day, lst) of shape (lat*long*day, 4)
    #data_array = np.stack((lat_mesh.ravel(), long_mesh.ravel(), 
    #                       day_mesh.ravel(), raw_data.ravel()), axis=1)
#########
    grid_size = GENERAL_CONSTS.GRID_SIZE
    grid_count_long = longs.shape[0] - grid_size + 1
    grid_count_lat = lats.shape[0] - grid_size + 1
    count_days = days.shape[0]
    
    output = np.empty((count_days * grid_count_long * grid_count_lat, DATASET_FEATURES))
    for day in range(count_days):
        for long in range(grid_count_long):
            for lat in range(grid_count_lat):
                grid_avg =  np.mean(raw_data[day, long:long+grid_size, lat:lat+grid_size])

                grid_middle_long = long + (grid_size // 2)
                grid_middle_lat = lat + (grid_size // 2)
                grid_middle = raw_data[day, grid_middle_long, grid_middle_lat]

                output_idx = (day * grid_count_long * grid_count_lat
                            + long * grid_count_lat + lat)
                
                output[output_idx] = [days[day],
                                      longs[grid_middle_long],
                                      lats[grid_middle_lat],
                                      grid_avg,
                                      grid_middle]
# untestes not finished func