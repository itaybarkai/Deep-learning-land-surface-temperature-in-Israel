import os

class FILE_CONSTS:
    def PROCESSED_BY_YEAR_FILE(year, days):
        return f"./Data/Processed/DATA_{year}_{days}_days_processed.npz"
    def LST_FORMAT_BY_YEAR_FILE(year):
        return f"./Data/LST/unprocessed/LST_{year}_NetCDF.nc"
    def TOPOGRAPHY_FILE():
        return "./Data/Topography/Topography_Israel.tif"
    def CONFIG_FOLDER():
        return "./Configs/"


class DATA_CONSTS:
    class VARIABLES:
        LAT = "y"
        LON = "x"
        BAND = "band"
        DAY = "time"
        DATA = "__xarray_dataarray_variable__"


    class BAND:
        NIGHT_LST = 0
        DAY_LST = 1
        DAILY_LST = 2
        QA = 3

    class QA:
        NO_DATA_BOTH_DAY_NIGHT = 0
        NO_DATA_DAY = 1
        NO_DATA_NIGHT = 2
        BOTH_VALID = 3

class GENERAL_CONSTS:
    GRID_SIZE = 9
    LONG_UNITS = 409
    LAT_UNITS = 603
    UNIT_LENGTH = 0.00832402
    LONG_MINIMUM = 33.20034536095077 
    LAT_MINIMUM = 28.996716325654262 


class DATASET_CONSTS:
    FEATURES = ["day_cos", "day_sin", "long", "lat", "lst_avg", "height_avg"] + \
                ["h_" + str(i) for i in range(1, GENERAL_CONSTS.GRID_SIZE * GENERAL_CONSTS.GRID_SIZE + 1)]
    FEATURES_COUNT = len(FEATURES)
    DAY_COS_INDEX = 0
    DAY_SIN_INDEX = 1
    LONG_INDEX = 2
    LAT_INDEX = 3
    LST_AVG_INDEX = 4
    HEIGHT_AVG_INDEX = 5
    FIRST_HEIGHT_DIFF = 6
    LAST_HEIGHT_DIFF = 86