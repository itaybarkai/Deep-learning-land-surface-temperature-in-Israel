import os

class FILE_CONSTS:
    def LST_FORMAT_BY_YEAR_FILE(year):
        return os.getcwd() + f"\\..\\Data\\LST\\unprocessed\\LST_{year}_NetCDF.nc"
    def TOPOGRAPHY_FILE():
        return os.getcwd() + "\\..\\Data\\Topography\\Topography_Israel.tif"

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

class DATASET_CONSTS:
    FEATURES_COUNT = 5