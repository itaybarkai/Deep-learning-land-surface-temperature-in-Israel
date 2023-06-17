import logging
import netCDF4 as nc
from consts import FILE_CONSTS, DATA_CONSTS

class NetCDFHandler:
    def __init__(self, filename=None, year=None, only_day=True):
        """
        Handles netCDF files.
        Must get filename of year. giving year will take this years lst file.
        only_day: if raw data should return only the "day" band (changes shape)
        """
        if year:
            filename = FILE_CONSTS.LST_FORMAT_BY_YEAR_FILE(year=int(year))

        if filename:
            self.nc = nc.Dataset(filename, 'r')
        else:
            raise ValueError("must specify filename or year")
        
        self.only_day = True if only_day else False

    def get_variable(self, variable_name):
        variable_obj = self.nc.variables[variable_name]

        # lower ram usage by not returning other bands
        if self.only_day and variable_name == DATA_CONSTS.VARIABLES.DATA:
            return variable_obj[:, DATA_CONSTS.BAND.DAY_LST, :, :]
        return variable_obj[:]
    
    def get_dimensions(self):
        return self.nc.dimensions
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.nc.close()

    def close(self):
        self.nc.close()


def get_raw_LST_data(year, test_lower_days=None, only_day=True):
    """
    year: of LST file
    test_lower_days: int, how many days to slice. Use for fast debugging
    only_day: if raw data should return only the "day" band (changes shape)
    
    returns (longs, lats, days, raw_data)
    raw_data.shape = (days, lats, longs)
    """
    n = NetCDFHandler(year=year, only_day=only_day)
    longs = n.get_variable(DATA_CONSTS.VARIABLES.LON)
    lats = n.get_variable(DATA_CONSTS.VARIABLES.LAT)
    days = n.get_variable(DATA_CONSTS.VARIABLES.DAY)
    raw_data = n.get_variable(DATA_CONSTS.VARIABLES.DATA)
    # raw_data.shape = (days, lats, longs)
    n.close()

    if isinstance(test_lower_days, int):
        days = days[:test_lower_days]
        raw_data = raw_data[:test_lower_days, :, :]

    assert(longs.shape[0] == 409 and lats.shape[0] == 603)

    logging.info(f"get_raw_LST_data DONE: year={year}, lower_days={test_lower_days}, only_day={only_day}")
    return (longs, lats, days, raw_data)