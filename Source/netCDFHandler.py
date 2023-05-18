import netCDF4 as nc
from consts import FILE_CONSTS, DATA_CONSTS

class NetCDFHandler:
    def __init__(self, filename=None, year=None, only_day=True):
        """
        Handles netCDF files.
        Must get filename of year. giving year will take this years lst file.
        only_day: if only day samples requires, make the output more ram efficient
        """
        if year:
            filename = FILE_CONSTS.LST_FORMAT_BY_YEAR_FILE.format(year=2020)

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
