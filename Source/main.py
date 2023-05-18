import os
from netCDFHandler import NetCDFHandler
from consts import FILE_CONSTS

lst_handler_2020 = NetCDFHandler(FILE_CONSTS.LST_FORMAT_BY_YEAR_FILE.format(year=2020))
print(lst_handler_2020.get_dimensions())

lst_handler_2020.close()