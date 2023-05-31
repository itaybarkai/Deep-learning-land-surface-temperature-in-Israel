import os
import logging
from preprocessing import get_day_dataset
from netCDFHandler import NetCDFHandler
from consts import FILE_CONSTS

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    if os.path.basename(os.getcwd()) == "git":
        os.chdir("Source")

    output = get_day_dataset(year=2020, test_lower_days=2)
    print (output.shape)