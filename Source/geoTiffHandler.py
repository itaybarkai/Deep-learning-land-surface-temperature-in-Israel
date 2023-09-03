import tifffile
import logging
from consts import FILE_CONSTS

class GeoTiffHandler:
    def __init__(self, filename=None):
        """
        Handles GeoTiff format files from google earth engine.
        """
        if filename is None:
            filename = FILE_CONSTS.TOPOGRAPHY_FILE()
        self.data = tifffile.imread(filename)

def get_raw_topography_data(filename=None):
    """
    Returns a numpy array of the topography data of "filename", defaults to FILE_CONSTS.TOPOGRAPHY_FILE
    """
    print(f"get_raw_topography_data DONE: filename={filename}")
    return GeoTiffHandler(filename).data
