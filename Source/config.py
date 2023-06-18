import yaml
import os
from consts import FILE_CONSTS, DATASET_CONSTS

def load_config(config_name):
    with open(os.path.join(FILE_CONSTS.CONFIG_FOLDER(), config_name)) as file:
        config = yaml.safe_load(file)

    return config

class Configuration():
    def __init__(self, config_name):
        self.config = load_config(config_name)
        self.config_fs = self.config["feature_scaling"]

        self.scale_method_by_feature_index = {
            DATASET_CONSTS.DAY_COS_INDEX : self.config_fs["day_cos"],
            DATASET_CONSTS.DAY_SIN_INDEX : self.config_fs["day_sin"],
            DATASET_CONSTS.LONG_INDEX : self.config_fs["long"],
            DATASET_CONSTS.LAT_INDEX : self.config_fs["lat"],
            DATASET_CONSTS.LST_AVG_INDEX : self.config_fs["lst_avg"],
            DATASET_CONSTS.HEIGHT_AVG_INDEX : self.config_fs["height_avg"],
            DATASET_CONSTS.FIRST_HEIGHT_DIFF : self.config_fs["height_diffs"],
            }
        
if __name__ == "__main__":
    config = load_config("default.yaml")
