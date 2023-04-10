import pandas as pd

class Input:

    def __init__(self,
                 config: pd.DataFrame,
                 etd: pd.DataFrame,
                 first_position: pd.DataFrame,
                 lamp: pd.DataFrame,
                 mfs: pd.DataFrame,
                 runways: pd.DataFrame,
                 standtimes: pd.DataFrame,
                 tbfm: pd.DataFrame,
                 tfm: pd.DataFrame,
                 ):
        self.config = config
        self.etd = etd
        self.first_position = first_position
        self.lamp = lamp
        self.mfs = mfs
        self.runways = runways
        self.standtimes = standtimes
        self.tbfm = tbfm
        self.tfm = tfm
