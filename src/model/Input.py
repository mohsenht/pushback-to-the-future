import cudf

class Input:

    def __init__(self,
                 config: cudf.DataFrame,
                 etd: cudf.DataFrame,
                 first_position: cudf.DataFrame,
                 lamp: cudf.DataFrame,
                 mfs: cudf.DataFrame,
                 runways: cudf.DataFrame,
                 standtimes: cudf.DataFrame,
                 tbfm: cudf.DataFrame,
                 tfm: cudf.DataFrame,
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
