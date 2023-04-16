import pandas as pd
from pandas import Timestamp

from src.constants import FILE_NAME_ETD, FILE_NAME_RUNWAYS, FILE_NAME_STANDTIMES, FILE_NAME_LAMP, FILE_NAME_CONFIG, \
    FILE_NAME_MFS, LAMP_COLUMN_FORECAST_TIMESTAMP, STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME, \
    RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME, COLUMN_NAME_TIMESTAMP, \
    RUNWAYS_COLUMN_TIMESTAMP, STANDTIMES_COLUMN_TIMESTAMP, LAMP_COLUMN_TIMESTAMP, CONFIG_COLUMN_TIMESTAMP, \
    FILE_NAME_TFM, FILE_NAME_TBFM, FILE_NAME_FIRST_POSITION, FILE_NAME_LABELS
from src.model.Input import Input
from src.path_generator_utility import path_generator
from src.prediction.InBoundDataLoader import InBoundDataLoader
from src.prediction.utility import crop_data_in_30h


class LoadRawData:

    def __init__(
            self,
            in_bound_data_loader: InBoundDataLoader,
            start_time: pd.Timestamp,
            end_time: pd.Timestamp
    ):
        self.data = in_bound_data_loader.get_data(start_time, end_time, FILE_NAME_LABELS)
        self.config = in_bound_data_loader.get_data(start_time, end_time, FILE_NAME_CONFIG)
        self.etd = in_bound_data_loader.get_data(start_time, end_time, FILE_NAME_ETD)
        self.first_position = in_bound_data_loader.get_data(start_time, end_time, FILE_NAME_FIRST_POSITION)
        self.lamp = in_bound_data_loader.get_data(start_time, end_time, FILE_NAME_LAMP)
        self.mfs = in_bound_data_loader.get_data(start_time, end_time, FILE_NAME_MFS)
        self.runways = in_bound_data_loader.get_data(start_time, end_time, FILE_NAME_RUNWAYS)
        self.standtimes = in_bound_data_loader.get_data(start_time, end_time, FILE_NAME_STANDTIMES)
        self.tbfm = in_bound_data_loader.get_data(start_time, end_time, FILE_NAME_TBFM)
        self.tfm = in_bound_data_loader.get_data(start_time, end_time, FILE_NAME_TFM)

    def get_input(self, now: Timestamp):
        return Input(
            config=crop_data_in_30h(now, self.config),
            etd=crop_data_in_30h(now, self.etd),
            first_position=self.first_position,
            lamp=crop_data_in_30h(now, self.lamp),
            mfs=self.mfs,
            runways=crop_data_in_30h(now, self.runways),
            standtimes=crop_data_in_30h(now, self.standtimes),
            tbfm=crop_data_in_30h(now, self.tbfm),
            tfm=crop_data_in_30h(now, self.tfm),
        )

    def get_data(self) -> pd.DataFrame:
        return self.data

    def print(self):
        print(f"config: start_time: {self.config.iloc[0][COLUMN_NAME_TIMESTAMP]} end_time: {self.config.iloc[-1][COLUMN_NAME_TIMESTAMP]}")
        print(f"etd: start_time: {self.etd.iloc[0][COLUMN_NAME_TIMESTAMP]} end_time: {self.etd.iloc[-1][COLUMN_NAME_TIMESTAMP]}")
        print(f"lamp: start_time: {self.lamp.iloc[0][COLUMN_NAME_TIMESTAMP]} end_time: {self.lamp.iloc[-1][COLUMN_NAME_TIMESTAMP]}")
        print(f"runways: start_time: {self.runways.iloc[0][COLUMN_NAME_TIMESTAMP]} end_time: {self.runways.iloc[-1][COLUMN_NAME_TIMESTAMP]}")
        print(f"standtimes: start_time: {self.standtimes.iloc[0][COLUMN_NAME_TIMESTAMP]} end_time: {self.standtimes.iloc[-1][COLUMN_NAME_TIMESTAMP]}")
        print(f"tbfm: start_time: {self.tbfm.iloc[0][COLUMN_NAME_TIMESTAMP]} end_time: {self.tbfm.iloc[-1][COLUMN_NAME_TIMESTAMP]}")
        print(f"tfm: start_time: {self.tfm.iloc[0][COLUMN_NAME_TIMESTAMP]} end_time: {self.tfm.iloc[-1][COLUMN_NAME_TIMESTAMP]}")
