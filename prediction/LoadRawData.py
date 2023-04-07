import pandas as pd
from pandas import Timestamp

from constants import FILE_NAME_ETD, FILE_NAME_RUNWAYS, FILE_NAME_STANDTIMES, FILE_NAME_LAMP, FILE_NAME_CONFIG, \
    FILE_NAME_MFS, LAMP_COLUMN_FORECAST_TIMESTAMP
from model.Input import Input
from path_generator_utility import path_generator
from prediction.utility import crop_data_in_30h


class LoadRawData:

    def __init__(self, airport):
        self.etd = pd.read_csv(
            path_generator(airport, FILE_NAME_ETD),
            parse_dates=["departure_runway_estimated_time", "timestamp"],
        ).sort_values("timestamp")
        self.runways = pd.read_csv(
            path_generator(airport, FILE_NAME_RUNWAYS),
            parse_dates=["arrival_runway_actual_time", "timestamp"],
        ).sort_values("timestamp")
        self.standtimes = pd.read_csv(
            path_generator(airport, FILE_NAME_STANDTIMES),
            parse_dates=["arrival_stand_actual_time", "timestamp"],
        ).sort_values("timestamp")
        self.weather = pd.read_csv(
            path_generator(airport, FILE_NAME_LAMP),
            parse_dates=[LAMP_COLUMN_FORECAST_TIMESTAMP, "timestamp"],
        ).sort_values("timestamp")
        self.config = pd.read_csv(
            path_generator(airport, FILE_NAME_CONFIG),
            parse_dates=["timestamp"],
        ).sort_values("timestamp")
        self.mfs = pd.read_csv(path_generator(airport, FILE_NAME_MFS))

    def get_input(self, now: Timestamp):
        return Input(
            config=crop_data_in_30h(now, self.config),
            etd=crop_data_in_30h(now, self.etd),
            first_position=pd.DataFrame(),
            lamp=crop_data_in_30h(now, self.weather),
            mfs=self.mfs,
            runways=crop_data_in_30h(now, self.runways),
            standtimes=crop_data_in_30h(now, self.standtimes),
            tbfm=pd.DataFrame(),
            tfm=pd.DataFrame()
        )
