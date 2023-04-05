import pandas as pd
from datetime import timedelta
from pandas import Timestamp

from Input import Input
from constants import file_name_etd, file_name_runways, file_name_standtimes, file_name_lamp, file_name_config, \
    file_name_mfs
from path_generator import path_generator


class LoadRawData:

    def __init__(self, airport):
        self.etd = pd.read_csv(
            path_generator(airport, file_name_etd),
            parse_dates=["departure_runway_estimated_time", "timestamp"],
        ).sort_values("timestamp")
        self.runways = pd.read_csv(
            path_generator(airport, file_name_runways),
            parse_dates=["arrival_runway_actual_time", "timestamp"],
        ).sort_values("timestamp")
        self.standtimes = pd.read_csv(
            path_generator(airport, file_name_standtimes),
            parse_dates=["arrival_stand_actual_time", "timestamp"],
        ).sort_values("timestamp")
        self.weather = pd.read_csv(
            path_generator(airport, file_name_lamp),
            parse_dates=["forecast_timestamp", "timestamp"],
        ).sort_values("timestamp")
        self.config = pd.read_csv(
            path_generator(airport, file_name_config),
            parse_dates=["timestamp"],
        ).sort_values("timestamp")
        self.mfs = pd.read_csv(path_generator(airport, file_name_mfs))

    def get_input(self, now: Timestamp):
        config = self.config.loc[
            (self.config.timestamp > now - timedelta(hours=30)) & (self.config.timestamp <= now)]
        etd = self.etd.loc[(self.etd.timestamp > now - timedelta(hours=30)) & (self.etd.timestamp <= now)]
        first_position = pd.DataFrame()
        lamp = self.weather.loc[(self.weather.timestamp > now - timedelta(hours=30)) & (self.weather.timestamp <= now)]
        mfs = self.mfs
        runways = self.runways.loc[
            (self.runways.timestamp > now - timedelta(hours=30)) & (self.runways.timestamp <= now)]
        standtimes = self.standtimes.loc[
            (self.standtimes.timestamp > now - timedelta(hours=30)) & (self.standtimes.timestamp <= now)]
        tbfm = pd.DataFrame()
        tfm = pd.DataFrame()
        return Input(
            config,
            etd,
            first_position,
            lamp,
            mfs,
            runways,
            standtimes,
            tbfm,
            tfm
        )
