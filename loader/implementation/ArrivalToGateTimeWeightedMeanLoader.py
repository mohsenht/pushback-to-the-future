from pandas import DataFrame, Timestamp
import pandas as pd
from datetime import timedelta

from constants import file_name_runways, file_name_standtimes
from loader.DataLoader import DataLoader
from path_generator import path_generator


class ArrivalToGateTimeWeightedMeanLoader(DataLoader):
    def __init__(self, airport):
        self.runways = pd.read_csv(
            path_generator(airport, file_name_runways),
            parse_dates=["arrival_runway_actual_time", "timestamp"],
        ).sort_values("timestamp")
        self.standtimes = pd.read_csv(
            path_generator(airport, file_name_standtimes),
            parse_dates=["arrival_stand_actual_time", "timestamp"],
        ).sort_values("timestamp")

    def load_data(self, now: Timestamp, data: DataFrame) -> DataFrame:
        now_runways = self.runways.loc[
            (self.runways.timestamp > now - timedelta(hours=30)) & (self.runways.timestamp <= now)]
        now_standtimes = self.standtimes.loc[
            (self.standtimes.timestamp > now - timedelta(hours=30)) & (self.standtimes.timestamp <= now)]
        merged_standtimes_runways = pd.merge(now_standtimes, now_runways, on='gufi')
        merged_standtimes_runways['diff_stand_runway'] = (
                (merged_standtimes_runways['arrival_stand_actual_time']
                 - merged_standtimes_runways['arrival_runway_actual_time']
                 ).dt.total_seconds() / 60)
        diff_stand_runway = merged_standtimes_runways[merged_standtimes_runways['diff_stand_runway'] >= 0]
        ewm = pd.Series(diff_stand_runway['diff_stand_runway']).ewm(span=2).mean().iloc[-1]
        data['mean_runway_pushback'] = ewm
        return data
