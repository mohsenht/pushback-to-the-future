from pandas import DataFrame, Timestamp
import pandas as pd
from datetime import timedelta

from constants import flight_id
from loader.DataLoader import DataLoader


class RunwayETDMeanLoader(DataLoader):
    def __init__(self, file_path_runway, file_path_etd):
        self.runways = pd.read_csv(
            file_path_runway,
            parse_dates=["departure_runway_actual_time", "timestamp"],
        ).sort_values("timestamp")
        self.etd = pd.read_csv(
            file_path_etd,
            parse_dates=["departure_runway_estimated_time", "timestamp"],
        ).sort_values("timestamp")

    def load_data(self, now: Timestamp, data: DataFrame) -> DataFrame:
        now_runways = self.runways.loc[
            (self.runways.timestamp > now - timedelta(hours=30))
            & (self.runways.timestamp <= now)] \
            .dropna(subset=['departure_runway_actual_time'])
        now_etd = self.etd.loc[(self.etd.timestamp > now - timedelta(hours=30)) & (self.etd.timestamp <= now)]
        now_etd = now_etd[now_etd[flight_id].isin(now_runways[flight_id])]
        grouped_now_etd = now_etd.groupby(flight_id)
        results = grouped_now_etd.apply(self.find_etd_for_each_gufi, arg1=now_runways)

        df = pd.DataFrame(results.to_list(), columns=[flight_id, 'etd_new'])
        merged_df = now_runways.merge(df[[flight_id, 'etd_new']], on=flight_id, how='left')

        data['etd_new'] = merged_df.dropna(subset=['etd_new'])['etd_new'].mean()

        return data

    def find_etd_for_each_gufi(self, group, arg1):
        runway = arg1.loc[group.name == arg1.gufi]
        if runway.empty:
            return
        runway_unique = runway.iloc[0]
        etd = group.loc[runway_unique['departure_runway_actual_time'] > group.timestamp]
        if etd.empty:
            return
        return runway_unique[flight_id], runway_unique['departure_runway_actual_time'] - etd.iloc[-1]['departure_runway_estimated_time']
