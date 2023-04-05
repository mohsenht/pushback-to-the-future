from pandas import DataFrame, Timestamp
import pandas as pd
from datetime import timedelta

from constants import file_name_etd, flight_id
from loader.DataLoader import DataLoader
from path_generator import path_generator


class WeightedETDLoader(DataLoader):
    def __init__(self, airport):
        self.etd = pd.read_csv(
            path_generator(airport, file_name_etd),
            parse_dates=["departure_runway_estimated_time", "timestamp"],
        ).sort_values("timestamp")

    def load_data(self, now: Timestamp, data: DataFrame) -> DataFrame:
        now_etd = self.etd.loc[(self.etd.timestamp > now - timedelta(hours=30)) & (self.etd.timestamp <= now)]
        w_etd = pd.DataFrame(0, index=now_etd.index, columns=['w_etd'])
        w_etd['w_etd'] = (now_etd['departure_runway_estimated_time'] - now).dt.total_seconds()
        now_etd = pd.concat([now_etd, w_etd], axis=1)
        latest_now_etd = now_etd.groupby(flight_id)['w_etd'].ewm(span=2).mean()
        etd = data.merge(
            latest_now_etd, how="left", on=flight_id
        ).w_etd
        data["w_etd"] = etd / 60
        return data
