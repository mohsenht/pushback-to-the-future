from pandas import DataFrame, Timestamp
import pandas as pd
from datetime import timedelta

from constants import file_name_etd
from loader.DataLoader import DataLoader
from path_generator import path_generator


class LastTwoETDLoader(DataLoader):
    def __init__(self, airport):
        self.etd = pd.read_csv(
            path_generator(airport, file_name_etd),
            parse_dates=["departure_runway_estimated_time", "timestamp"],
        ).sort_values("timestamp")

    def load_data(self, now: Timestamp, data: DataFrame) -> DataFrame:
        now_etd = self.etd.loc[(self.etd.timestamp > now - timedelta(hours=30)) & (self.etd.timestamp <= now)]
        now_etd = now_etd[now_etd['gufi'].isin(data['gufi'])]
        now_etd['diff_etd'] = (now_etd['departure_runway_estimated_time'] - now).dt.total_seconds()
        latest_now_etd = now_etd.groupby("gufi").apply(self.last_two_diff).reset_index(drop=True)
        etd = data.merge(
            latest_now_etd, how="left", on="gufi"
        ).diff_two_etd
        data["diff_two_etd"] = etd / 60
        return data

    def last_two_diff(self, x):
        if len(x) < 2:
            last = x.iloc[-1]
            last['diff_two_etd'] = last["diff_etd"] - last["diff_etd"]
            return last
        last = x.iloc[-1]
        second_last = x.iloc[-2]

        last['diff_two_etd'] = last["diff_etd"] - second_last["diff_etd"]

        return last
