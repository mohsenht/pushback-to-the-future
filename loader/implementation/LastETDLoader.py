from pandas import DataFrame, Timestamp
import pandas as pd
from datetime import timedelta

from loader.DataLoader import DataLoader


class LastETDLoader(DataLoader):
    def __init__(self, file_path):
        self.etd = pd.read_csv(
            file_path,
            parse_dates=["departure_runway_estimated_time", "timestamp"],
        ).sort_values("timestamp")

    def load_data(self, now: Timestamp, data: DataFrame) -> DataFrame:
        now_etd = self.etd.loc[(self.etd.timestamp > now - timedelta(hours=30)) & (self.etd.timestamp <= now)]
        latest_now_etd = now_etd.groupby("gufi").last().departure_runway_estimated_time
        departure_runway_estimated_time = data.merge(
            latest_now_etd, how="left", on="gufi"
        ).departure_runway_estimated_time
        data["last_etd"] = (departure_runway_estimated_time - now).dt.total_seconds() / 60
        data["last_etd"] = data.last_etd.clip(lower=0).astype(int)
        return data
