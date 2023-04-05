from pandas import DataFrame, Timestamp
import pandas as pd
from datetime import timedelta

from constants import file_name_etd, flight_id
from loader.DataLoader import DataLoader
from path_generator import path_generator


class BusyETDLoader(DataLoader):
    def __init__(self, airport):
        self.etd = pd.read_csv(
            path_generator(airport, file_name_etd),
            parse_dates=["departure_runway_estimated_time", "timestamp"],
        ).sort_values("timestamp")

    def load_data(self, now: Timestamp, data: DataFrame) -> DataFrame:
        now_etd = self.etd.loc[(self.etd.timestamp > now - timedelta(hours=30)) & (self.etd.timestamp <= now)]
        latest_now_etd = now_etd.groupby(flight_id).last()
        latest_now_etd = latest_now_etd.sort_values("departure_runway_estimated_time")
        results = data.apply(self.calculate_how_busy_is_departure, args=(latest_now_etd, now), axis=1)
        return results

    def calculate_how_busy_is_departure(self, x, latest_now_etd, now):
        last_etd = x["last_etd"]
        lower_bound = now + timedelta(minutes=(last_etd - 5))
        upper_bound = now + timedelta(minutes=(last_etd + 5))
        asghar = latest_now_etd.loc[latest_now_etd.departure_runway_estimated_time > lower_bound]
        asghar = asghar.loc[asghar.departure_runway_estimated_time < upper_bound]
        x['departure_business'] = asghar.shape[0]
        return x
