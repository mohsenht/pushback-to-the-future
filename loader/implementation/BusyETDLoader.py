from datetime import timedelta

import pandas as pd

from Input import Input
from clean.extract.TypeContainer import TypeContainer
from constants import flight_id
from loader.DataLoader import DataLoader


class BusyETDLoader(DataLoader):

    def load_data(self, now: pd.Timestamp, data: pd.DataFrame, input: Input, type_container: TypeContainer) -> pd.DataFrame:
        latest_now_etd = input.etd.groupby(flight_id).last()
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
