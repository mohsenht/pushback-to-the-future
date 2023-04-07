from datetime import timedelta

import pandas as pd

from model.Input import Input
from clean.TypeContainer import TypeContainer
from constants import FLIGHT_ID
from loader.FeatureExtractor import FeatureExtractor


class BusyETDExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: pd.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        latest_now_etd = input_data.etd.groupby(FLIGHT_ID).last()
        latest_now_etd = latest_now_etd.sort_values("departure_runway_estimated_time")
        results = data.apply(self.calculate_how_busy_is_departure, args=(latest_now_etd, now), axis=1)
        return results

    def calculate_how_busy_is_departure(self, x, latest_now_etd, now):
        last_etd = x["last_etd"]
        lower_bound = now + timedelta(minutes=(last_etd - 5))
        upper_bound = now + timedelta(minutes=(last_etd + 5))
        more_than_lower_bound_data = latest_now_etd.loc[latest_now_etd.departure_runway_estimated_time > lower_bound]
        in_bound_data = more_than_lower_bound_data.loc[more_than_lower_bound_data.departure_runway_estimated_time < upper_bound]
        x['departure_business'] = in_bound_data.shape[0]
        return x
