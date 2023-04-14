from datetime import timedelta

import pandas as pd

from src.clean.TypeContainer import TypeContainer
from src.constants import FLIGHT_ID, TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME
from src.loader.FeatureExtractor import FeatureExtractor
from src.model.Input import Input


class BusyTFMExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: pd.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        latest = input_data.tfm.groupby(FLIGHT_ID).last()
        latest = latest.sort_values(TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME)
        results = data.apply(self.calculate_how_busy_airport_is, args=(latest, now), axis=1)
        return results

    def calculate_how_busy_airport_is(self, x, latest, now):
        last_etd = x["last_etd"]
        lower_bound = now + timedelta(minutes=(last_etd - 30))
        upper_bound = now + timedelta(minutes=(last_etd + 30))
        in_bound_data = latest.loc[latest[TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME] > lower_bound]
        in_bound_data = in_bound_data.loc[
            in_bound_data[TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME] < upper_bound]
        x['arrival_tbfm_business_1'] = len(in_bound_data)

        lower_bound = now + timedelta(minutes=(last_etd - 15))
        upper_bound = now + timedelta(minutes=(last_etd + 15))
        in_bound_data = in_bound_data.loc[in_bound_data[TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME] > lower_bound]
        in_bound_data = in_bound_data.loc[
            in_bound_data[TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME] < upper_bound]
        x['arrival_tbfm_business_2'] = len(in_bound_data)

        lower_bound = now + timedelta(minutes=(last_etd - 5))
        upper_bound = now + timedelta(minutes=(last_etd + 5))
        in_bound_data = in_bound_data.loc[in_bound_data[TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME] > lower_bound]
        in_bound_data = in_bound_data.loc[
            in_bound_data[TFM_COLUMN_ARRIVAL_RUNWAY_ESTIMATED_TIME] < upper_bound]
        x['arrival_tbfm_business_3'] = len(in_bound_data)

        return x
