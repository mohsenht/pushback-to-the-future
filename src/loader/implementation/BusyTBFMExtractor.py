from datetime import timedelta

import pandas as pd

from src.clean.TypeContainer import TypeContainer
from src.constants import FLIGHT_ID, TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME
from src.loader.FeatureExtractor import FeatureExtractor
from src.model.Input import Input


class BusyTBFMExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: pd.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        latest = input_data.tbfm.groupby(FLIGHT_ID).last()
        latest = latest.sort_values(TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME)
        results = data.apply(self.calculate_how_busy_airport_is, args=(latest, now), axis=1)
        return results

    def calculate_how_busy_airport_is(self, x, latest, now):
        last_etd = x["last_etd"]
        lower_bound = now + timedelta(minutes=(last_etd - 30))
        upper_bound = now + timedelta(minutes=(last_etd + 30))
        in_bound_data = latest.loc[latest[TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME] > lower_bound]
        in_bound_data = in_bound_data.loc[
            in_bound_data[TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME] < upper_bound]
        x['arrival_tfm_business_1'] = len(in_bound_data)

        lower_bound = now + timedelta(minutes=(last_etd - 15))
        upper_bound = now + timedelta(minutes=(last_etd + 15))
        in_bound_data = in_bound_data.loc[in_bound_data[TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME] > lower_bound]
        in_bound_data = in_bound_data.loc[
            in_bound_data[TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME] < upper_bound]
        x['arrival_tfm_business_2'] = len(in_bound_data)

        lower_bound = now + timedelta(minutes=(last_etd - 5))
        upper_bound = now + timedelta(minutes=(last_etd + 5))
        in_bound_data = in_bound_data.loc[in_bound_data[TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME] > lower_bound]
        in_bound_data = in_bound_data.loc[
            in_bound_data[TBFM_COLUMN_SCHEDULED_RUNWAY_ESTIMATED_TIME] < upper_bound]
        x['arrival_tfm_business_3'] = len(in_bound_data)

        return x
