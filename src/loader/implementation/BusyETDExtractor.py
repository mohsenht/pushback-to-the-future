from datetime import timedelta

import cudf
import pandas as pd
import numpy as np

from src.model.Input import Input
from src.clean.TypeContainer import TypeContainer
from src.constants import FLIGHT_ID, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME
from src.loader.FeatureExtractor import FeatureExtractor


class BusyETDExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: cudf.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> cudf.DataFrame:
        latest_now_etd = input_data.etd.groupby(FLIGHT_ID).last()
        latest_now_etd = latest_now_etd.sort_values(ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME)
        latest_now_etd = latest_now_etd.reset_index()
        departure_runway_estimated_time = latest_now_etd.departure_runway_estimated_time.astype(
            'int64') / 1000000000
        epoch_now = now.timestamp()
        pandas_data = data.to_pandas()
        results = pandas_data.apply(
            func=self.calculate_how_busy_is_departure,
            axis=1,
            args=(departure_runway_estimated_time, epoch_now)
        )
        return cudf.DataFrame.from_pandas(results)

    def calculate_how_busy_is_departure(self, row, departure_runway_estimated_time, now):
        last_etd = row.last_etd
        lower_bound = now + (last_etd - 5) * 60
        upper_bound = now + (last_etd + 5) * 60
        more_than_lower_bound_data = departure_runway_estimated_time.loc[
            departure_runway_estimated_time > lower_bound]
        in_bound_data = more_than_lower_bound_data.loc[
            more_than_lower_bound_data < upper_bound]
        row['departure_business'] = in_bound_data.shape[0]
        return row
