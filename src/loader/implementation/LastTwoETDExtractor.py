import cudf
import pandas as pd

from src.clean.TypeContainer import TypeContainer
from src.constants import FLIGHT_ID, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME
from src.loader.FeatureExtractor import FeatureExtractor
from src.model.Input import Input


def get_difference(row):
    second_last_etd = row['second_last_etd']
    last_etd = row['last_etd']
    if row['second_last_etd'] is cudf.NA:
        return 0
    if row['last_etd'] is cudf.NA:
        return 0
    return int(last_etd - second_last_etd)


class LastTwoETDExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: cudf.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> cudf.DataFrame:
        now_etd = input_data.etd[input_data.etd[FLIGHT_ID].isin(data[FLIGHT_ID])]
        now_etd['second_last_etd'] = (now_etd[ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME] - now).dt.seconds
        latest_now_etd = now_etd.groupby(FLIGHT_ID).nth(-2)

        if len(latest_now_etd) == 0:
            data["diff_two_etd"] = 0
        latest_now_etd = latest_now_etd.reset_index()
        latest_now_etd['second_last_etd'] = latest_now_etd['second_last_etd'] / 60
        latest_now_etd = latest_now_etd[[FLIGHT_ID, 'second_last_etd']]
        data = data.merge(
            latest_now_etd,
            how="left",
            on=FLIGHT_ID
        )
        data['diff_two_etd'] = data.apply(func=get_difference, axis=1)
        data = data.drop(columns=['second_last_etd'], axis='columns')
        return data
