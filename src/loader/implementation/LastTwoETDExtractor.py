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
    return last_etd - second_last_etd


class LastTwoETDExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: cudf.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> cudf.DataFrame:
        now_etd = input_data.etd[input_data.etd[FLIGHT_ID].isin(data[FLIGHT_ID])]
        diff_etd = cudf.DataFrame(index=now_etd.index, columns=['diff_etd']).fillna(0)
        diff_etd = cudf.concat([now_etd, diff_etd], axis=1)
        diff_etd['diff_etd'] = (now_etd[ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME] - now).dt.seconds
        latest_now_etd = diff_etd.groupby(FLIGHT_ID).nth(-2)
        latest_now_etd = latest_now_etd.reset_index()
        data['second_last_etd'] = (data.merge(
            latest_now_etd,
            how="left",
            on=FLIGHT_ID
        ).diff_etd.fillna(0) / 60)

        if len(latest_now_etd) == 0:
            data["diff_two_etd"] = 0
        data['diff_two_etd'] = data.apply(func=get_difference, axis=1)
        etd = data.merge(
            latest_now_etd,
            how="left",
            on=FLIGHT_ID
        ).diff_two_etd
        data["diff_two_etd"] = etd
        data.drop(columns=['second_last_etd'], axis='columns')
        return data
