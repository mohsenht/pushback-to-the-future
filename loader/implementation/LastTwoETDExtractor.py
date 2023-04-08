import pandas as pd

from model.Input import Input
from clean.TypeContainer import TypeContainer
from constants import FLIGHT_ID, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME
from loader.FeatureExtractor import FeatureExtractor


class LastTwoETDExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: pd.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        now_etd = input_data.etd[input_data.etd[FLIGHT_ID].isin(data[FLIGHT_ID])]
        diff_etd = pd.DataFrame(0, index=now_etd.index, columns=['diff_etd'])
        diff_etd = pd.concat([now_etd, diff_etd], axis=1)
        diff_etd['diff_etd'] = (now_etd[ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME] - now).dt.total_seconds()
        latest_now_etd = diff_etd.groupby(FLIGHT_ID).apply(self.last_two_diff).reset_index(drop=True)

        etd = data.merge(
            latest_now_etd, how="left", on=FLIGHT_ID
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
