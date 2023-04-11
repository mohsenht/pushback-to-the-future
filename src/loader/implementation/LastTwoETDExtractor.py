import cudf

from src.model.Input import Input
from src.clean.TypeContainer import TypeContainer
from src.constants import FLIGHT_ID, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME
from src.loader.FeatureExtractor import FeatureExtractor


class LastTwoETDExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: cudf.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> cudf.DataFrame:
        now_etd = input_data.etd[input_data.etd[FLIGHT_ID].isin(data[FLIGHT_ID])]
        diff_etd = cudf.DataFrame(0, index=now_etd.index, columns=['diff_etd'])
        diff_etd = cudf.concat([now_etd, diff_etd], axis=1)
        diff_etd['diff_etd'] = (now_etd[ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME] - now).dt.total_seconds()
        latest_now_etd = diff_etd.groupby(FLIGHT_ID).apply(self.last_two_diff).reset_index(drop=True)
        if len(latest_now_etd) == 0:
            data["diff_two_etd"] = 0
        etd = data.merge(
            latest_now_etd, how="left", on=FLIGHT_ID
        ).diff_two_etd
        data["diff_two_etd"] = etd / 60
        return data

    def last_two_diff(self, x):
        if len(x) < 2:
            last = x.iloc[-1]
            last['diff_two_etd'] = 0
            return last
        last = x.iloc[-1]
        second_last = x.iloc[-2]

        last['diff_two_etd'] = last["diff_etd"] - second_last["diff_etd"]

        return last
