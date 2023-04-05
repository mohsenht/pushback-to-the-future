import pandas as pd

from Input import Input
from clean.extract.TypeContainer import TypeContainer
from constants import flight_id
from loader.DataLoader import DataLoader


class LastTwoETDLoader(DataLoader):

    def load_data(self, now: pd.Timestamp, data: pd.DataFrame, input: Input, type_container: TypeContainer) -> pd.DataFrame:
        now_etd = input.etd[input.etd[flight_id].isin(data[flight_id])]
        diff_etd = pd.DataFrame(0, index=now_etd.index, columns=['diff_etd'])
        diff_etd = pd.concat([now_etd, diff_etd], axis=1)
        diff_etd['diff_etd'] = (now_etd['departure_runway_estimated_time'] - now).dt.total_seconds()
        latest_now_etd = diff_etd.groupby(flight_id).apply(self.last_two_diff).reset_index(drop=True)

        etd = data.merge(
            latest_now_etd, how="left", on=flight_id
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
