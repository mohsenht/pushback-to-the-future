import pandas as pd

from Input import Input
from clean.extract.TypeContainer import TypeContainer
from constants import flight_id
from loader.DataLoader import DataLoader


class WeightedETDLoader(DataLoader):

    def load_data(self, now: pd.Timestamp, data: pd.DataFrame, input: Input, type_container: TypeContainer) -> pd.DataFrame:
        w_etd = pd.DataFrame(0, index=input.etd.index, columns=['w_etd'])
        w_etd['w_etd'] = (input.etd['departure_runway_estimated_time'] - now).dt.total_seconds()
        now_etd = pd.concat([input.etd, w_etd], axis=1)
        latest_now_etd = now_etd.groupby(flight_id)['w_etd'].ewm(span=2).mean()
        etd = data.merge(
            latest_now_etd, how="left", on=flight_id
        ).w_etd
        data["w_etd"] = etd / 60
        return data
