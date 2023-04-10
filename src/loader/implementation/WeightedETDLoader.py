import pandas as pd

from src.model.Input import Input
from src.clean.TypeContainer import TypeContainer
from src.constants import FLIGHT_ID
from src.loader.FeatureExtractor import FeatureExtractor


class WeightedETDLoader(FeatureExtractor):

    def load_data(self, now: pd.Timestamp, data: pd.DataFrame, input_data: Input, type_container: TypeContainer) -> pd.DataFrame:
        w_etd = pd.DataFrame(0, index=input_data.etd.index, columns=['w_etd'])
        w_etd['w_etd'] = (input_data.etd['departure_runway_estimated_time'] - now).dt.total_seconds()
        now_etd = pd.concat([input_data.etd, w_etd], axis=1)
        latest_now_etd = now_etd.groupby(FLIGHT_ID)['w_etd'].ewm(span=2).mean()
        etd = data.merge(
            latest_now_etd, how="left", on=FLIGHT_ID
        ).w_etd
        data["w_etd"] = etd / 60
        return data
