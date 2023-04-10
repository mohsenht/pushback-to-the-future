from pandas import DataFrame, Timestamp

from src.model.Input import Input
from src.clean.TypeContainer import TypeContainer
from src.constants import FLIGHT_ID
from src.loader.FeatureExtractor import FeatureExtractor


class LastETDExtractor(FeatureExtractor):

    def load_data(self,
                  now: Timestamp,
                  data: DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> DataFrame:
        latest_now_etd = input_data.etd.groupby(FLIGHT_ID).last().departure_runway_estimated_time
        departure_runway_estimated_time = data.merge(
            latest_now_etd,
            how="left",
            on=FLIGHT_ID
        ).departure_runway_estimated_time
        departure_runway_estimated_time.fillna(now, inplace=True)
        data["last_etd"] = (departure_runway_estimated_time - now).dt.total_seconds() / 60
        data["last_etd"] = data.last_etd.clip(lower=0).astype(int)
        return data
