import pandas as pd

from model.Input import Input
from clean.TypeContainer import TypeContainer
from constants import flight_id
from loader.FeatureExtractor import FeatureExtractor


class ArrivalToGateTimeWeightedMeanExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: pd.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        merged_standtimes_runways = pd.merge(input_data.standtimes, input_data.runways, on=flight_id)
        merged_standtimes_runways['diff_stand_runway'] = (
                (merged_standtimes_runways['arrival_stand_actual_time']
                 - merged_standtimes_runways['arrival_runway_actual_time']
                 ).dt.total_seconds() / 60)
        diff_stand_runway = merged_standtimes_runways[merged_standtimes_runways['diff_stand_runway'] >= 0]
        ewm = pd.Series(diff_stand_runway['diff_stand_runway']).ewm(span=2).mean().iloc[-1]
        data['mean_runway_pushback'] = ewm
        return data
