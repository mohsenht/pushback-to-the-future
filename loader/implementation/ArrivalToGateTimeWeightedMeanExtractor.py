import pandas as pd

from model.Input import Input
from clean.TypeContainer import TypeContainer
from constants import FLIGHT_ID, FEATURE_COLUMN_WEIGHTED_MEAN_ARRIVAL_TO_GATE
from loader.FeatureExtractor import FeatureExtractor


class ArrivalToGateTimeWeightedMeanExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: pd.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        merged_standtimes_runways = pd.merge(input_data.standtimes, input_data.runways, on=FLIGHT_ID)
        merged_standtimes_runways = merged_standtimes_runways.dropna(subset=['arrival_stand_actual_time', 'arrival_runway_actual_time'])
        merged_standtimes_runways['diff_stand_runway'] = (
                (merged_standtimes_runways['arrival_stand_actual_time']
                 - merged_standtimes_runways['arrival_runway_actual_time']
                 ).dt.total_seconds() / 60)
        merged_standtimes_runways = merged_standtimes_runways.sort_values('arrival_stand_actual_time')
        diff_stand_runway = merged_standtimes_runways[merged_standtimes_runways['diff_stand_runway'] >= 0]
        ewm = pd.Series(diff_stand_runway['diff_stand_runway']).ewm(span=4).mean()
        if ewm.empty:
            weighted_mean = 10
        else:
            weighted_mean = ewm.iloc[-1]
        data[FEATURE_COLUMN_WEIGHTED_MEAN_ARRIVAL_TO_GATE] = weighted_mean
        return data
