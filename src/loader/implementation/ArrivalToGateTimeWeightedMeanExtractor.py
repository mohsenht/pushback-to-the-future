import cudf
import pandas as pd

from src.hyper_parameters import ARRIVAL_TO_GATE_WEIGHTED_MEAN_TIME_CONSTANT_FOR_EMPTY_ARRIVAL
from src.model.Input import Input
from src.clean.TypeContainer import TypeContainer
from src.constants import FLIGHT_ID, FEATURE_COLUMN_WEIGHTED_MEAN_ARRIVAL_TO_GATE, \
    RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME, STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME
from src.loader.FeatureExtractor import FeatureExtractor


class ArrivalToGateTimeWeightedMeanExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: cudf.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> cudf.DataFrame:
        merged_standtimes_runways = cudf.merge(input_data.standtimes, input_data.runways, on=FLIGHT_ID)
        weighted_mean = ARRIVAL_TO_GATE_WEIGHTED_MEAN_TIME_CONSTANT_FOR_EMPTY_ARRIVAL
        if not merged_standtimes_runways.empty:
            merged_standtimes_runways = merged_standtimes_runways.dropna(subset=[STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME, RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME])
            merged_standtimes_runways['diff_stand_runway'] = (
                    (merged_standtimes_runways[STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME]
                     - merged_standtimes_runways[RUNWAYS_COLUMN_ARRIVAL_RUNWAY_ACTUAL_TIME]
                     ).dt.seconds / 60)
            merged_standtimes_runways = merged_standtimes_runways.sort_values(STANDTIMES_COLUMN_ARRIVAL_STAND_ACTUAL_TIME)
            diff_stand_runway = merged_standtimes_runways[merged_standtimes_runways['diff_stand_runway'] >= 0]
            pandas_diff_stand_runway = diff_stand_runway['diff_stand_runway'].to_pandas()
            ewm = pd.Series(pandas_diff_stand_runway).ewm(span=4).mean()
            if not ewm.empty:
                weighted_mean = ewm.iloc[-1]
        data[FEATURE_COLUMN_WEIGHTED_MEAN_ARRIVAL_TO_GATE] = weighted_mean
        return data
