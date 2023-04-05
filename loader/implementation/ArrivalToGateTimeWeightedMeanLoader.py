import pandas as pd

from Input import Input
from clean.extract.TypeContainer import TypeContainer
from constants import flight_id
from loader.DataLoader import DataLoader


class ArrivalToGateTimeWeightedMeanLoader(DataLoader):

    def load_data(self, now: pd.Timestamp, data: pd.DataFrame, input: Input, type_container: TypeContainer) -> pd.DataFrame:
        merged_standtimes_runways = pd.merge(input.standtimes, input.runways, on=flight_id)
        merged_standtimes_runways['diff_stand_runway'] = (
                (merged_standtimes_runways['arrival_stand_actual_time']
                 - merged_standtimes_runways['arrival_runway_actual_time']
                 ).dt.total_seconds() / 60)
        diff_stand_runway = merged_standtimes_runways[merged_standtimes_runways['diff_stand_runway'] >= 0]
        ewm = pd.Series(diff_stand_runway['diff_stand_runway']).ewm(span=2).mean().iloc[-1]
        data['mean_runway_pushback'] = ewm
        return data
