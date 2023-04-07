import pandas as pd

from model.Input import Input
from constants import FLIGHT_ID
from loader.FeatureExtractor import FeatureExtractor


class TypeContainer:
    pass


class RunwayETDMeanExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: pd.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        now_runways = input_data.runways.dropna(subset=['departure_runway_actual_time'])
        now_etd = input_data.etd[input_data.etd[FLIGHT_ID].isin(now_runways[FLIGHT_ID])]
        grouped_now_etd = now_etd.groupby(FLIGHT_ID).last()
        results = grouped_now_etd.apply(self.find_etd_for_each_gufi, arg1=now_runways)

        df = pd.DataFrame(results.to_list(), columns=[FLIGHT_ID, 'etd_new'])
        merged_df = now_runways.merge(df[[FLIGHT_ID, 'etd_new']], on=FLIGHT_ID, how='left')

        data['etd_new'] = merged_df.dropna(subset=['etd_new'])['etd_new'].mean()

        return data

    def find_etd_for_each_gufi(self, group, arg1):
        runway = arg1.loc[group.name == arg1.gufi]
        if runway.empty:
            return
        runway_unique = runway.iloc[0]
        etd = group.loc[runway_unique['departure_runway_actual_time'] > group.timestamp]
        if etd.empty:
            return
        return runway_unique[FLIGHT_ID], runway_unique['departure_runway_actual_time'] - etd.iloc[-1][
            'departure_runway_estimated_time']
