import pandas as pd

from model.Input import Input
from constants import flight_id
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
        now_etd = input_data.etd[input_data.etd[flight_id].isin(now_runways[flight_id])]
        grouped_now_etd = now_etd.groupby(flight_id).last()
        results = grouped_now_etd.apply(self.find_etd_for_each_gufi, arg1=now_runways)

        df = pd.DataFrame(results.to_list(), columns=[flight_id, 'etd_new'])
        merged_df = now_runways.merge(df[[flight_id, 'etd_new']], on=flight_id, how='left')

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
        return runway_unique[flight_id], runway_unique['departure_runway_actual_time'] - etd.iloc[-1][
            'departure_runway_estimated_time']
