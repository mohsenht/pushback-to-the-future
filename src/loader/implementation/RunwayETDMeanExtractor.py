import cudf
import pandas as pd

from src.model.Input import Input
from src.constants import FLIGHT_ID, RUNWAYS_COLUMN_DEPARTURE_RUNWAY_ACTUAL, ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME
from src.loader.FeatureExtractor import FeatureExtractor


class TypeContainer:
    pass


class RunwayETDMeanExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: cudf.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> cudf.DataFrame:
        now_runways = input_data.runways.dropna(subset=[RUNWAYS_COLUMN_DEPARTURE_RUNWAY_ACTUAL])
        now_etd = input_data.etd[input_data.etd[FLIGHT_ID].isin(now_runways[FLIGHT_ID])]
        grouped_now_etd = now_etd.groupby(FLIGHT_ID).last()
        results = grouped_now_etd.apply(self.find_etd_for_each_gufi, arg1=now_runways)

        df = cudf.DataFrame(results.to_list(), columns=[FLIGHT_ID, 'etd_new'])
        merged_df = now_runways.merge(df[[FLIGHT_ID, 'etd_new']], on=FLIGHT_ID, how='left')

        data['etd_new'] = merged_df.dropna(subset=['etd_new'])['etd_new'].mean()

        return data

    def find_etd_for_each_gufi(self, group, arg1):
        runway = arg1.loc[group.name == arg1.gufi]
        if runway.empty:
            return
        runway_unique = runway.iloc[0]
        etd = group.loc[runway_unique[RUNWAYS_COLUMN_DEPARTURE_RUNWAY_ACTUAL] > group.timestamp]
        if etd.empty:
            return
        return runway_unique[FLIGHT_ID], runway_unique[RUNWAYS_COLUMN_DEPARTURE_RUNWAY_ACTUAL] - etd.iloc[-1][
            ETD_COLUMN_DEPARTURE_RUNWAY_ESTIMATED_TIME]
