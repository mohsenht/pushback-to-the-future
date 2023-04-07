import pandas as pd

from model.Input import Input
from clean.TypeContainer import TypeContainer
from constants import RUNWAYS_COLUMN_DEPARTURE_RUNWAYS, RUNWAYS_COLUMN_ARRIVAL_RUNWAYS
from loader.FeatureExtractor import FeatureExtractor


class RunningRunwayInfoExtractor(FeatureExtractor):

    def load_data(self,
                  now: pd.Timestamp,
                  data: pd.DataFrame,
                  input_data: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        now_running_runway = input_data.config.iloc[-1]
        boolean_feature_names = []
        boolean_feature_names.extend(['de_' + s for s in type_container.runways_names])
        boolean_feature_names.extend(['ar_' + s for s in type_container.runways_names])
        new_data = pd.DataFrame(False, index=data.index, columns=boolean_feature_names)
        new_data = pd.concat([data, new_data], axis=1)
        results = new_data.apply(self.fill_runways_for_each_flight, args=(now_running_runway,), axis=1)
        return results

    def fill_runways_for_each_flight(self, x, now_running_runway):
        departure_runways = now_running_runway[RUNWAYS_COLUMN_DEPARTURE_RUNWAYS].split(', ')
        arrival_runways = now_running_runway[RUNWAYS_COLUMN_ARRIVAL_RUNWAYS].split(', ')
        for departure_runway in departure_runways:
            x[f"de_{departure_runway}"] = True
        for arrival_runway in arrival_runways:
            x[f"ar_{arrival_runway}"] = True
        return x
