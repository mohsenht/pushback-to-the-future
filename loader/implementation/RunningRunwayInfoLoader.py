import pandas as pd

from Input import Input
from clean.extract.TypeContainer import TypeContainer
from constants import runways_column_departure_runways, runways_column_arrival_runways
from loader.DataLoader import DataLoader


class RunningRunwayInfoLoader(DataLoader):

    def load_data(self, now: pd.Timestamp, data: pd.DataFrame, input: Input,
                  type_container: TypeContainer) -> pd.DataFrame:
        now_running_runway = input.config.iloc[-1]
        boolean_feature_names = []
        boolean_feature_names.extend(['de_' + s for s in type_container.runways_names])
        boolean_feature_names.extend(['ar_' + s for s in type_container.runways_names])
        new_data = pd.DataFrame(False, index=data.index, columns=boolean_feature_names)
        new_data = pd.concat([data, new_data], axis=1)
        results = new_data.apply(self.fill_runways_for_each_flight, args=(now_running_runway,), axis=1)
        return results

    def fill_runways_for_each_flight(self, x, now_running_runway):
        departure_runways = now_running_runway[runways_column_departure_runways].split(', ')
        arrival_runways = now_running_runway[runways_column_arrival_runways].split(', ')
        for departure_runway in departure_runways:
            x[f"de_{departure_runway}"] = True
        for arrival_runway in arrival_runways:
            x[f"ar_{arrival_runway}"] = True
        return x
